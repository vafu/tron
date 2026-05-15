use std::time::Instant;

use anyhow::Result;
use tron_api::{
    GestureFrame, HandGesture, NoContext, PalmPose2d, Point2d, Processor, RoiResult, Size,
};

use crate::roi::mediapipe::HandLandmarks;

const THUMB_TIP: usize = 4;
const INDEX_TIP: usize = 8;
const PINCH_DISTANCE_OF_PALM: f64 = 0.1;

#[derive(Clone, Copy, Debug)]
pub struct GesturePreprocessorInput<'a> {
    pub landmarks: Option<&'a HandLandmarks>,
    pub palm_roi: Option<RoiResult>,
    pub frame_size: Size,
    pub timestamp: Instant,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GesturePreprocessor;

impl Processor<GesturePreprocessorInput<'_>, NoContext> for GesturePreprocessor {
    type Output = GestureFrame;

    fn process(
        &mut self,
        input: GesturePreprocessorInput<'_>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        let palm = input.palm_roi.map(|roi| palm_pose(roi, input.frame_size));
        let gesture = match input.landmarks {
            Some(landmarks) => classify_gesture(landmarks, palm, input.frame_size),
            None => HandGesture::NoHand,
        };
        let output = GestureFrame {
            timestamp: input.timestamp,
            palm,
            gesture,
        };
        trace_gesture_output(&output);
        Ok(output)
    }
}

fn trace_gesture_output(output: &GestureFrame) {
    let palm_center = output.palm.map(|palm| palm.center);
    let palm_extent = output.palm.map(|palm| palm.extent);
    match output.gesture {
        HandGesture::Pinch { strength, position } => {
            tracing::debug!(
                target: "tron_core::gesture",
                gesture = "pinch",
                pinch_strength = strength,
                pinch_x = position.x,
                pinch_y = position.y,
                palm_center_x = palm_center.map(|point| point.x),
                palm_center_y = palm_center.map(|point| point.y),
                palm_extent_x = palm_extent.map(|point| point.x),
                palm_extent_y = palm_extent.map(|point| point.y),
                ?output.timestamp,
                "gesture preprocessor output"
            );
        }
        HandGesture::Pointing => {
            tracing::debug!(
                target: "tron_core::gesture",
                gesture = "pointing",
                palm_center_x = palm_center.map(|point| point.x),
                palm_center_y = palm_center.map(|point| point.y),
                palm_extent_x = palm_extent.map(|point| point.x),
                palm_extent_y = palm_extent.map(|point| point.y),
                ?output.timestamp,
                "gesture preprocessor output"
            );
        }
        _ => {}
    }
}

fn classify_gesture(
    landmarks: &HandLandmarks,
    palm: Option<PalmPose2d>,
    frame_size: Size,
) -> HandGesture {
    let Some(thumb) = landmark_point(landmarks, THUMB_TIP, frame_size) else {
        return HandGesture::Unknown;
    };
    let Some(index) = landmark_point(landmarks, INDEX_TIP, frame_size) else {
        return HandGesture::Unknown;
    };
    let palm_extent = palm
        .map(|palm| palm.extent.x.max(palm.extent.y))
        .unwrap_or(1.0)
        .max(0.001);
    let distance = thumb.distance(index);
    let normalized_distance = distance / palm_extent;
    if normalized_distance <= PINCH_DISTANCE_OF_PALM {
        let strength = (1.0 - normalized_distance / PINCH_DISTANCE_OF_PALM).clamp(0.0, 1.0) as f32;
        return HandGesture::Pinch {
            strength,
            position: (thumb + index) * 0.5,
        };
    }
    HandGesture::OpenPalm
}

fn landmark_point(landmarks: &HandLandmarks, index: usize, frame_size: Size) -> Option<Point2d> {
    let point = landmarks.points[index];
    (point.x.is_finite() && point.y.is_finite()).then(|| {
        Point2d::new(
            point.x as f64 / frame_size.width.max(1) as f64,
            point.y as f64 / frame_size.height.max(1) as f64,
        )
    })
}

fn palm_pose(roi: RoiResult, frame_size: Size) -> PalmPose2d {
    let center = if let Some(oriented_box) = roi.oriented_box {
        let mut center = Point2d::ZERO;
        for [x, y] in oriented_box.corners {
            center += Point2d::new(x as f64, y as f64);
        }
        center / 4.0
    } else {
        Point2d::new(
            roi.rect.x as f64 + roi.rect.size.width as f64 * 0.5,
            roi.rect.y as f64 + roi.rect.size.height as f64 * 0.5,
        )
    };
    let extent = Point2d::new(
        roi.rect.size.width as f64 / frame_size.width.max(1) as f64,
        roi.rect.size.height as f64 / frame_size.height.max(1) as f64,
    );
    PalmPose2d {
        center: Point2d::new(
            center.x / frame_size.width.max(1) as f64,
            center.y / frame_size.height.max(1) as f64,
        ),
        rotation_radians: roi
            .oriented_box
            .map(|oriented_box| {
                let [x0, y0] = oriented_box.corners[0];
                let [x1, y1] = oriented_box.corners[1];
                f64::from(y1 - y0).atan2(f64::from(x1 - x0))
            })
            .unwrap_or(0.0),
        extent,
    }
}

#[cfg(test)]
mod tests {
    use tron_api::{Rect, Size};

    use super::*;
    use crate::roi::mediapipe::HandLandmark;

    fn landmarks(thumb: (f32, f32), index: (f32, f32)) -> HandLandmarks {
        let mut landmarks = HandLandmarks {
            points: [HandLandmark {
                x: f32::NAN,
                y: f32::NAN,
                z: f32::NAN,
            }; 21],
            presence: 1.0,
            handedness: None,
            timestamp: Instant::now(),
        };
        landmarks.points[THUMB_TIP] = HandLandmark {
            x: thumb.0,
            y: thumb.1,
            z: 0.0,
        };
        landmarks.points[INDEX_TIP] = HandLandmark {
            x: index.0,
            y: index.1,
            z: 0.0,
        };
        landmarks
    }

    #[test]
    fn detects_pinch_from_thumb_index_distance() {
        let mut processor = GesturePreprocessor;
        let output = processor
            .process(
                GesturePreprocessorInput {
                    landmarks: Some(&landmarks((100.0, 100.0), (105.0, 100.0))),
                    palm_roi: Some(RoiResult {
                        rect: Rect {
                            x: 80,
                            y: 80,
                            size: Size {
                                width: 80,
                                height: 80,
                            },
                        },
                        oriented_box: None,
                    }),
                    frame_size: Size {
                        width: 640,
                        height: 480,
                    },
                    timestamp: Instant::now(),
                },
                NoContext,
            )
            .unwrap();
        let HandGesture::Pinch { position, .. } = output.gesture else {
            panic!("expected pinch");
        };
        assert!(position.x > 0.15);
        assert!(position.x < 0.17);
    }
}
