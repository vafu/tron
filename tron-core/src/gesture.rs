use std::time::Instant;

use anyhow::Result;
use tron_api::{
    GestureFrame, GestureSignal, HandGesture, NoContext, PalmPose2d, Point2d, Processor, RoiResult,
    Size,
};

use crate::process::landmark_velocity::{HandLandmarkMotion, HandLandmarkVelocity};
use crate::roi::mediapipe::HandLandmarks;

const THUMB_TIP: usize = 4;
const WRIST: usize = 0;
const INDEX_MCP: usize = 5;
const INDEX_TIP: usize = 8;
const MIDDLE_MCP: usize = 9;
const MIDDLE_PIP: usize = 10;
const MIDDLE_TIP: usize = 12;
const RING_MCP: usize = 13;
const RING_PIP: usize = 14;
const RING_TIP: usize = 16;
const PINKY_MCP: usize = 17;
const PINKY_PIP: usize = 18;
const PINKY_TIP: usize = 20;
const PINCH_DISTANCE_OF_PALM: f64 = 0.1;
const CLUTCH_FOLDED_STRENGTH: f64 = 0.1;
const PALM_MOTION_LANDMARKS: [usize; 5] = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP];

#[derive(Clone, Copy, Debug)]
pub struct GesturePreprocessorInput<'a> {
    pub landmarks: Option<&'a HandLandmarks>,
    pub motion: Option<&'a HandLandmarkMotion>,
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
        let classification = match input.landmarks {
            Some(landmarks) => classify_gesture(landmarks, input.motion, palm, input.frame_size),
            None => GestureClassification::no_hand(),
        };
        let output = GestureFrame {
            timestamp: input.timestamp,
            palm,
            signals: classification.signals,
            gesture: classification.gesture,
        };
        trace_gesture_output(&output);
        Ok(output)
    }
}

fn trace_gesture_output(output: &GestureFrame) {
    let palm_center = output.palm.map(|palm| palm.center);
    let palm_extent = output.palm.map(|palm| palm.extent);
    for signal in &output.signals {
        tracing::debug!(
            target: "tron_core::gesture",
            gesture = ?signal.gesture,
            gesture_strength = signal.strength,
            gesture_x = signal.position.x,
            gesture_y = signal.position.y,
            palm_center_x = palm_center.map(|point| point.x),
            palm_center_y = palm_center.map(|point| point.y),
            palm_extent_x = palm_extent.map(|point| point.x),
            palm_extent_y = palm_extent.map(|point| point.y),
            ?output.timestamp,
            "gesture preprocessor output"
        );
    }
    match output.gesture {
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

#[derive(Clone, Debug)]
struct GestureClassification {
    signals: Vec<GestureSignal>,
    gesture: HandGesture,
}

impl GestureClassification {
    fn no_hand() -> Self {
        Self {
            signals: Vec::new(),
            gesture: HandGesture::NoHand,
        }
    }
}

fn classify_gesture(
    landmarks: &HandLandmarks,
    motion: Option<&HandLandmarkMotion>,
    palm: Option<PalmPose2d>,
    frame_size: Size,
) -> GestureClassification {
    let Some(thumb) = landmark_point(landmarks, THUMB_TIP, frame_size) else {
        return GestureClassification {
            signals: Vec::new(),
            gesture: HandGesture::Unknown,
        };
    };
    let Some(index) = landmark_point(landmarks, INDEX_TIP, frame_size) else {
        return GestureClassification {
            signals: Vec::new(),
            gesture: HandGesture::Unknown,
        };
    };
    let mut signals = Vec::new();
    if let Some((strength, position)) = clutch(landmarks, palm, frame_size) {
        signals.push(GestureSignal {
            gesture: HandGesture::Clutch,
            strength,
            position,
            velocity: palm_velocity(motion, frame_size),
        });
    }
    let palm_extent = palm
        .map(|palm| palm.extent.x.max(palm.extent.y))
        .unwrap_or(1.0)
        .max(0.001);
    let distance = thumb.distance(index);
    let normalized_distance = distance / palm_extent;
    if normalized_distance <= PINCH_DISTANCE_OF_PALM {
        let strength = (1.0 - normalized_distance / PINCH_DISTANCE_OF_PALM).clamp(0.0, 1.0) as f32;
        signals.push(GestureSignal {
            gesture: HandGesture::Pinch,
            strength,
            position: (thumb + index) * 0.5,
            velocity: None,
        });
    }
    let gesture = if signals
        .iter()
        .any(|signal| signal.gesture == HandGesture::Pinch)
    {
        HandGesture::Pinch
    } else if signals
        .iter()
        .any(|signal| signal.gesture == HandGesture::Clutch)
    {
        HandGesture::Clutch
    } else {
        HandGesture::OpenPalm
    };
    GestureClassification { signals, gesture }
}

fn clutch(
    landmarks: &HandLandmarks,
    palm: Option<PalmPose2d>,
    frame_size: Size,
) -> Option<(f32, Point2d)> {
    let wrist = landmark_point(landmarks, WRIST, frame_size)?;
    let middle_mcp = landmark_point(landmarks, MIDDLE_MCP, frame_size)?;
    let palm_axis = middle_mcp - wrist;
    let palm_axis_len = palm_axis.length();
    if palm_axis_len <= f64::EPSILON {
        return None;
    }
    let palm_axis = palm_axis / palm_axis_len;
    let palm_extent = palm
        .map(|palm| palm.extent.x.max(palm.extent.y))
        .unwrap_or(palm_axis_len)
        .max(0.001);

    let middle = folded_strength(
        landmarks, MIDDLE_PIP, MIDDLE_TIP, frame_size, wrist, palm_axis,
    );
    let ring = folded_strength(landmarks, RING_PIP, RING_TIP, frame_size, wrist, palm_axis);
    let pinky = folded_strength(
        landmarks, PINKY_PIP, PINKY_TIP, frame_size, wrist, palm_axis,
    );
    let strength = middle.min(ring).min(pinky);
    if strength < CLUTCH_FOLDED_STRENGTH {
        return None;
    }

    let position = middle_palm_point(wrist, middle_mcp).unwrap_or_else(|| {
        palm.map(|palm| palm.center)
            .unwrap_or(wrist + palm_axis * palm_extent)
    });
    Some((strength as f32, position.clamp(Point2d::ZERO, Point2d::ONE)))
}

fn middle_palm_point(wrist: Point2d, middle_mcp: Point2d) -> Option<Point2d> {
    (wrist.is_finite() && middle_mcp.is_finite()).then_some((wrist + middle_mcp) * 0.5)
}

fn palm_velocity(motion: Option<&HandLandmarkMotion>, frame_size: Size) -> Option<Point2d> {
    let motion = motion?;
    let mut sum = Point2d::ZERO;
    let mut count = 0.0;
    for index in PALM_MOTION_LANDMARKS {
        if let Some(velocity) = normalized_velocity(motion.velocities[index], frame_size) {
            sum += velocity;
            count += 1.0;
        }
    }
    (count > 0.0).then_some(sum / count)
}

fn normalized_velocity(velocity: HandLandmarkVelocity, frame_size: Size) -> Option<Point2d> {
    (velocity.x.is_finite() && velocity.y.is_finite()).then(|| {
        Point2d::new(
            velocity.x / frame_size.width.max(1) as f64,
            velocity.y / frame_size.height.max(1) as f64,
        )
    })
}

fn folded_strength(
    landmarks: &HandLandmarks,
    pip_index: usize,
    tip_index: usize,
    frame_size: Size,
    wrist: Point2d,
    palm_axis: Point2d,
) -> f64 {
    let Some(pip) = landmark_point(landmarks, pip_index, frame_size) else {
        return 0.0;
    };
    let Some(tip) = landmark_point(landmarks, tip_index, frame_size) else {
        return 0.0;
    };
    let pip_projection = (pip - wrist).dot(palm_axis);
    let tip_projection = (tip - wrist).dot(palm_axis);
    ((pip_projection - tip_projection) / pip_projection.abs().max(0.001)).clamp(0.0, 1.0)
}

fn landmark_point(landmarks: &HandLandmarks, index: usize, frame_size: Size) -> Option<Point2d> {
    let point = landmarks.points[index];
    (point.x.is_finite() && point.y.is_finite()).then(|| {
        Point2d::new(
            point.x / frame_size.width.max(1) as f64,
            point.y / frame_size.height.max(1) as f64,
        )
    })
}

fn palm_pose(roi: RoiResult, frame_size: Size) -> PalmPose2d {
    let center = if let Some(oriented_box) = roi.oriented_box {
        let mut center = Point2d::ZERO;
        for corner in oriented_box.corners {
            center += corner.as_dvec2();
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
                let edge = (oriented_box.corners[1] - oriented_box.corners[0]).as_dvec2();
                edge.y.atan2(edge.x)
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
            points: [HandLandmark::splat(f64::NAN); 21],
            presence: 1.0,
            handedness: None,
            timestamp: Instant::now(),
        };
        landmarks.points[THUMB_TIP] = HandLandmark::new(thumb.0 as f64, thumb.1 as f64, 0.0);
        landmarks.points[INDEX_TIP] = HandLandmark::new(index.0 as f64, index.1 as f64, 0.0);
        landmarks
    }

    fn set_landmark(landmarks: &mut HandLandmarks, index: usize, x: f32, y: f32) {
        landmarks.points[index] = HandLandmark::new(x as f64, y as f64, 0.0);
    }

    #[test]
    fn detects_pinch_from_thumb_index_distance() {
        let mut processor = GesturePreprocessor;
        let output = processor
            .process(
                GesturePreprocessorInput {
                    landmarks: Some(&landmarks((100.0, 100.0), (105.0, 100.0))),
                    motion: None,
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
        let Some(signal) = output.signal(HandGesture::Pinch) else {
            panic!("expected pinch signal");
        };
        let position = signal.position;
        assert!(position.x > 0.15);
        assert!(position.x < 0.17);
    }

    #[test]
    fn detects_clutch_from_middle_ring_pinky_folded() {
        let mut landmarks = landmarks((220.0, 240.0), (450.0, 260.0));
        set_landmark(&mut landmarks, 0, 500.0, 700.0);
        set_landmark(&mut landmarks, MIDDLE_MCP, 500.0, 500.0);
        set_landmark(&mut landmarks, MIDDLE_PIP, 500.0, 360.0);
        set_landmark(&mut landmarks, MIDDLE_TIP, 500.0, 650.0);
        set_landmark(&mut landmarks, RING_PIP, 560.0, 390.0);
        set_landmark(&mut landmarks, RING_TIP, 540.0, 650.0);
        set_landmark(&mut landmarks, PINKY_PIP, 620.0, 430.0);
        set_landmark(&mut landmarks, PINKY_TIP, 570.0, 650.0);

        let classification = classify_gesture(
            &landmarks,
            None,
            Some(PalmPose2d {
                center: Point2d::new(0.5, 0.55),
                rotation_radians: 0.0,
                extent: Point2d::splat(0.2),
            }),
            Size {
                width: 1000,
                height: 1000,
            },
        );

        let Some(signal) = classification
            .signals
            .iter()
            .find(|signal| signal.gesture == HandGesture::Clutch)
        else {
            panic!("expected clutch signal");
        };
        assert_eq!(signal.position, Point2d::new(0.5, 0.6));
    }

    #[test]
    fn pinch_and_clutch_can_coexist() {
        let mut landmarks = landmarks((500.0, 600.0), (505.0, 600.0));
        set_landmark(&mut landmarks, 0, 500.0, 700.0);
        set_landmark(&mut landmarks, MIDDLE_MCP, 500.0, 500.0);
        set_landmark(&mut landmarks, MIDDLE_PIP, 500.0, 360.0);
        set_landmark(&mut landmarks, MIDDLE_TIP, 500.0, 650.0);
        set_landmark(&mut landmarks, RING_PIP, 560.0, 390.0);
        set_landmark(&mut landmarks, RING_TIP, 540.0, 650.0);
        set_landmark(&mut landmarks, PINKY_PIP, 620.0, 430.0);
        set_landmark(&mut landmarks, PINKY_TIP, 570.0, 650.0);

        let classification = classify_gesture(
            &landmarks,
            None,
            Some(PalmPose2d {
                center: Point2d::new(0.5, 0.55),
                rotation_radians: 0.0,
                extent: Point2d::splat(0.2),
            }),
            Size {
                width: 1000,
                height: 1000,
            },
        );

        assert!(
            classification
                .signals
                .iter()
                .any(|signal| signal.gesture == HandGesture::Pinch)
        );
        assert!(
            classification
                .signals
                .iter()
                .any(|signal| signal.gesture == HandGesture::Clutch)
        );
        assert_eq!(classification.gesture, HandGesture::Pinch);
    }
}
