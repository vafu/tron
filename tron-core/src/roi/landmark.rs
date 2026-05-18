use anyhow::Result;
use glam::Vec2;
use tron_api::{NoContext, OrientedBoundingBox, Processor, Rect, RoiResult, Size};

use crate::process::landmark_velocity::{HandLandmarkMotion, HandLandmarkVelocity};
use crate::roi::mediapipe::HandLandmarks;

const ROI_MOTION_LANDMARKS: [usize; 12] = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];

#[derive(Clone, Copy, Debug)]
pub struct LandmarkRoiInput<'a> {
    pub landmarks: Option<&'a HandLandmarks>,
    pub frame_size: Size,
}

#[derive(Clone, Copy, Debug)]
pub struct LandmarkVelocityRoiInput<'a> {
    pub roi: Option<RoiResult>,
    pub motion: Option<&'a HandLandmarkMotion>,
    pub frame_size: Size,
}

#[derive(Clone, Copy, Debug)]
pub struct LandmarkRoiProcessor {}

#[derive(Clone, Copy, Debug)]
pub struct LandmarkTrackingRoiProcessor {}

#[derive(Clone, Copy, Debug)]
pub struct LandmarkVelocityRoiProcessor {
    prediction_scale: f32,
}

impl LandmarkRoiProcessor {
    pub fn new() -> Self {
        Self {}
    }
}

impl LandmarkTrackingRoiProcessor {
    pub fn new() -> Self {
        Self {}
    }
}

impl LandmarkVelocityRoiProcessor {
    pub fn new() -> Self {
        Self {
            prediction_scale: 1.0,
        }
    }
}

impl Processor<LandmarkRoiInput<'_>, NoContext> for LandmarkRoiProcessor {
    type Output = Option<RoiResult>;

    fn process(
        &mut self,
        input: LandmarkRoiInput<'_>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        Ok(input
            .landmarks
            .and_then(|landmarks| landmarks.bounding_roi(input.frame_size)))
    }
}

impl Processor<LandmarkVelocityRoiInput<'_>, NoContext> for LandmarkVelocityRoiProcessor {
    type Output = Option<RoiResult>;

    fn process(
        &mut self,
        input: LandmarkVelocityRoiInput<'_>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        let Some(roi) = input.roi else {
            return Ok(None);
        };
        let Some(motion) = input.motion else {
            return Ok(Some(roi));
        };
        let Some([dx, dy]) = average_landmark_displacement(motion, self.prediction_scale) else {
            return Ok(Some(roi));
        };
        Ok(Some(translate_roi(roi, dx, dy, input.frame_size)))
    }
}

fn average_landmark_displacement(
    motion: &HandLandmarkMotion,
    prediction_scale: f32,
) -> Option<[f32; 2]> {
    if motion.dt_secs <= 0.0 || !motion.dt_secs.is_finite() || !prediction_scale.is_finite() {
        return None;
    }

    let mut count = 0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for index in ROI_MOTION_LANDMARKS {
        let velocity = motion.velocities[index];
        if finite_velocity(velocity) {
            dx += velocity.x * f64::from(motion.dt_secs) * f64::from(prediction_scale);
            dy += velocity.y * f64::from(motion.dt_secs) * f64::from(prediction_scale);
            count += 1;
        }
    }
    (count > 0).then_some([(dx / count as f64) as f32, (dy / count as f64) as f32])
}

fn finite_velocity(velocity: HandLandmarkVelocity) -> bool {
    velocity.x.is_finite() && velocity.y.is_finite()
}

fn translate_roi(roi: RoiResult, dx: f32, dy: f32, frame_size: Size) -> RoiResult {
    RoiResult {
        rect: translate_rect(roi.rect, dx, dy, frame_size),
        oriented_box: roi
            .oriented_box
            .map(|oriented_box| translate_oriented_box(oriented_box, dx, dy)),
    }
}

fn translate_rect(rect: Rect, dx: f32, dy: f32, frame_size: Size) -> Rect {
    let x = (rect.x as f32 + dx).round().max(0.0) as u32;
    let y = (rect.y as f32 + dy).round().max(0.0) as u32;
    Rect {
        x,
        y,
        size: rect.size,
    }
    .clamp_to(frame_size)
}

fn translate_oriented_box(
    oriented_box: OrientedBoundingBox,
    dx: f32,
    dy: f32,
) -> OrientedBoundingBox {
    OrientedBoundingBox {
        corners: oriented_box
            .corners
            .map(|corner| corner + Vec2::new(dx, dy)),
    }
}

impl Processor<LandmarkRoiInput<'_>, NoContext> for LandmarkTrackingRoiProcessor {
    type Output = Option<RoiResult>;

    fn process(
        &mut self,
        input: LandmarkRoiInput<'_>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        Ok(input
            .landmarks
            .and_then(|landmarks| landmarks.tracking_roi(input.frame_size)))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::process::landmark_velocity::HandLandmarkVelocity;
    use crate::roi::mediapipe::HandLandmark;

    use super::*;

    fn roi() -> RoiResult {
        RoiResult {
            rect: Rect {
                x: 10,
                y: 20,
                size: Size {
                    width: 50,
                    height: 60,
                },
            },
            oriented_box: Some(OrientedBoundingBox {
                corners: [
                    Vec2::new(10.0, 20.0),
                    Vec2::new(60.0, 20.0),
                    Vec2::new(60.0, 80.0),
                    Vec2::new(10.0, 80.0),
                ],
            }),
        }
    }

    fn motion_with(velocity: HandLandmarkVelocity, dt_secs: f32) -> HandLandmarkMotion {
        let mut velocities = [HandLandmarkVelocity::default(); 21];
        for index in ROI_MOTION_LANDMARKS {
            velocities[index] = velocity;
        }
        let timestamp = Instant::now();
        HandLandmarkMotion {
            landmarks: HandLandmarks {
                points: [HandLandmark::default(); 21],
                presence: 1.0,
                handedness: None,
                timestamp,
            },
            velocities,
            dt_secs,
            timestamp,
        }
    }

    #[test]
    fn velocity_roi_moves_rect_and_oriented_box() {
        let mut processor = LandmarkVelocityRoiProcessor::new();
        let motion = motion_with(
            HandLandmarkVelocity {
                x: 30.0,
                y: -20.0,
                z: 0.0,
            },
            0.1,
        );

        let output = processor
            .process(
                LandmarkVelocityRoiInput {
                    roi: Some(roi()),
                    motion: Some(&motion),
                    frame_size: Size {
                        width: 200,
                        height: 200,
                    },
                },
                NoContext,
            )
            .unwrap()
            .unwrap();

        assert_eq!(output.rect.x, 13);
        assert_eq!(output.rect.y, 18);
        assert_eq!(
            output.oriented_box.unwrap().corners,
            [
                Vec2::new(13.0, 18.0),
                Vec2::new(63.0, 18.0),
                Vec2::new(63.0, 78.0),
                Vec2::new(13.0, 78.0),
            ]
        );
    }

    #[test]
    fn velocity_roi_ignores_fingertip_only_motion() {
        let mut processor = LandmarkVelocityRoiProcessor::new();
        let mut motion = motion_with(HandLandmarkVelocity::default(), 0.1);
        motion.velocities[4] = HandLandmarkVelocity {
            x: 1000.0,
            y: 1000.0,
            z: 0.0,
        };

        let output = processor
            .process(
                LandmarkVelocityRoiInput {
                    roi: Some(roi()),
                    motion: Some(&motion),
                    frame_size: Size {
                        width: 200,
                        height: 200,
                    },
                },
                NoContext,
            )
            .unwrap()
            .unwrap();

        assert_eq!(output.rect, roi().rect);
    }
}
