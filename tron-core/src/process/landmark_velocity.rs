use std::time::Instant;

use anyhow::Result;
use tron_api::{NoContext, Processor};

use crate::roi::mediapipe::{HandLandmark, HandLandmarks};

const HAND_LANDMARKS: usize = 21;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct HandLandmarkVelocity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl HandLandmarkVelocity {
    fn nan() -> Self {
        Self {
            x: f32::NAN,
            y: f32::NAN,
            z: f32::NAN,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct HandLandmarkMotion {
    pub landmarks: HandLandmarks,
    pub velocities: [HandLandmarkVelocity; HAND_LANDMARKS],
    #[serde(skip)]
    pub timestamp: Instant,
}

#[derive(Clone, Debug, Default)]
pub struct LandmarkVelocityProcessor {
    previous: Option<HandLandmarks>,
}

impl LandmarkVelocityProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    fn reset(&mut self) {
        self.previous = None;
    }
}

impl Processor<Option<HandLandmarks>, NoContext> for LandmarkVelocityProcessor {
    type Output = Option<HandLandmarkMotion>;

    fn process(
        &mut self,
        input: Option<HandLandmarks>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        let Some(landmarks) = input else {
            self.reset();
            return Ok(None);
        };

        let velocities = self
            .previous
            .as_ref()
            .and_then(|previous| landmark_velocities(previous, &landmarks));
        self.previous = Some(landmarks.clone());

        Ok(Some(HandLandmarkMotion {
            timestamp: landmarks.timestamp,
            landmarks,
            velocities: velocities.unwrap_or([HandLandmarkVelocity::default(); HAND_LANDMARKS]),
        }))
    }
}

fn landmark_velocities(
    previous: &HandLandmarks,
    current: &HandLandmarks,
) -> Option<[HandLandmarkVelocity; HAND_LANDMARKS]> {
    let dt = current
        .timestamp
        .checked_duration_since(previous.timestamp)?
        .as_secs_f32();
    if dt <= 0.0 || !dt.is_finite() {
        return None;
    }

    let mut velocities = [HandLandmarkVelocity::nan(); HAND_LANDMARKS];
    for ((velocity, previous), current) in velocities
        .iter_mut()
        .zip(previous.points.iter())
        .zip(current.points.iter())
    {
        *velocity = landmark_velocity(*previous, *current, dt);
    }
    Some(velocities)
}

fn landmark_velocity(
    previous: HandLandmark,
    current: HandLandmark,
    dt: f32,
) -> HandLandmarkVelocity {
    if !previous.x.is_finite()
        || !previous.y.is_finite()
        || !previous.z.is_finite()
        || !current.x.is_finite()
        || !current.y.is_finite()
        || !current.z.is_finite()
    {
        return HandLandmarkVelocity::nan();
    }

    HandLandmarkVelocity {
        x: (current.x - previous.x) / dt,
        y: (current.y - previous.y) / dt,
        z: (current.z - previous.z) / dt,
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn landmarks(point: HandLandmark, timestamp: Instant) -> HandLandmarks {
        let mut points = [HandLandmark::default(); HAND_LANDMARKS];
        points[0] = point;
        HandLandmarks {
            points,
            presence: 1.0,
            handedness: None,
            timestamp,
        }
    }

    #[test]
    fn first_frame_has_zero_velocity() {
        let mut processor = LandmarkVelocityProcessor::new();
        let output = processor
            .process(
                Some(landmarks(
                    HandLandmark {
                        x: 10.0,
                        y: 20.0,
                        z: 0.5,
                    },
                    Instant::now(),
                )),
                NoContext,
            )
            .unwrap()
            .unwrap();

        assert_eq!(output.velocities[0].x, 0.0);
        assert_eq!(output.velocities[0].y, 0.0);
        assert_eq!(output.velocities[0].z, 0.0);
    }

    #[test]
    fn computes_per_second_velocity() {
        let mut processor = LandmarkVelocityProcessor::new();
        let timestamp = Instant::now();
        processor
            .process(
                Some(landmarks(
                    HandLandmark {
                        x: 10.0,
                        y: 20.0,
                        z: 0.5,
                    },
                    timestamp,
                )),
                NoContext,
            )
            .unwrap();

        let output = processor
            .process(
                Some(landmarks(
                    HandLandmark {
                        x: 14.0,
                        y: 12.0,
                        z: 1.5,
                    },
                    timestamp + Duration::from_millis(100),
                )),
                NoContext,
            )
            .unwrap()
            .unwrap();

        assert_eq!(output.velocities[0].x, 40.0);
        assert_eq!(output.velocities[0].y, -80.0);
        assert_eq!(output.velocities[0].z, 10.0);
    }

    #[test]
    fn resets_after_missing_landmarks() {
        let mut processor = LandmarkVelocityProcessor::new();
        let timestamp = Instant::now();
        processor
            .process(
                Some(landmarks(HandLandmark::default(), timestamp)),
                NoContext,
            )
            .unwrap();
        assert!(processor.process(None, NoContext).unwrap().is_none());

        let output = processor
            .process(
                Some(landmarks(
                    HandLandmark {
                        x: 100.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    timestamp + Duration::from_millis(100),
                )),
                NoContext,
            )
            .unwrap()
            .unwrap();

        assert_eq!(output.velocities[0].x, 0.0);
    }
}
