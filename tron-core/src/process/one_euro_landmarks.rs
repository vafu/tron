use anyhow::Result;
use tron_api::{NoContext, Processor};

use crate::filter::{OneEuroConfig, OneEuroScalar};
use crate::roi::mediapipe::{HandLandmark, HandLandmarks};

const HAND_LANDMARKS: usize = 21;
const DEFAULT_FRAME_INTERVAL_SECONDS: f64 = 1.0 / 60.0;

#[derive(Clone, Copy, Debug)]
pub struct OneEuroLandmarkConfig {
    pub min_cutoff: f64,
    pub beta: f64,
    pub derivative_cutoff: f64,
}

impl Default for OneEuroLandmarkConfig {
    fn default() -> Self {
        Self {
            min_cutoff: 1.0,
            beta: 0.04,
            derivative_cutoff: 1.0,
        }
    }
}

impl From<OneEuroLandmarkConfig> for OneEuroConfig {
    fn from(config: OneEuroLandmarkConfig) -> Self {
        Self {
            min_cutoff: config.min_cutoff,
            beta: config.beta,
            derivative_cutoff: config.derivative_cutoff,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OneEuroLandmarkFilter {
    config: OneEuroLandmarkConfig,
    state: [LandmarkFilterState; HAND_LANDMARKS],
    previous_timestamp: Option<std::time::Instant>,
}

impl OneEuroLandmarkFilter {
    pub fn new(config: OneEuroLandmarkConfig) -> Self {
        Self {
            config,
            state: [LandmarkFilterState::default(); HAND_LANDMARKS],
            previous_timestamp: None,
        }
    }

    pub fn config(&self) -> OneEuroLandmarkConfig {
        self.config
    }

    fn reset(&mut self) {
        self.state = [LandmarkFilterState::default(); HAND_LANDMARKS];
        self.previous_timestamp = None;
    }
}

impl Default for OneEuroLandmarkFilter {
    fn default() -> Self {
        Self::new(OneEuroLandmarkConfig::default())
    }
}

impl Processor<Option<HandLandmarks>, NoContext> for OneEuroLandmarkFilter {
    type Output = Option<HandLandmarks>;

    fn process(
        &mut self,
        input: Option<HandLandmarks>,
        _context: NoContext,
    ) -> Result<Self::Output> {
        let Some(mut landmarks) = input else {
            self.reset();
            return Ok(None);
        };

        let dt = self
            .previous_timestamp
            .and_then(|previous| landmarks.timestamp.checked_duration_since(previous))
            .map(|duration| duration.as_secs_f64())
            .filter(|seconds| *seconds > 0.0 && seconds.is_finite())
            .unwrap_or(DEFAULT_FRAME_INTERVAL_SECONDS);
        self.previous_timestamp = Some(landmarks.timestamp);

        for (point, state) in landmarks.points.iter_mut().zip(self.state.iter_mut()) {
            *point = state.filter(*point, dt, self.config);
        }

        Ok(Some(landmarks))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LandmarkFilterState {
    x: OneEuroScalar,
    y: OneEuroScalar,
    z: OneEuroScalar,
}

impl LandmarkFilterState {
    fn filter(
        &mut self,
        point: HandLandmark,
        dt: f64,
        config: OneEuroLandmarkConfig,
    ) -> HandLandmark {
        HandLandmark::new(
            self.x.filter(point.x, dt, config.into()),
            self.y.filter(point.y, dt, config.into()),
            self.z.filter(point.z, dt, config.into()),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn landmarks(x: f32, timestamp: Instant) -> HandLandmarks {
        let mut points = [HandLandmark::default(); HAND_LANDMARKS];
        points[0] = HandLandmark::new(x as f64, 2.0, 0.0);
        HandLandmarks {
            points,
            presence: 1.0,
            handedness: None,
            timestamp,
        }
    }

    #[test]
    fn smooths_landmark_motion() {
        let mut filter = OneEuroLandmarkFilter::new(OneEuroLandmarkConfig {
            min_cutoff: 0.5,
            beta: 0.0,
            derivative_cutoff: 1.0,
        });
        let timestamp = Instant::now();
        let first = filter
            .process(Some(landmarks(0.0, timestamp)), NoContext)
            .unwrap()
            .unwrap();
        assert_eq!(first.points[0].x, 0.0);

        let second = filter
            .process(
                Some(landmarks(100.0, timestamp + Duration::from_millis(16))),
                NoContext,
            )
            .unwrap()
            .unwrap();
        assert!(second.points[0].x > 0.0);
        assert!(second.points[0].x < 100.0);
    }

    #[test]
    fn resets_after_missing_landmarks() {
        let mut filter = OneEuroLandmarkFilter::default();
        let timestamp = Instant::now();
        filter
            .process(Some(landmarks(0.0, timestamp)), NoContext)
            .unwrap();
        assert!(filter.process(None, NoContext).unwrap().is_none());

        let output = filter
            .process(
                Some(landmarks(100.0, timestamp + Duration::from_millis(32))),
                NoContext,
            )
            .unwrap()
            .unwrap();
        assert_eq!(output.points[0].x, 100.0);
    }
}
