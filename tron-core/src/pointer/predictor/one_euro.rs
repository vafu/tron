use std::time::Instant;

use tron_api::{Point2d, PointerPredictionInput, PointerPredictor};

use crate::filter::{OneEuroConfig, OneEuroPoint2d};

use super::velocity::DecayingVelocityPointerPredictorConfig;

#[derive(Clone, Copy, Debug)]
pub struct OneEuroVelocityPointerPredictorConfig {
    pub one_euro: OneEuroConfig,
    pub decay: DecayingVelocityPointerPredictorConfig,
}

impl Default for OneEuroVelocityPointerPredictorConfig {
    fn default() -> Self {
        Self {
            one_euro: OneEuroConfig {
                min_cutoff: 5.0,
                beta: 0.08,
                derivative_cutoff: 2.0,
            },
            decay: DecayingVelocityPointerPredictorConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct OneEuroVelocityEstimator {
    config: OneEuroConfig,
    velocity_filter: OneEuroPoint2d,
    latest_position: Option<Point2d>,
    latest_timestamp: Option<Instant>,
    filtered_velocity: Option<Point2d>,
}

impl OneEuroVelocityEstimator {
    pub(super) fn new(config: OneEuroConfig) -> Self {
        Self {
            config,
            velocity_filter: OneEuroPoint2d::default(),
            latest_position: None,
            latest_timestamp: None,
            filtered_velocity: None,
        }
    }

    pub(super) fn estimate(&mut self, input: PointerPredictionInput<'_>) -> Option<Point2d> {
        let latest = input.history.last()?;
        if self.latest_timestamp == Some(latest.timestamp) {
            return self.filtered_velocity;
        }

        let previous_position = self.latest_position;
        let previous_timestamp = self.latest_timestamp;
        self.latest_position = Some(latest.position);
        self.latest_timestamp = Some(latest.timestamp);

        let explicit_velocity = latest.velocity.filter(|velocity| velocity.is_finite());
        let raw_velocity = explicit_velocity.or_else(|| {
            let previous_position = previous_position?;
            let previous_timestamp = previous_timestamp?;
            let dt = latest
                .timestamp
                .checked_duration_since(previous_timestamp)?
                .as_secs_f64();
            if dt <= 0.0 || !dt.is_finite() {
                return None;
            }
            Some((latest.position - previous_position) / dt)
        })?;
        let dt = previous_timestamp
            .and_then(|previous| latest.timestamp.checked_duration_since(previous))
            .map(|duration| duration.as_secs_f64())
            .filter(|dt| *dt > 0.0 && dt.is_finite())
            .unwrap_or(0.0);
        self.filtered_velocity = Some(self.velocity_filter.filter(raw_velocity, dt, self.config))
            .filter(|velocity| velocity.is_finite());
        self.filtered_velocity
    }

    pub(super) fn reset(&mut self) {
        self.velocity_filter.reset();
        self.latest_position = None;
        self.latest_timestamp = None;
        self.filtered_velocity = None;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OneEuroVelocityPointerPredictor {
    config: OneEuroVelocityPointerPredictorConfig,
    velocity: OneEuroVelocityEstimator,
}

impl OneEuroVelocityPointerPredictor {
    pub fn new(config: OneEuroVelocityPointerPredictorConfig) -> Self {
        Self {
            config,
            velocity: OneEuroVelocityEstimator::new(config.one_euro),
        }
    }
}

impl Default for OneEuroVelocityPointerPredictor {
    fn default() -> Self {
        Self::new(OneEuroVelocityPointerPredictorConfig::default())
    }
}

impl PointerPredictor for OneEuroVelocityPointerPredictor {
    fn predict(&mut self, input: PointerPredictionInput<'_>) -> Option<Point2d> {
        let latest = input.history.last()?;
        let horizon = input.horizon.as_secs_f64();
        if !horizon.is_finite() || horizon < 0.0 {
            return Some(latest.position);
        }

        let velocity = self.velocity.estimate(input)?;
        let decay_time = self.config.decay.decay_time.as_secs_f64();
        if decay_time <= 0.0 || !decay_time.is_finite() {
            return Some(latest.position);
        }

        let effective_horizon = decay_time * (1.0 - (-horizon / decay_time).exp());
        Some(latest.position + velocity * effective_horizon)
    }

    fn reset(&mut self) {
        self.velocity.reset();
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use tron_api::{PointerPredictionInput, PointerPredictionSample, PointerPredictor};

    use super::*;

    #[test]
    fn predicts_from_filtered_velocity() {
        let timestamp = Instant::now();
        let history = [
            sample(timestamp, 0, Point2d::new(0.5, 0.5)),
            sample(timestamp, 16, Point2d::new(0.516, 0.5)),
        ];
        let mut predictor = OneEuroVelocityPointerPredictor::default();
        predictor.predict(PointerPredictionInput {
            history: &history[..1],
            horizon: Duration::ZERO,
        });
        let output = predictor
            .predict(PointerPredictionInput {
                history: &history,
                horizon: Duration::from_millis(20),
            })
            .unwrap();

        assert!(output.x > 0.516);
        assert!(output.x < 0.536);
        assert!((output.y - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn repeated_prediction_does_not_reingest_same_sample() {
        let timestamp = Instant::now();
        let history = [
            sample(timestamp, 0, Point2d::new(0.5, 0.5)),
            sample(timestamp, 16, Point2d::new(0.516, 0.5)),
        ];
        let mut predictor = OneEuroVelocityPointerPredictor::default();
        predictor.predict(PointerPredictionInput {
            history: &history[..1],
            horizon: Duration::ZERO,
        });

        let input = PointerPredictionInput {
            history: &history,
            horizon: Duration::from_millis(20),
        };
        let first = predictor.predict(input).unwrap();
        let second = predictor.predict(input).unwrap();

        assert!((first - second).length() < 1.0e-12);
    }

    fn sample(base: Instant, millis: u64, position: Point2d) -> PointerPredictionSample {
        PointerPredictionSample {
            timestamp: base + Duration::from_millis(millis),
            position,
            velocity: None,
        }
    }
}
