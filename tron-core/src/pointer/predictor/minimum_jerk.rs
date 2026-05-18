use std::time::Duration;

use tron_api::{Point2d, PointerPredictionInput, PointerPredictor};

use crate::filter::OneEuroConfig;

use super::one_euro::OneEuroVelocityEstimator;

#[derive(Clone, Copy, Debug)]
pub struct MinimumJerkPointerPredictorConfig {
    pub one_euro: OneEuroConfig,
    pub brake_time: Duration,
}

impl Default for MinimumJerkPointerPredictorConfig {
    fn default() -> Self {
        Self {
            one_euro: OneEuroConfig {
                min_cutoff: 5.0,
                beta: 0.08,
                derivative_cutoff: 1.0,
            },
            brake_time: Duration::from_millis(70),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MinimumJerkPointerPredictor {
    config: MinimumJerkPointerPredictorConfig,
    velocity: OneEuroVelocityEstimator,
}

impl MinimumJerkPointerPredictor {
    pub fn new(config: MinimumJerkPointerPredictorConfig) -> Self {
        Self {
            config,
            velocity: OneEuroVelocityEstimator::new(config.one_euro),
        }
    }
}

impl Default for MinimumJerkPointerPredictor {
    fn default() -> Self {
        Self::new(MinimumJerkPointerPredictorConfig::default())
    }
}

impl PointerPredictor for MinimumJerkPointerPredictor {
    fn predict(&mut self, input: PointerPredictionInput<'_>) -> Option<Point2d> {
        let latest = input.history.last()?;
        let horizon = input.horizon.as_secs_f64();
        if !horizon.is_finite() || horizon < 0.0 {
            return Some(latest.position);
        }

        let velocity = self.velocity.estimate(input)?;
        let effective_horizon =
            minimum_jerk_braking_horizon(horizon, self.config.brake_time.as_secs_f64())?;
        Some(latest.position + velocity * effective_horizon)
    }

    fn reset(&mut self) {
        self.velocity.reset();
    }
}

fn minimum_jerk_braking_horizon(horizon: f64, brake_time: f64) -> Option<f64> {
    if brake_time <= 0.0 || !brake_time.is_finite() {
        return Some(0.0);
    }
    let s = (horizon / brake_time).clamp(0.0, 1.0);
    let ease_integral = 2.5 * s.powi(4) - 3.0 * s.powi(5) + s.powi(6);
    Some(brake_time * (s - ease_integral))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use tron_api::{PointerPredictionInput, PointerPredictionSample, PointerPredictor};

    use super::*;

    #[test]
    fn braking_horizon_starts_like_linear_prediction() {
        let horizon = minimum_jerk_braking_horizon(0.001, 0.070).unwrap();

        assert!((horizon - 0.001).abs() < 1.0e-8);
    }

    #[test]
    fn braking_horizon_caps_at_half_brake_time() {
        let horizon = minimum_jerk_braking_horizon(1.0, 0.070).unwrap();

        assert!((horizon - 0.035).abs() < 1.0e-12);
    }

    #[test]
    fn predicts_with_minimum_jerk_braking() {
        let timestamp = Instant::now();
        let history = [
            sample(timestamp, 0, Point2d::new(0.5, 0.5)),
            sample(timestamp, 16, Point2d::new(0.516, 0.5)),
        ];
        let mut predictor = MinimumJerkPointerPredictor::default();
        predictor.predict(PointerPredictionInput {
            history: &history[..1],
            horizon: Duration::ZERO,
        });
        let output = predictor
            .predict(PointerPredictionInput {
                history: &history,
                horizon: Duration::from_millis(70),
            })
            .unwrap();

        assert!(output.x > 0.516);
        assert!(output.x < 0.552);
        assert!((output.y - 0.5).abs() < 1.0e-12);
    }

    fn sample(base: Instant, millis: u64, position: Point2d) -> PointerPredictionSample {
        PointerPredictionSample {
            timestamp: base + Duration::from_millis(millis),
            position,
            velocity: None,
        }
    }
}
