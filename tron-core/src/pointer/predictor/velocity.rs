use std::time::Duration;

use tron_api::{Point2d, PointerPredictionInput, PointerPredictor};

#[derive(Clone, Copy, Debug, Default)]
pub struct VelocityPointerPredictor;

impl PointerPredictor for VelocityPointerPredictor {
    fn predict(&mut self, input: PointerPredictionInput<'_>) -> Option<Point2d> {
        let latest = input.history.last()?;
        let dt = input.horizon.as_secs_f64();
        if !dt.is_finite() || dt < 0.0 {
            return Some(latest.position);
        }

        if let Some(velocity) = latest.velocity.filter(|velocity| velocity.is_finite()) {
            return Some(latest.position + velocity * dt);
        }

        let previous = input
            .history
            .iter()
            .rev()
            .skip(1)
            .find(|sample| sample.position.is_finite())?;
        let sample_dt = latest
            .timestamp
            .checked_duration_since(previous.timestamp)?
            .as_secs_f64();
        if sample_dt <= 0.0 || !sample_dt.is_finite() {
            return Some(latest.position);
        }

        let velocity = (latest.position - previous.position) / sample_dt;
        Some(latest.position + velocity * dt)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DecayingVelocityPointerPredictorConfig {
    pub decay_time: Duration,
}

impl Default for DecayingVelocityPointerPredictorConfig {
    fn default() -> Self {
        Self {
            decay_time: Duration::from_millis(35),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DecayingVelocityPointerPredictor {
    config: DecayingVelocityPointerPredictorConfig,
}

impl DecayingVelocityPointerPredictor {
    pub fn new(config: DecayingVelocityPointerPredictorConfig) -> Self {
        Self { config }
    }
}

impl Default for DecayingVelocityPointerPredictor {
    fn default() -> Self {
        Self::new(DecayingVelocityPointerPredictorConfig::default())
    }
}

impl PointerPredictor for DecayingVelocityPointerPredictor {
    fn predict(&mut self, input: PointerPredictionInput<'_>) -> Option<Point2d> {
        let latest = input.history.last()?;
        let horizon = input.horizon.as_secs_f64();
        if !horizon.is_finite() || horizon < 0.0 {
            return Some(latest.position);
        }

        let velocity = latest
            .velocity
            .filter(|velocity| velocity.is_finite())
            .or_else(|| derived_velocity(input))?;
        let decay_time = self.config.decay_time.as_secs_f64();
        if decay_time <= 0.0 || !decay_time.is_finite() {
            return Some(latest.position);
        }

        let effective_horizon = decay_time * (1.0 - (-horizon / decay_time).exp());
        Some(latest.position + velocity * effective_horizon)
    }
}

fn derived_velocity(input: PointerPredictionInput<'_>) -> Option<Point2d> {
    let latest = input.history.last()?;
    let previous = input
        .history
        .iter()
        .rev()
        .skip(1)
        .find(|sample| sample.position.is_finite())?;
    let sample_dt = latest
        .timestamp
        .checked_duration_since(previous.timestamp)?
        .as_secs_f64();
    if sample_dt <= 0.0 || !sample_dt.is_finite() {
        return None;
    }

    Some((latest.position - previous.position) / sample_dt)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use tron_api::{PointerPredictionInput, PointerPredictionSample, PointerPredictor};

    use super::*;

    #[test]
    fn predicts_from_explicit_velocity() {
        let timestamp = Instant::now();
        let history = [PointerPredictionSample {
            timestamp,
            position: Point2d::new(0.5, 0.5),
            velocity: Some(Point2d::new(1.0, -0.5)),
        }];
        let output = VelocityPointerPredictor
            .predict(PointerPredictionInput {
                history: &history,
                horizon: Duration::from_millis(20),
            })
            .unwrap();

        assert!((output - Point2d::new(0.52, 0.49)).length() < 1.0e-12);
    }

    #[test]
    fn derives_velocity_from_history() {
        let timestamp = Instant::now();
        let history = [
            PointerPredictionSample {
                timestamp,
                position: Point2d::new(0.4, 0.5),
                velocity: None,
            },
            PointerPredictionSample {
                timestamp: timestamp + Duration::from_millis(100),
                position: Point2d::new(0.5, 0.5),
                velocity: None,
            },
        ];
        let output = VelocityPointerPredictor
            .predict(PointerPredictionInput {
                history: &history,
                horizon: Duration::from_millis(20),
            })
            .unwrap();

        assert!((output - Point2d::new(0.52, 0.5)).length() < 1.0e-12);
    }

    #[test]
    fn decaying_predictor_trusts_velocity_less_over_time() {
        let timestamp = Instant::now();
        let history = [PointerPredictionSample {
            timestamp,
            position: Point2d::new(0.5, 0.5),
            velocity: Some(Point2d::new(1.0, 0.0)),
        }];
        let mut predictor =
            DecayingVelocityPointerPredictor::new(DecayingVelocityPointerPredictorConfig {
                decay_time: Duration::from_millis(35),
            });
        let output = predictor
            .predict(PointerPredictionInput {
                history: &history,
                horizon: Duration::from_millis(80),
            })
            .unwrap();

        assert!(output.x > 0.5);
        assert!(output.x < 0.58);
        assert!((output.x - 0.5) < 0.035);
    }

    #[test]
    fn decaying_predictor_matches_linear_for_tiny_horizon() {
        let timestamp = Instant::now();
        let history = [PointerPredictionSample {
            timestamp,
            position: Point2d::new(0.5, 0.5),
            velocity: Some(Point2d::new(1.0, 0.0)),
        }];
        let mut predictor =
            DecayingVelocityPointerPredictor::new(DecayingVelocityPointerPredictorConfig {
                decay_time: Duration::from_millis(35),
            });
        let output = predictor
            .predict(PointerPredictionInput {
                history: &history,
                horizon: Duration::from_millis(1),
            })
            .unwrap();

        assert!((output.x - 0.501).abs() < 0.00002);
    }
}
