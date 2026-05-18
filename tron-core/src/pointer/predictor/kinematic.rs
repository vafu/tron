use tron_api::{Point2d, PointerPredictionInput, PointerPredictionSample, PointerPredictor};

#[derive(Clone, Copy, Debug)]
pub struct KinematicPointerPredictorConfig {
    pub velocity_smoothing: f64,
    pub acceleration_smoothing: f64,
    pub error_gain: f64,
    pub max_speed: f64,
    pub max_acceleration: f64,
}

impl Default for KinematicPointerPredictorConfig {
    fn default() -> Self {
        Self {
            velocity_smoothing: 0.55,
            acceleration_smoothing: 0.35,
            error_gain: 0.35,
            max_speed: 8.0,
            max_acceleration: 80.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KinematicPointerPredictor {
    config: KinematicPointerPredictorConfig,
    last_predicted_sample: Option<PointerPredictionSample>,
}

impl KinematicPointerPredictor {
    pub fn new(config: KinematicPointerPredictorConfig) -> Self {
        Self {
            config,
            last_predicted_sample: None,
        }
    }
}

impl Default for KinematicPointerPredictor {
    fn default() -> Self {
        Self::new(KinematicPointerPredictorConfig::default())
    }
}

impl PointerPredictor for KinematicPointerPredictor {
    fn predict(&mut self, input: PointerPredictionInput<'_>) -> Option<Point2d> {
        let latest = *input.history.last()?;
        let horizon = input.horizon.as_secs_f64();
        if horizon < 0.0 || !horizon.is_finite() {
            return Some(latest.position);
        }

        let velocity = smoothed_velocity(input.history, self.config.velocity_smoothing)
            .or_else(|| latest.velocity.filter(|velocity| velocity.is_finite()))
            .unwrap_or(Point2d::ZERO);
        let acceleration = smoothed_acceleration(input.history, self.config.acceleration_smoothing)
            .unwrap_or(Point2d::ZERO);
        let corrected_velocity = self
            .correct_velocity(latest, velocity)
            .clamp_length_max(self.config.max_speed);
        let acceleration = acceleration.clamp_length_max(self.config.max_acceleration);
        let predicted = latest.position
            + corrected_velocity * horizon
            + acceleration * (0.5 * horizon * horizon);

        self.last_predicted_sample = Some(PointerPredictionSample {
            timestamp: latest.timestamp + input.horizon,
            position: predicted,
            velocity: Some(corrected_velocity + acceleration * horizon),
        });
        Some(predicted)
    }

    fn reset(&mut self) {
        self.last_predicted_sample = None;
    }
}

impl KinematicPointerPredictor {
    fn correct_velocity(&self, latest: PointerPredictionSample, velocity: Point2d) -> Point2d {
        let Some(predicted) = self.last_predicted_sample else {
            return velocity;
        };
        let Some(dt) = latest
            .timestamp
            .checked_duration_since(predicted.timestamp)
            .map(|duration| duration.as_secs_f64())
            .filter(|dt| *dt > 0.0 && dt.is_finite())
        else {
            return velocity;
        };

        let error = latest.position - predicted.position;
        velocity + error * (self.config.error_gain / dt)
    }
}

fn smoothed_velocity(history: &[PointerPredictionSample], smoothing: f64) -> Option<Point2d> {
    let mut velocity: Option<Point2d> = None;
    for pair in history.windows(2) {
        let segment = segment_velocity(pair[0], pair[1])?;
        velocity = Some(match velocity {
            Some(previous) => previous.lerp(segment, smoothing.clamp(0.0, 1.0)),
            None => segment,
        });
    }
    velocity
}

fn smoothed_acceleration(history: &[PointerPredictionSample], smoothing: f64) -> Option<Point2d> {
    let mut acceleration: Option<Point2d> = None;
    let mut previous_velocity: Option<Point2d> = None;
    for pair in history.windows(2) {
        let velocity = segment_velocity(pair[0], pair[1])?;
        if let Some(previous) = previous_velocity {
            let dt = pair[1]
                .timestamp
                .checked_duration_since(pair[0].timestamp)?
                .as_secs_f64();
            if dt > 0.0 && dt.is_finite() {
                let segment = (velocity - previous) / dt;
                acceleration = Some(match acceleration {
                    Some(previous) => previous.lerp(segment, smoothing.clamp(0.0, 1.0)),
                    None => segment,
                });
            }
        }
        previous_velocity = Some(velocity);
    }
    acceleration
}

fn segment_velocity(
    previous: PointerPredictionSample,
    current: PointerPredictionSample,
) -> Option<Point2d> {
    let dt = current
        .timestamp
        .checked_duration_since(previous.timestamp)?
        .as_secs_f64();
    if dt <= 0.0 || !dt.is_finite() {
        return None;
    }
    Some((current.position - previous.position) / dt)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use tron_api::{PointerPredictionInput, PointerPredictionSample, PointerPredictor};

    use super::*;

    #[test]
    fn predicts_with_acceleration_for_curve_like_motion() {
        let timestamp = Instant::now();
        let history = [
            sample(timestamp, 0.0, Point2d::new(0.00, 0.00)),
            sample(timestamp, 100.0, Point2d::new(0.10, 0.00)),
            sample(timestamp, 200.0, Point2d::new(0.18, 0.04)),
        ];
        let output = KinematicPointerPredictor::default()
            .predict(PointerPredictionInput {
                history: &history,
                horizon: Duration::from_millis(50),
            })
            .unwrap();

        assert!(output.x > 0.18);
        assert!(output.y > 0.04);
    }

    #[test]
    fn error_feedback_reduces_repeated_overshoot() {
        let timestamp = Instant::now();
        let mut predictor = KinematicPointerPredictor::new(KinematicPointerPredictorConfig {
            error_gain: 0.5,
            ..KinematicPointerPredictorConfig::default()
        });
        let history = [
            sample(timestamp, 0.0, Point2d::new(0.0, 0.0)),
            sample(timestamp, 100.0, Point2d::new(0.1, 0.0)),
        ];
        predictor.predict(PointerPredictionInput {
            history: &history,
            horizon: Duration::from_millis(100),
        });

        let corrected_history = [
            sample(timestamp, 0.0, Point2d::new(0.0, 0.0)),
            sample(timestamp, 100.0, Point2d::new(0.1, 0.0)),
            sample(timestamp, 200.0, Point2d::new(0.15, 0.0)),
        ];
        let output = predictor
            .predict(PointerPredictionInput {
                history: &corrected_history,
                horizon: Duration::from_millis(50),
            })
            .unwrap();

        assert!(output.x < 0.20);
    }

    fn sample(base: Instant, millis: f64, position: Point2d) -> PointerPredictionSample {
        PointerPredictionSample {
            timestamp: base + Duration::from_secs_f64(millis / 1000.0),
            position,
            velocity: None,
        }
    }
}
