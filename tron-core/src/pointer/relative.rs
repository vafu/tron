use std::time::Instant;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{Duration, MissedTickBehavior};
use tron_api::{
    EventProducer, HandGesture, Point2d, PointerCancelReason, PointerEvent, PointerInput,
    PointerJoystickVisualization, PointerOutput, PointerPredictionInput, PointerPredictionSample,
    PointerPredictor, PointerVisualization,
};

use crate::pointer::OneEuroVelocityPointerPredictor;

#[derive(Clone, Copy, Debug)]
pub struct RelativePointerProducer {
    pub clutch_down_strength: f32,
    pub clutch_up_strength: f32,
    pub pinch_down_strength: f32,
    pub pinch_up_strength: f32,
    pub deadzone: f64,
    pub sensitivity: f64,
    pub smoothing_min_cutoff_hz: f64,
    pub smoothing_beta: f64,
    pub max_prediction_horizon: Duration,
    pub tick_interval: Duration,
}

impl Default for RelativePointerProducer {
    fn default() -> Self {
        Self {
            clutch_down_strength: 0.2,
            clutch_up_strength: 0.25,
            pinch_down_strength: 0.55,
            pinch_up_strength: 0.35,
            deadzone: 0.015,
            sensitivity: 1.0,
            smoothing_min_cutoff_hz: 5.0,
            smoothing_beta: 0.08,
            max_prediction_horizon: Duration::from_millis(80),
            tick_interval: Duration::from_millis(16),
        }
    }
}

impl EventProducer<PointerInput, PointerOutput> for RelativePointerProducer {
    fn spawn(
        self,
        mut input: mpsc::Receiver<PointerInput>,
        output: mpsc::Sender<PointerOutput>,
    ) -> JoinHandle<Result<()>> {
        tokio::spawn(async move {
            let mut state = RelativePointerState::default();
            let mut interval = tokio::time::interval(self.tick_interval);
            interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    input = input.recv() => {
                        let Some(input) = input else {
                            return Ok(());
                        };
                        for output_item in state.update_input(input, self) {
                            if output.send(output_item).await.is_err() {
                                return Ok(());
                            }
                        }
                    }
                    _ = interval.tick() => {
                        for output_item in state.tick(self) {
                            if output.send(output_item).await.is_err() {
                                return Ok(());
                            }
                        }
                    }
                }
            }
        })
    }
}

#[derive(Default)]
struct RelativePointerState {
    anchor_position: Option<Point2d>,
    current_position: Option<Point2d>,
    filtered_position: Option<Point2d>,
    last_timestamp: Option<Instant>,
    prediction_history: Vec<PointerPredictionSample>,
    predictor: OneEuroVelocityPointerPredictor,
    emitted_offset: Point2d,
    clutch_engaged: bool,
    button_down: bool,
}

#[derive(Clone, Copy, Debug)]
struct ClutchSample {
    strength: f32,
    position: Point2d,
    velocity: Point2d,
}

impl RelativePointerState {
    fn update_input(
        &mut self,
        input: PointerInput,
        config: RelativePointerProducer,
    ) -> Vec<PointerOutput> {
        let timestamp = input.gesture.timestamp;
        if input.gesture.palm.is_none() {
            return self.cancel(timestamp, config);
        }
        if matches!(input.gesture.gesture, HandGesture::NoHand) {
            return self.cancel(timestamp, config);
        }

        let clutch = input
            .gesture
            .signal(HandGesture::Clutch)
            .map(|signal| ClutchSample {
                strength: signal.strength,
                position: signal.position.clamp(Point2d::ZERO, Point2d::ONE),
                velocity: signal.velocity.unwrap_or(Point2d::ZERO),
            })
            .map(|sample| self.smooth_clutch_sample(sample, timestamp, config));
        if let Some(sample) = clutch {
            self.current_position = Some(sample.position);
        }
        let pinch_strength = input
            .gesture
            .signal(HandGesture::Pinch)
            .map(|signal| signal.strength)
            .unwrap_or(0.0);

        let mut events = Vec::new();
        match (self.clutch_engaged, clutch) {
            (false, Some(sample)) if sample.strength >= config.clutch_down_strength => {
                self.clutch_engaged = true;
                self.anchor_position = Some(sample.position);
                self.current_position = Some(sample.position);
                self.push_prediction_sample(timestamp, sample);
                self.emitted_offset = Point2d::ZERO;
            }
            (true, Some(sample)) if sample.strength <= config.clutch_up_strength => {
                self.reset_clutch();
            }
            (true, Some(sample)) => {
                self.push_prediction_sample(timestamp, sample);
                if let Some(event) = self.move_to_position(timestamp, sample.position, config) {
                    events.push(event);
                }
            }
            (true, None) => {
                self.reset_clutch();
            }
            _ => {}
        }

        let pinch_strength = if self.clutch_engaged {
            pinch_strength
        } else {
            0.0
        };
        if !self.button_down && pinch_strength >= config.pinch_down_strength {
            self.button_down = true;
            events.push(PointerOutput::Event(PointerEvent::Down { timestamp }));
        } else if self.button_down && pinch_strength <= config.pinch_up_strength {
            self.button_down = false;
            events.push(PointerOutput::Event(PointerEvent::Up { timestamp }));
        }

        events.push(self.visualization(timestamp, config));
        events
    }

    fn tick(&mut self, config: RelativePointerProducer) -> Vec<PointerOutput> {
        if !self.clutch_engaged {
            return Vec::new();
        }
        let timestamp = Instant::now();
        self.predict_position(timestamp, config)
            .and_then(|position| self.move_to_position(timestamp, position, config))
            .into_iter()
            .collect()
    }

    fn smooth_clutch_sample(
        &mut self,
        sample: ClutchSample,
        timestamp: Instant,
        config: RelativePointerProducer,
    ) -> ClutchSample {
        let measured_position = sample.position.clamp(Point2d::ZERO, Point2d::ONE);
        let Some(previous) = self.filtered_position else {
            self.filtered_position = Some(measured_position);
            self.last_timestamp = Some(timestamp);
            return ClutchSample {
                position: measured_position,
                ..sample
            };
        };

        let dt = self
            .last_timestamp
            .and_then(|previous_timestamp| timestamp.checked_duration_since(previous_timestamp))
            .map(|duration| duration.as_secs_f64())
            .filter(|dt| *dt > 0.0 && dt.is_finite())
            .unwrap_or(1.0 / 60.0);
        let speed = sample.velocity.length();
        let cutoff = (config.smoothing_min_cutoff_hz + config.smoothing_beta * speed).max(0.001);
        let alpha = low_pass_alpha(dt, cutoff);
        let position = previous.lerp(measured_position, alpha);

        self.filtered_position = Some(position);
        self.last_timestamp = Some(timestamp);
        ClutchSample { position, ..sample }
    }

    fn push_prediction_sample(&mut self, timestamp: Instant, sample: ClutchSample) {
        self.prediction_history.push(PointerPredictionSample {
            timestamp,
            position: sample.position,
            velocity: Some(sample.velocity),
        });
        let excess = self.prediction_history.len().saturating_sub(8);
        if excess > 0 {
            self.prediction_history.drain(0..excess);
        }
    }

    fn predict_position(
        &mut self,
        timestamp: Instant,
        config: RelativePointerProducer,
    ) -> Option<Point2d> {
        let latest = self.prediction_history.last()?;
        let horizon = timestamp
            .checked_duration_since(latest.timestamp)
            .unwrap_or_default()
            .min(config.max_prediction_horizon);
        self.predictor
            .predict(PointerPredictionInput {
                history: &self.prediction_history,
                horizon,
            })
            .map(|position| position.clamp(Point2d::ZERO, Point2d::ONE))
    }

    fn move_to_position(
        &mut self,
        timestamp: Instant,
        position: Point2d,
        config: RelativePointerProducer,
    ) -> Option<PointerOutput> {
        let anchor = self.anchor_position?;
        let offset = active_offset(position - anchor, config);
        let delta = offset - self.emitted_offset;
        self.emitted_offset = offset;
        (delta != Point2d::ZERO).then_some(PointerOutput::Event(PointerEvent::Move {
            timestamp,
            position: None,
            delta,
        }))
    }

    fn cancel(
        &mut self,
        timestamp: Instant,
        config: RelativePointerProducer,
    ) -> Vec<PointerOutput> {
        let was_button_down = self.button_down;
        self.reset_clutch();
        self.button_down = false;
        let mut events = Vec::new();
        if was_button_down {
            events.push(PointerOutput::Event(PointerEvent::Cancel {
                timestamp,
                reason: PointerCancelReason::LostTracking,
            }));
        }
        events.push(self.visualization(timestamp, config));
        events
    }

    fn reset_clutch(&mut self) {
        self.anchor_position = None;
        self.current_position = None;
        self.filtered_position = None;
        self.last_timestamp = None;
        self.prediction_history.clear();
        self.predictor.reset();
        self.emitted_offset = Point2d::ZERO;
        self.clutch_engaged = false;
    }

    fn visualization(&self, timestamp: Instant, config: RelativePointerProducer) -> PointerOutput {
        PointerOutput::Visualization(PointerVisualization::Joystick(
            PointerJoystickVisualization {
                timestamp,
                anchor: self.anchor_position,
                current: self.current_position,
                deadzone_radius: config.deadzone,
                engaged: self.clutch_engaged,
            },
        ))
    }
}

fn active_offset(displacement: Point2d, config: RelativePointerProducer) -> Point2d {
    let length = displacement.length();
    if length <= config.deadzone {
        Point2d::ZERO
    } else {
        displacement * (((length - config.deadzone) / length) * config.sensitivity)
    }
}

fn low_pass_alpha(dt: f64, cutoff_hz: f64) -> f64 {
    let tau = 1.0 / (std::f64::consts::TAU * cutoff_hz);
    1.0 / (1.0 + tau / dt)
}

#[cfg(test)]
mod tests {
    use tron_api::{GestureFrame, GestureSignal, PalmPose2d};

    use super::*;

    fn input(position: Point2d, gesture: HandGesture, signals: Vec<GestureSignal>) -> PointerInput {
        input_at(Instant::now(), position, gesture, signals)
    }

    fn input_at(
        timestamp: Instant,
        position: Point2d,
        gesture: HandGesture,
        signals: Vec<GestureSignal>,
    ) -> PointerInput {
        PointerInput {
            gesture: GestureFrame {
                timestamp,
                palm: Some(PalmPose2d {
                    center: position,
                    rotation_radians: 0.0,
                    extent: Point2d::splat(0.1),
                }),
                signals,
                gesture,
            },
        }
    }

    fn signal(gesture: HandGesture, strength: f32, position: Point2d) -> GestureSignal {
        GestureSignal {
            gesture,
            strength,
            position,
            velocity: None,
        }
    }

    fn signal_with_velocity(
        gesture: HandGesture,
        strength: f32,
        position: Point2d,
        velocity: Point2d,
    ) -> GestureSignal {
        GestureSignal {
            gesture,
            strength,
            position,
            velocity: Some(velocity),
        }
    }

    #[test]
    fn relative_pointer_applies_camera_sample_immediately() {
        let mut state = RelativePointerState::default();
        let config = RelativePointerProducer {
            deadzone: 0.0,
            smoothing_min_cutoff_hz: 1.0e9,
            tick_interval: Duration::from_millis(16),
            ..RelativePointerProducer::default()
        };
        let timestamp = Instant::now();

        state.update_input(
            input_at(
                timestamp,
                Point2d::new(0.5, 0.5),
                HandGesture::Clutch,
                vec![signal(HandGesture::Clutch, 0.8, Point2d::new(0.50, 0.50))],
            ),
            config,
        );
        let events = state.update_input(
            input_at(
                timestamp + Duration::from_millis(128),
                Point2d::new(0.5, 0.5),
                HandGesture::Clutch,
                vec![signal(HandGesture::Clutch, 0.8, Point2d::new(0.66, 0.50))],
            ),
            config,
        );

        let PointerOutput::Event(PointerEvent::Move { delta, .. }) = events[0] else {
            panic!("expected immediate relative move");
        };
        assert_point_near(delta, Point2d::new(0.16, 0.0));
    }

    #[test]
    fn relative_pointer_disengages_without_button_up() {
        let mut state = RelativePointerState::default();
        let config = RelativePointerProducer::default();
        state.update_input(
            input(
                Point2d::new(0.5, 0.5),
                HandGesture::Clutch,
                vec![signal(HandGesture::Clutch, 0.8, Point2d::new(0.5, 0.5))],
            ),
            config,
        );

        let events = state.update_input(
            input(Point2d::new(0.5, 0.5), HandGesture::OpenPalm, vec![]),
            config,
        );
        assert!(matches!(
            events[0],
            PointerOutput::Visualization(PointerVisualization::Joystick(_))
        ));
    }

    #[test]
    fn relative_pointer_uses_pinch_for_button_while_clutch_moves() {
        let mut state = RelativePointerState::default();
        let config = RelativePointerProducer {
            deadzone: 0.0,
            smoothing_min_cutoff_hz: 1.0e9,
            ..RelativePointerProducer::default()
        };
        state.update_input(
            input(
                Point2d::new(0.5, 0.5),
                HandGesture::Clutch,
                vec![signal(HandGesture::Clutch, 0.8, Point2d::new(0.5, 0.5))],
            ),
            config,
        );

        let events = state.update_input(
            input(
                Point2d::new(0.5, 0.5),
                HandGesture::Pinch,
                vec![
                    signal(HandGesture::Clutch, 0.8, Point2d::new(0.5, 0.5)),
                    signal(HandGesture::Pinch, 0.8, Point2d::new(0.52, 0.48)),
                ],
            ),
            config,
        );
        assert!(matches!(
            events[0],
            PointerOutput::Event(PointerEvent::Down { .. })
        ));
        assert!(matches!(
            events[1],
            PointerOutput::Visualization(PointerVisualization::Joystick(_))
        ));

        let events = state.update_input(
            input(
                Point2d::new(0.5, 0.5),
                HandGesture::Clutch,
                vec![signal(HandGesture::Clutch, 0.8, Point2d::new(0.6, 0.5))],
            ),
            config,
        );
        assert!(matches!(
            events.first(),
            Some(PointerOutput::Event(PointerEvent::Move { .. }))
        ));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, PointerOutput::Event(PointerEvent::Up { .. })))
        );
    }

    #[test]
    fn relative_pointer_predicts_between_camera_samples_from_velocity() {
        let mut state = RelativePointerState::default();
        let config = RelativePointerProducer::default();
        let timestamp = Instant::now();

        state.update_input(
            input_at(
                timestamp,
                Point2d::new(0.5, 0.5),
                HandGesture::Clutch,
                vec![signal_with_velocity(
                    HandGesture::Clutch,
                    0.8,
                    Point2d::new(0.5, 0.5),
                    Point2d::new(1.0, 0.0),
                )],
            ),
            config,
        );
        let predicted = state
            .predict_position(timestamp + Duration::from_millis(20), config)
            .unwrap();

        assert!(predicted.x > 0.5);
        assert!(predicted.x < 0.52);
        assert!((predicted.y - 0.5).abs() < 1.0e-12);
    }

    fn assert_point_near(actual: Point2d, expected: Point2d) {
        assert!(
            (actual - expected).length() < 1.0e-4,
            "actual={actual:?} expected={expected:?}"
        );
    }
}
