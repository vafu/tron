use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{Duration, MissedTickBehavior};
use tron_api::{
    EventProducer, HandGesture, Point2d, PointerEvent, PointerInput, PointerJoystickVisualization,
    PointerOutput, PointerVisualization,
};

#[derive(Clone, Copy, Debug)]
pub struct JoystickPointerProducer {
    pub pinch_down_strength: f32,
    pub pinch_up_strength: f32,
    pub deadzone: f64,
    pub speed_per_second: f64,
    pub tick_interval: Duration,
}

impl Default for JoystickPointerProducer {
    fn default() -> Self {
        Self {
            pinch_down_strength: 0.55,
            pinch_up_strength: 0.35,
            deadzone: 0.015,
            speed_per_second: 2.8,
            tick_interval: Duration::from_millis(16),
        }
    }
}

impl EventProducer<PointerInput, PointerOutput> for JoystickPointerProducer {
    fn spawn(
        self,
        mut input: mpsc::Receiver<PointerInput>,
        output: mpsc::Sender<PointerOutput>,
    ) -> JoinHandle<Result<()>> {
        tokio::spawn(async move {
            let mut state = JoystickPointerState::default();
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
struct JoystickPointerState {
    anchor_position: Option<Point2d>,
    current_position: Option<Point2d>,
    primary_down: bool,
    tracking: bool,
}

impl JoystickPointerState {
    fn update_input(
        &mut self,
        input: PointerInput,
        config: JoystickPointerProducer,
    ) -> Vec<PointerOutput> {
        let timestamp = input.gesture.timestamp;
        if input.gesture.palm.is_none() {
            return self.cancel(timestamp, config);
        }
        if matches!(input.gesture.gesture, HandGesture::NoHand) {
            return self.cancel(timestamp, config);
        }

        self.tracking = true;

        let pinch = input.gesture.signal(HandGesture::Pinch).map(|signal| {
            (
                signal.strength,
                signal.position.clamp(Point2d::ZERO, Point2d::ONE),
            )
        });
        if let Some((_, position)) = pinch {
            self.current_position = Some(position);
        }

        let mut events = Vec::new();
        match (self.primary_down, pinch) {
            (false, Some((strength, position))) if strength >= config.pinch_down_strength => {
                self.primary_down = true;
                self.anchor_position = Some(position);
                events.push(PointerOutput::Event(PointerEvent::Down { timestamp }));
            }
            (true, Some((strength, _))) if strength <= config.pinch_up_strength => {
                self.primary_down = false;
                self.anchor_position = None;
                self.current_position = None;
                events.push(PointerOutput::Event(PointerEvent::Up { timestamp }));
            }
            (true, None) => {
                self.primary_down = false;
                self.anchor_position = None;
                self.current_position = None;
                events.push(PointerOutput::Event(PointerEvent::Up { timestamp }));
            }
            _ => {}
        }

        events.push(self.visualization(timestamp, config));
        events
    }

    fn tick(&mut self, config: JoystickPointerProducer) -> Vec<PointerOutput> {
        if !self.primary_down {
            return Vec::new();
        }

        let (Some(anchor), Some(position)) = (self.anchor_position, self.current_position) else {
            return Vec::new();
        };
        let displacement = position - anchor;
        let length = displacement.length();
        if length <= config.deadzone {
            return Vec::new();
        }

        let active = displacement * ((length - config.deadzone) / length);
        let seconds = config.tick_interval.as_secs_f64();
        vec![PointerOutput::Event(PointerEvent::Move {
            timestamp: std::time::Instant::now(),
            position: None,
            delta: active * config.speed_per_second * seconds,
        })]
    }

    fn cancel(
        &mut self,
        timestamp: std::time::Instant,
        config: JoystickPointerProducer,
    ) -> Vec<PointerOutput> {
        let was_down = self.primary_down;
        self.anchor_position = None;
        self.current_position = None;
        self.primary_down = false;
        self.tracking = false;
        let mut events = vec![self.visualization(timestamp, config)];
        if was_down {
            events.push(PointerOutput::Event(PointerEvent::Up { timestamp }));
        }
        events
    }

    fn visualization(
        &self,
        timestamp: std::time::Instant,
        config: JoystickPointerProducer,
    ) -> PointerOutput {
        PointerOutput::Visualization(PointerVisualization::Joystick(
            PointerJoystickVisualization {
                timestamp,
                anchor: self.anchor_position,
                current: self.current_position,
                deadzone_radius: config.deadzone,
                engaged: self.primary_down,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use tron_api::{GestureFrame, GestureSignal, PalmPose2d};

    use super::*;

    fn input(position: Point2d, gesture: HandGesture, signals: Vec<GestureSignal>) -> PointerInput {
        PointerInput {
            gesture: GestureFrame {
                timestamp: Instant::now(),
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
        }
    }

    #[test]
    fn joystick_pointer_emits_periodic_relative_move_after_pinch() {
        let mut state = JoystickPointerState::default();
        let config = JoystickPointerProducer::default();

        let events = state.update_input(
            input(
                Point2d::new(0.5, 0.5),
                HandGesture::Pinch,
                vec![signal(HandGesture::Pinch, 0.8, Point2d::new(0.52, 0.48))],
            ),
            config,
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            PointerOutput::Event(PointerEvent::Down { .. })
        ));
        assert!(matches!(
            events[1],
            PointerOutput::Visualization(PointerVisualization::Joystick(
                PointerJoystickVisualization {
                    anchor: Some(anchor),
                    engaged: true,
                    ..
                }
            )) if anchor == Point2d::new(0.52, 0.48)
        ));

        let events = state.update_input(
            input(
                Point2d::new(0.65, 0.5),
                HandGesture::Pinch,
                vec![signal(HandGesture::Pinch, 0.8, Point2d::new(0.70, 0.48))],
            ),
            config,
        );
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            PointerOutput::Visualization(PointerVisualization::Joystick(
                PointerJoystickVisualization {
                    current: Some(current),
                    engaged: true,
                    ..
                }
            )) if current == Point2d::new(0.70, 0.48)
        ));

        let events = state.tick(config);
        assert_eq!(events.len(), 1);
        let PointerOutput::Event(PointerEvent::Move {
            position, delta, ..
        }) = events[0]
        else {
            panic!("expected relative move");
        };
        assert_eq!(position, None);
        assert!(delta.x > 0.0);
        assert_eq!(delta.y, 0.0);
    }

    #[test]
    fn joystick_pointer_releases_when_pinch_strength_drops() {
        let mut state = JoystickPointerState::default();
        let config = JoystickPointerProducer::default();
        state.update_input(
            input(
                Point2d::new(0.5, 0.5),
                HandGesture::Pinch,
                vec![signal(HandGesture::Pinch, 0.8, Point2d::new(0.5, 0.5))],
            ),
            config,
        );

        let events = state.update_input(
            input(Point2d::new(0.5, 0.5), HandGesture::OpenPalm, vec![]),
            config,
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            PointerOutput::Event(PointerEvent::Up { .. })
        ));
        assert!(matches!(
            events[1],
            PointerOutput::Visualization(PointerVisualization::Joystick(
                PointerJoystickVisualization {
                    anchor: None,
                    engaged: false,
                    ..
                }
            ))
        ));
        assert!(state.tick(config).is_empty());
    }
}
