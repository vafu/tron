use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tron_api::{
    EventProducer, HandGesture, Point2d, PointerCancelReason, PointerEvent, PointerInput,
    PointerJoystickVisualization, PointerOutput, PointerVisualization,
};

#[derive(Clone, Copy, Debug)]
pub struct RelativePointerProducer {
    pub clutch_down_strength: f32,
    pub clutch_up_strength: f32,
    pub pinch_down_strength: f32,
    pub pinch_up_strength: f32,
    pub deadzone: f64,
    pub sensitivity: f64,
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
            while let Some(input) = input.recv().await {
                for output_item in state.update_input(input, self) {
                    if output.send(output_item).await.is_err() {
                        return Ok(());
                    }
                }
            }
            Ok(())
        })
    }
}

#[derive(Default)]
struct RelativePointerState {
    anchor_position: Option<Point2d>,
    current_position: Option<Point2d>,
    last_offset: Point2d,
    clutch_engaged: bool,
    button_down: bool,
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

        let clutch = input.gesture.signal(HandGesture::Clutch).map(|signal| {
            (
                signal.strength,
                signal.position.clamp(Point2d::ZERO, Point2d::ONE),
            )
        });
        if let Some((_, position)) = clutch {
            self.current_position = Some(position);
        }
        let pinch_strength = input
            .gesture
            .signal(HandGesture::Pinch)
            .map(|signal| signal.strength)
            .unwrap_or(0.0);

        let mut events = Vec::new();
        match (self.clutch_engaged, clutch) {
            (false, Some((strength, position))) if strength >= config.clutch_down_strength => {
                self.clutch_engaged = true;
                self.anchor_position = Some(position);
                self.current_position = Some(position);
                self.last_offset = Point2d::ZERO;
            }
            (true, Some((strength, _))) if strength <= config.clutch_up_strength => {
                self.clutch_engaged = false;
                self.anchor_position = None;
                self.current_position = None;
                self.last_offset = Point2d::ZERO;
            }
            (true, Some((_, position))) => {
                if let Some(anchor) = self.anchor_position {
                    let offset = active_offset(position - anchor, config);
                    let delta = offset - self.last_offset;
                    self.last_offset = offset;
                    if delta != Point2d::ZERO {
                        events.push(PointerOutput::Event(PointerEvent::Move {
                            timestamp,
                            position: None,
                            delta,
                        }));
                    }
                }
            }
            (true, None) => {
                self.clutch_engaged = false;
                self.anchor_position = None;
                self.current_position = None;
                self.last_offset = Point2d::ZERO;
            }
            _ => {}
        }

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

    fn cancel(
        &mut self,
        timestamp: std::time::Instant,
        config: RelativePointerProducer,
    ) -> Vec<PointerOutput> {
        let was_button_down = self.button_down;
        self.anchor_position = None;
        self.current_position = None;
        self.last_offset = Point2d::ZERO;
        self.clutch_engaged = false;
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

    fn visualization(
        &self,
        timestamp: std::time::Instant,
        config: RelativePointerProducer,
    ) -> PointerOutput {
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
    fn relative_pointer_uses_clutch_home_as_origin_without_button_down() {
        let mut state = RelativePointerState::default();
        let config = RelativePointerProducer {
            deadzone: 0.0,
            ..RelativePointerProducer::default()
        };

        let events = state.update_input(
            input(
                Point2d::new(0.5, 0.5),
                HandGesture::Clutch,
                vec![signal(HandGesture::Clutch, 0.8, Point2d::new(0.52, 0.48))],
            ),
            config,
        );
        assert!(matches!(
            events[0],
            PointerOutput::Visualization(PointerVisualization::Joystick(_))
        ));

        let events = state.update_input(
            input(
                Point2d::new(0.5, 0.5),
                HandGesture::Clutch,
                vec![signal(HandGesture::Clutch, 0.8, Point2d::new(0.62, 0.43))],
            ),
            config,
        );
        let PointerOutput::Event(PointerEvent::Move {
            position, delta, ..
        }) = events[0]
        else {
            panic!("expected relative move");
        };
        assert_eq!(position, None);
        assert_point_near(delta, Point2d::new(0.10, -0.05));

        let events = state.update_input(
            input(
                Point2d::new(0.5, 0.5),
                HandGesture::Clutch,
                vec![signal(HandGesture::Clutch, 0.8, Point2d::new(0.57, 0.46))],
            ),
            config,
        );
        let PointerOutput::Event(PointerEvent::Move { delta, .. }) = events[0] else {
            panic!("expected relative move");
        };
        assert_point_near(delta, Point2d::new(-0.05, 0.03));
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
            events[0],
            PointerOutput::Event(PointerEvent::Move { .. })
        ));
        assert!(matches!(
            events[1],
            PointerOutput::Event(PointerEvent::Up { .. })
        ));
    }

    fn assert_point_near(actual: Point2d, expected: Point2d) {
        assert!(
            (actual - expected).length() < 1.0e-12,
            "actual={actual:?} expected={expected:?}"
        );
    }
}
