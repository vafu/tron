use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tron_api::{
    EventProducer, HandGesture, Point2d, PointerCancelReason, PointerEvent, PointerInput,
    PointerOutput,
};

#[derive(Clone, Copy, Debug)]
pub struct AbsolutePointerProducer {
    pub pinch_down_strength: f32,
    pub pinch_up_strength: f32,
}

impl Default for AbsolutePointerProducer {
    fn default() -> Self {
        Self {
            pinch_down_strength: 0.55,
            pinch_up_strength: 0.35,
        }
    }
}

impl EventProducer<PointerInput, PointerOutput> for AbsolutePointerProducer {
    fn spawn(
        self,
        mut input: mpsc::Receiver<PointerInput>,
        output: mpsc::Sender<PointerOutput>,
    ) -> JoinHandle<Result<()>> {
        tokio::spawn(async move {
            let mut state = AbsolutePointerState::default();
            while let Some(input) = input.recv().await {
                for event in state.update(input, self) {
                    if output.send(PointerOutput::Event(event)).await.is_err() {
                        return Ok(());
                    }
                }
            }
            Ok(())
        })
    }
}

#[derive(Default)]
struct AbsolutePointerState {
    last_position: Option<Point2d>,
    primary_down: bool,
    tracking: bool,
}

impl AbsolutePointerState {
    fn update(
        &mut self,
        input: PointerInput,
        config: AbsolutePointerProducer,
    ) -> Vec<PointerEvent> {
        let timestamp = input.gesture.timestamp;
        let Some(palm) = input.gesture.palm else {
            return self.cancel(timestamp, PointerCancelReason::LostTracking);
        };
        if matches!(input.gesture.gesture, HandGesture::NoHand) {
            return self.cancel(timestamp, PointerCancelReason::LostTracking);
        }

        let position = palm.center.clamp(Point2d::ZERO, Point2d::ONE);
        let delta = self
            .last_position
            .map(|last| position - last)
            .unwrap_or(Point2d::ZERO);
        self.last_position = Some(position);
        self.tracking = true;

        let mut events = vec![PointerEvent::Move {
            timestamp,
            position: Some(position),
            delta,
        }];

        let pinch_strength = input
            .gesture
            .signal(HandGesture::Pinch)
            .map(|signal| signal.strength)
            .unwrap_or(0.0);
        if !self.primary_down && pinch_strength >= config.pinch_down_strength {
            self.primary_down = true;
            events.push(PointerEvent::Down { timestamp });
        } else if self.primary_down && pinch_strength <= config.pinch_up_strength {
            self.primary_down = false;
            events.push(PointerEvent::Up { timestamp });
        }

        events
    }

    fn cancel(
        &mut self,
        timestamp: std::time::Instant,
        reason: PointerCancelReason,
    ) -> Vec<PointerEvent> {
        if !self.tracking && !self.primary_down {
            return Vec::new();
        }
        self.last_position = None;
        self.tracking = false;
        self.primary_down = false;
        vec![PointerEvent::Cancel { timestamp, reason }]
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
            velocity: None,
        }
    }

    #[test]
    fn absolute_pointer_emits_move_delta_and_down() {
        let mut state = AbsolutePointerState::default();
        let events = state.update(
            input(Point2d::new(0.25, 0.5), HandGesture::OpenPalm, vec![]),
            AbsolutePointerProducer::default(),
        );
        assert!(matches!(events[0], PointerEvent::Move { .. }));

        let events = state.update(
            input(
                Point2d::new(0.30, 0.5),
                HandGesture::Pinch,
                vec![signal(HandGesture::Pinch, 0.8, Point2d::new(0.30, 0.5))],
            ),
            AbsolutePointerProducer::default(),
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(events[1], PointerEvent::Down { .. }));
    }
}
