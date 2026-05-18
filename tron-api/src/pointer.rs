use std::time::{Duration, Instant};

use crate::{GestureFrame, Point2d, Sink};

#[derive(Clone, Debug)]
pub struct PointerInput {
    pub gesture: GestureFrame,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointerPredictionSample {
    pub timestamp: Instant,
    pub position: Point2d,
    pub velocity: Option<Point2d>,
}

#[derive(Clone, Copy, Debug)]
pub struct PointerPredictionInput<'a> {
    pub history: &'a [PointerPredictionSample],
    pub horizon: Duration,
}

pub trait PointerPredictor {
    fn predict(&mut self, input: PointerPredictionInput<'_>) -> Option<Point2d>;

    fn reset(&mut self) {}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerOutput {
    Event(PointerEvent),
    Visualization(PointerVisualization),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerVisualization {
    Joystick(PointerJoystickVisualization),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointerJoystickVisualization {
    pub timestamp: Instant,
    pub anchor: Option<Point2d>,
    pub current: Option<Point2d>,
    pub deadzone_radius: f64,
    pub engaged: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerEvent {
    Move {
        timestamp: Instant,
        position: Option<Point2d>,
        delta: Point2d,
    },
    Down {
        timestamp: Instant,
    },
    Up {
        timestamp: Instant,
    },
    Click {
        timestamp: Instant,
    },
    Cancel {
        timestamp: Instant,
        reason: PointerCancelReason,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerCancelReason {
    LostTracking,
    GestureChanged,
    ProducerReset,
}

pub trait PointerSink: Sink<PointerEvent> {}

impl<T> PointerSink for T where T: Sink<PointerEvent> {}
