use std::time::Instant;

use crate::{GestureFrame, Point2d, Sink};

#[derive(Clone, Debug)]
pub struct PointerInput {
    pub gesture: GestureFrame,
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
