use std::time::Instant;

use crate::Point2d;

#[derive(Clone, Debug, serde::Serialize)]
pub struct GestureFrame {
    #[serde(skip)]
    pub timestamp: Instant,
    pub palm: Option<PalmPose2d>,
    pub signals: Vec<GestureSignal>,
    pub gesture: HandGesture,
}

impl GestureFrame {
    pub fn signal(&self, gesture: HandGesture) -> Option<GestureSignal> {
        self.signals
            .iter()
            .copied()
            .find(|signal| signal.gesture == gesture)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct PalmPose2d {
    /// Normalized frame-space center, where x/y are expected to be in 0..=1.
    pub center: Point2d,
    /// Rotation in radians in image/frame coordinates.
    pub rotation_radians: f64,
    /// Normalized palm extent. Implementations can treat this as a scale cue.
    pub extent: Point2d,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct GestureSignal {
    pub gesture: HandGesture,
    pub strength: f32,
    /// Normalized frame-space position, where x/y are expected to be in 0..=1.
    pub position: Point2d,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HandGesture {
    NoHand,
    OpenPalm,
    Clutch,
    Pinch,
    Pointing,
    Unknown,
}
