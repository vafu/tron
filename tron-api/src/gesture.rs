use std::time::Instant;

use crate::Point2d;

#[derive(Clone, Debug)]
pub struct GestureFrame {
    pub timestamp: Instant,
    pub palm: Option<PalmPose2d>,
    pub gesture: HandGesture,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PalmPose2d {
    /// Normalized frame-space center, where x/y are expected to be in 0..=1.
    pub center: Point2d,
    /// Rotation in radians in image/frame coordinates.
    pub rotation_radians: f64,
    /// Normalized palm extent. Implementations can treat this as a scale cue.
    pub extent: Point2d,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HandGesture {
    NoHand,
    OpenPalm,
    Pinch { strength: f32, position: Point2d },
    Pointing,
    Unknown,
}
