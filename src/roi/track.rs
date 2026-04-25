use super::RoiHinter;
use crate::pipeline::FrameContext;
use crate::types::{HandLandmarks, RectNorm};

/// Build a ROI from the previous frame's landmarks: their bounding box, padded.
pub struct TrackFromLastRoi {
    pub pad: f32,
}

impl TrackFromLastRoi {
    pub fn new() -> Self {
        Self { pad: 0.25 }
    }
}

impl Default for TrackFromLastRoi {
    fn default() -> Self {
        Self::new()
    }
}

impl RoiHinter for TrackFromLastRoi {
    fn hint(&mut self, ctx: &FrameContext) -> Option<RectNorm> {
        let lm = ctx.last?;
        Some(landmarks_bbox(lm).padded(self.pad))
    }
}

fn landmarks_bbox(lm: &HandLandmarks) -> RectNorm {
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in &lm.points {
        if p.x < min_x { min_x = p.x; }
        if p.y < min_y { min_y = p.y; }
        if p.x > max_x { max_x = p.x; }
        if p.y > max_y { max_y = p.y; }
    }
    RectNorm {
        x: min_x.clamp(0.0, 1.0),
        y: min_y.clamp(0.0, 1.0),
        w: (max_x - min_x).clamp(0.0, 1.0),
        h: (max_y - min_y).clamp(0.0, 1.0),
    }
}
