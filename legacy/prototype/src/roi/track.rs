use super::RoiHinter;
use crate::pipeline::FrameContext;
use crate::types::{HandLandmarks, Image, RectNorm};

/// Build a ROI from the previous frame's landmarks: their bounding box, padded.
pub struct TrackFromLastRoi {
    pub pad: f32,
    /// Minimum side-length (normalized) before we declare the track degenerate
    /// and fall through. A collapsed bbox means the previous landmarks all
    /// landed near a point — feeding that to mediapipe just produces another
    /// collapsed result, so we'd loop forever instead of letting the palm
    /// detector re-acquire.
    pub min_side: f32,
    had_last: bool,
}

impl TrackFromLastRoi {
    pub fn new() -> Self {
        Self {
            pad: 0.25,
            min_side: 0.08,
            had_last: false,
        }
    }
}

impl Default for TrackFromLastRoi {
    fn default() -> Self {
        Self::new()
    }
}

impl RoiHinter for TrackFromLastRoi {
    fn hint(&mut self, ctx: &FrameContext) -> (Option<RectNorm>, Option<Image>) {
        let lm = match &ctx.last {
            Some(l) => l,
            None => {
                if self.had_last {
                    eprintln!("track: lost — ctx.last cleared");
                    self.had_last = false;
                }
                return (None, None);
            }
        };
        if !self.had_last {
            eprintln!("track: acquired");
            self.had_last = true;
        }
        let bbox = landmarks_bbox(lm);
        if bbox.w < self.min_side || bbox.h < self.min_side {
            eprintln!(
                "track: degenerate bbox=[{:.3},{:.3} {:.3}x{:.3}] presence={:.2} — falling through",
                bbox.x, bbox.y, bbox.w, bbox.h, lm.presence
            );
            self.had_last = false;
            return (None, None);
        }
        (Some(bbox.padded(self.pad)), None)
    }
}

fn landmarks_bbox(lm: &HandLandmarks) -> RectNorm {
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in &lm.points {
        if p.x < min_x {
            min_x = p.x;
        }
        if p.y < min_y {
            min_y = p.y;
        }
        if p.x > max_x {
            max_x = p.x;
        }
        if p.y > max_y {
            max_y = p.y;
        }
    }
    RectNorm {
        x: min_x.clamp(0.0, 1.0),
        y: min_y.clamp(0.0, 1.0),
        w: (max_x - min_x).clamp(0.0, 1.0),
        h: (max_y - min_y).clamp(0.0, 1.0),
    }
}
