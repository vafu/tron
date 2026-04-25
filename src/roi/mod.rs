use crate::pipeline::FrameContext;
use crate::types::RectNorm;

pub mod ir;
pub mod track;

pub trait RoiHinter: Send {
    fn hint(&mut self, ctx: &FrameContext) -> Option<RectNorm>;
}

/// Combinator: try each hinter in order; return the first `Some`.
pub struct CompositeRoiHinter(pub Vec<Box<dyn RoiHinter>>);

impl RoiHinter for CompositeRoiHinter {
    fn hint(&mut self, ctx: &FrameContext) -> Option<RectNorm> {
        for h in &mut self.0 {
            if let Some(r) = h.hint(ctx) {
                return Some(r);
            }
        }
        None
    }
}

/// Always returns the full frame — useful as a terminal fallback in a composite.
pub struct FullFrameRoi;

impl RoiHinter for FullFrameRoi {
    fn hint(&mut self, _ctx: &FrameContext) -> Option<RectNorm> {
        Some(RectNorm::FULL)
    }
}
