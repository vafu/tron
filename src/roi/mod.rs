use crate::pipeline::FrameContext;
use crate::types::{Image, RectNorm};

pub mod detector;
pub mod track;

/// ROI acquisition/tracking stage.
///
/// A hinter may run a detector, track from previous landmarks, fuse sensors, or
/// delegate to a future learned tracker. Coordinates are normalized in the RGB
/// image space expected by the landmarker.
pub trait RoiHinter: Send {
    /// Returns `(roi_hint, debug_visualization)`.
    fn hint(&mut self, ctx: &FrameContext) -> (Option<RectNorm>, Option<Image>);
}

/// Combinator: try each hinter in order; return the first `Some`.
pub struct CompositeRoiHinter {
    pub hinters: Vec<Box<dyn RoiHinter>>,
    last_winner: i32,
    frame: u64,
}

impl CompositeRoiHinter {
    pub fn new(hinters: Vec<Box<dyn RoiHinter>>) -> Self {
        Self {
            hinters,
            last_winner: -2,
            frame: 0,
        }
    }
}

impl RoiHinter for CompositeRoiHinter {
    fn hint(&mut self, ctx: &FrameContext) -> (Option<RectNorm>, Option<Image>) {
        self.frame = self.frame.wrapping_add(1);
        let mut first_debug = None;
        for (i, h) in self.hinters.iter_mut().enumerate() {
            let (r, d) = h.hint(ctx);
            if first_debug.is_none() {
                first_debug = d;
            }
            if let Some(rect) = r {
                let idx = i as i32;
                if idx != self.last_winner {
                    eprintln!(
                        "roi: winner {}->{} frame={} rect=[{:.2},{:.2} {:.2}x{:.2}]",
                        self.last_winner, idx, self.frame, rect.x, rect.y, rect.w, rect.h
                    );
                    self.last_winner = idx;
                }
                return (Some(rect), first_debug);
            }
        }
        if self.last_winner != -1 {
            eprintln!(
                "roi: winner {}->none frame={}",
                self.last_winner, self.frame
            );
            self.last_winner = -1;
        }
        (None, first_debug)
    }
}
