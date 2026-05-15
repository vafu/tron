use crate::{NoContext, Processor, Rect, Size};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct RoiCandidate {
    pub rect: Rect,
    pub area: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct RoiResult {
    pub rect: Rect,
    pub oriented_box: Option<OrientedBoundingBox>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct OrientedBoundingBox {
    pub corners: [[f32; 2]; 4],
}

impl OrientedBoundingBox {
    pub fn enclosing_rect(self, bounds: Size) -> Option<Rect> {
        let mut x0 = f32::INFINITY;
        let mut y0 = f32::INFINITY;
        let mut x1 = f32::NEG_INFINITY;
        let mut y1 = f32::NEG_INFINITY;
        for [x, y] in self.corners {
            if !x.is_finite() || !y.is_finite() {
                return None;
            }
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let x0 = x0.floor().max(0.0).min(bounds.width as f32) as u32;
        let y0 = y0.floor().max(0.0).min(bounds.height as f32) as u32;
        let x1 = x1.ceil().max(0.0).min(bounds.width as f32) as u32;
        let y1 = y1.ceil().max(0.0).min(bounds.height as f32) as u32;
        let width = x1.saturating_sub(x0);
        let height = y1.saturating_sub(y0);
        if width == 0 || height == 0 {
            return None;
        }
        Some(Rect {
            x: x0,
            y: y0,
            size: Size { width, height },
        })
    }
}

pub trait RoiProcessor<I, C = NoContext>: Processor<I, C, Output = Option<RoiResult>> {}

impl<T, I, C> RoiProcessor<I, C> for T where T: Processor<I, C, Output = Option<RoiResult>> {}
