use crate::{NoContext, Processor, Rect, Size};
use glam::Vec2;

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
    pub corners: [Vec2; 4],
}

impl OrientedBoundingBox {
    pub fn enclosing_rect(self, bounds: Size) -> Option<Rect> {
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for corner in self.corners {
            if !corner.is_finite() {
                return None;
            }
            min = min.min(corner);
            max = max.max(corner);
        }
        let bounds = bounds.as_uvec2().as_vec2();
        let min = min.floor().clamp(Vec2::ZERO, bounds).as_uvec2();
        let max = max.ceil().clamp(Vec2::ZERO, bounds).as_uvec2();
        let size = max.saturating_sub(min);
        let width = size.x;
        let height = size.y;
        if width == 0 || height == 0 {
            return None;
        }
        Some(Rect::new(min.x, min.y, Size::from_uvec2(size)))
    }
}

pub trait RoiProcessor<I, C = NoContext>: Processor<I, C, Output = Option<RoiResult>> {}

impl<T, I, C> RoiProcessor<I, C> for T where T: Processor<I, C, Output = Option<RoiResult>> {}
