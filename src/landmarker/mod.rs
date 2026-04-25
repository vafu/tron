use crate::pipeline::FrameContext;
use crate::types::{HandLandmarks, RectNorm};

pub mod mock;

pub trait HandLandmarker: Send {
    fn run(&mut self, ctx: &FrameContext, roi: Option<RectNorm>) -> Option<HandLandmarks>;
}
