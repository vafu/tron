use crate::pipeline::FrameContext;
use crate::types::{HandLandmarks, RectNorm};

pub mod mediapipe;
pub mod mock;

/// Landmark extraction stage.
///
/// Implementations own model-specific preprocessing and output decoding. ONNX
/// Runtime setup should go through `crate::inference` so GPU/provider changes
/// stay independent from model decoding code.
pub trait HandLandmarker: Send {
    fn run(&mut self, ctx: &FrameContext, roi: Option<RectNorm>) -> Option<HandLandmarks>;
}
