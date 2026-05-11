use anyhow::Result;

use crate::frame::FrameMut;
use crate::render::NoContext;

pub trait Processor<I, C = NoContext> {
    type Output;

    fn process(&mut self, input: I, context: C) -> Result<Self::Output>;
}

// TODO: This is intentionally unused until a stage has a real writable-frame
// use case. A likely production use is IR tinting directly into a working RGB
// frame when we do not need to preserve or display the original RGB pixels.
pub trait InPlaceFrameProcessor<C = NoContext> {
    fn process_in_place(&mut self, frame: &mut FrameMut<'_>, context: C) -> Result<()>;
}
