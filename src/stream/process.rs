use anyhow::Result;

use crate::stream::frame::{Frame, FrameMeta};

pub struct FrameContext<'a> {
    pub meta: FrameMeta,
    pub frame: Option<Frame<'a>>,
}

pub trait FrameProcessor {
    type Output;

    fn process(&mut self, context: &FrameContext<'_>) -> Result<Self::Output>;
}
