use anyhow::Result;

use crate::frame::{EncodedFrame, Frame};

pub trait FrameDecoder {
    fn decode<'a>(&'a mut self, frame: EncodedFrame<'_>) -> Result<Frame<'a>>;
}
