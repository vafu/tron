use anyhow::Result;

use crate::stream::frame::{EncodedFrame, Frame};

pub mod mjpeg;

pub trait FrameDecoder {
    fn decode<'a>(&'a mut self, frame: EncodedFrame<'_>) -> Result<Frame<'a>>;
}
