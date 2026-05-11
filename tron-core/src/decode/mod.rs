pub mod mjpeg;

use anyhow::Result;
use tron_api::{Frame, FrameMeta};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodedFormat {
    Mjpeg,
}

#[derive(Clone, Copy, Debug)]
pub struct EncodedFrame<'a> {
    pub meta: FrameMeta,
    pub format: EncodedFormat,
    pub data: &'a [u8],
}

pub trait FrameDecoder {
    fn decode<'a>(&'a mut self, frame: EncodedFrame<'_>) -> Result<Frame<'a>>;
}
