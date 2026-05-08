use anyhow::Result;

use crate::stream::frame::{EncodedFrame, Frame, OwnedFrame};

pub mod mjpeg;

pub enum DecodeOutput<'a> {
    Borrowed(Frame<'a>),
    Owned(OwnedFrame),
}

impl<'a> DecodeOutput<'a> {
    pub fn as_frame(&'a self) -> Frame<'a> {
        match self {
            DecodeOutput::Borrowed(frame) => *frame,
            DecodeOutput::Owned(frame) => frame.as_frame(),
        }
    }
}

pub trait FrameDecoder {
    fn decode<'a>(&'a mut self, frame: EncodedFrame<'_>) -> Result<DecodeOutput<'a>>;
}
