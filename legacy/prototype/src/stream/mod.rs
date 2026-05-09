use anyhow::Result;

use crate::stream::decode::FrameDecoder;
use crate::stream::frame::{CapturedFrame, Frame};
use crate::stream::source::FrameSource;

pub mod decode;
pub mod frame;
pub mod process;
pub mod render;
pub mod source;

pub trait FrameStream {
    fn next_frame(&mut self) -> Result<Frame<'_>>;
}

pub struct PassthroughStream<S> {
    source: S,
}

impl<S> PassthroughStream<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

impl<S> FrameStream for PassthroughStream<S>
where
    S: FrameSource,
{
    fn next_frame(&mut self) -> Result<Frame<'_>> {
        match self.source.next_frame()? {
            CapturedFrame::Frame(frame) => Ok(frame),
            CapturedFrame::Encoded(frame) => anyhow::bail!(
                "passthrough stream received encoded frame {:?} from sensor {:?}",
                frame.format,
                frame.meta.sensor
            ),
        }
    }
}

pub struct DecodeStream<S, D> {
    source: S,
    decoder: D,
}

impl<S, D> DecodeStream<S, D> {
    pub fn new(source: S, decoder: D) -> Self {
        Self { source, decoder }
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    pub fn decoder(&self) -> &D {
        &self.decoder
    }

    pub fn decoder_mut(&mut self) -> &mut D {
        &mut self.decoder
    }
}

impl<S, D> FrameStream for DecodeStream<S, D>
where
    S: FrameSource,
    D: FrameDecoder,
{
    fn next_frame(&mut self) -> Result<Frame<'_>> {
        match self.source.next_frame()? {
            CapturedFrame::Encoded(frame) => self.decoder.decode(frame),
            CapturedFrame::Frame(frame) => anyhow::bail!(
                "decode stream received pixel frame {:?} from sensor {:?}",
                frame.format,
                frame.meta.sensor
            ),
        }
    }
}
