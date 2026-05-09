use anyhow::Result;
use tron_api::{CapturedFrame, Frame, FrameDecoder, FrameSource};

pub trait FrameStream {
    fn next_frame(&mut self) -> Result<Option<Frame<'_>>>;
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
    fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        match self.source.next_frame()? {
            Some(CapturedFrame::Frame(frame)) => Ok(Some(frame)),
            Some(CapturedFrame::Encoded(frame)) => anyhow::bail!(
                "passthrough stream received encoded frame {:?} from sensor {:?}",
                frame.format,
                frame.meta.sensor
            ),
            None => Ok(None),
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
    fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        match self.source.next_frame()? {
            Some(CapturedFrame::Encoded(frame)) => self.decoder.decode(frame).map(Some),
            Some(CapturedFrame::Frame(frame)) => anyhow::bail!(
                "decode stream received pixel frame {:?} from sensor {:?}",
                frame.format,
                frame.meta.sensor
            ),
            None => Ok(None),
        }
    }
}
