use anyhow::Result;

use crate::frame::CapturedFrame;

pub trait FrameSource {
    fn next_frame(&mut self) -> Result<CapturedFrame<'_>>;
}
