use anyhow::Result;

use crate::capture::OpenedCameraInfo;
use crate::frame::CapturedFrame;

pub trait FrameSource {
    fn info(&self) -> &OpenedCameraInfo;

    fn next_frame(&mut self) -> Result<Option<CapturedFrame<'_>>>;
}
