use anyhow::Result;

use crate::stream::frame::{CaptureFormat, CapturedFrame, SensorKind};

pub mod v4l;

#[derive(Clone, Debug)]
pub struct SourceConfig {
    pub path: String,
    pub sensor: SensorKind,
    pub format: CaptureFormat,
    pub width: u32,
    pub height: u32,
    pub fps: Option<u32>,
    pub buffers: u32,
}

impl SourceConfig {
    pub fn new(
        path: impl Into<String>,
        sensor: SensorKind,
        format: CaptureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            path: path.into(),
            sensor,
            format,
            width,
            height,
            fps: None,
            buffers: 4,
        }
    }

    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = Some(fps);
        self
    }

    pub fn with_buffers(mut self, buffers: u32) -> Self {
        self.buffers = buffers;
        self
    }
}

pub trait FrameSource {
    fn next_frame(&mut self) -> Result<CapturedFrame<'_>>;
}
