use anyhow::Result;

use crate::frame::{CaptureFormat, CapturedFrame, FrameSize, SensorKind};

#[derive(Clone, Debug)]
pub struct CameraSelector {
    pub id: Option<String>,
    pub name: Option<String>,
    pub sensor: SensorKind,
}

#[derive(Clone, Debug)]
pub struct CameraOpenRequest {
    pub selector: CameraSelector,
    pub format: Option<CaptureFormat>,
    pub size: Option<FrameSize>,
    pub fps: Option<u32>,
    pub buffers: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct OpenedCameraInfo {
    pub id: String,
    pub sensor: SensorKind,
    pub format: CaptureFormat,
    pub size: FrameSize,
}

pub trait CameraOpener {
    type Source: FrameSource;

    fn open(&self, request: CameraOpenRequest) -> Result<Self::Source>;
}

pub trait FrameSource {
    fn info(&self) -> &OpenedCameraInfo;

    fn next_frame(&mut self) -> Result<Option<CapturedFrame<'_>>>;
}
