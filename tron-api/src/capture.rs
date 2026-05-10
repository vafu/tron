use anyhow::Result;

use crate::frame::{CaptureFormat, SensorKind};
use crate::{FrameSource, Rect, Size};

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
    pub size: Option<Size>,
    pub fps: Option<u32>,
    pub buffers: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct OpenedCameraInfo {
    pub id: String,
    pub sensor: SensorKind,
    pub format: CaptureFormat,
    pub size: Size,
}

pub trait CameraOpener {
    type Source: FrameSource;

    fn open(&self, request: CameraOpenRequest) -> Result<Self::Source>;
}

pub trait CameraRoiControl {
    fn roi_rect(&self) -> Result<Rect>;

    fn set_roi_rect(&mut self, rect: Rect) -> Result<()>;

    fn set_roi_auto(&mut self, enabled: bool) -> Result<()>;
}
