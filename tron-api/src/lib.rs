pub mod calib;
pub mod capture;
pub mod decode;
pub mod frame;
pub mod present;
pub mod process;
pub mod roi;
pub mod stream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub size: Size,
}

impl Rect {
    pub fn clamp_to(self, bounds: Size) -> Self {
        let width = self.size.width.min(bounds.width).max(1);
        let height = self.size.height.min(bounds.height).max(1);
        let x = self.x.min(bounds.width.saturating_sub(width));
        let y = self.y.min(bounds.height.saturating_sub(height));
        Self {
            x,
            y,
            size: Size { width, height },
        }
    }

    pub fn non_empty_or(self, size: Size) -> Self {
        if self.size.width > 0 && self.size.height > 0 {
            self
        } else {
            Self {
                x: self.x,
                y: self.y,
                size,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    pub meta: frame::FrameMeta,
    pub format: frame::PixelFormat,
    pub size: Size,
    pub stride: usize,
    pub data: &'a [u8],
}

pub use calib::{
    CalibrationFrameSide, CheckerboardDetection, CheckerboardProcessor, CheckerboardSample,
    CheckerboardSpec, StereoCalibrationProcessor,
};
pub use capture::{
    CameraOpenRequest, CameraOpener, CameraRoiControl, CameraSelector, OpenedCameraInfo,
};
pub use decode::FrameDecoder;
pub use frame::{
    CaptureFormat, CapturedFrame, EncodedFormat, EncodedFrame, Frame, FrameId, FrameMeta, FrameMut,
    FrameTimestamp, OwnedFrame, PixelFormat, SensorKind, TimestampSource,
};
pub use present::{NoContext, Presenter};
pub use process::{InPlaceFrameProcessor, Processor};
pub use roi::{RoiCandidate, RoiProcessor, RoiResult};
pub use stream::FrameSource;
