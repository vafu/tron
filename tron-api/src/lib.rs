pub mod calib;
pub mod capture;
pub mod depth;
pub mod frame;
pub mod process;
pub mod projection;
pub mod render;
pub mod roi;
pub mod view_buffer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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

pub type Point2d = glam::DVec2;
pub type Point3d = glam::DVec3;

pub use calib::{
    CalibrationFrameSide, CameraCalibration, CheckerboardDetection, CheckerboardProcessor,
    CheckerboardSample, CheckerboardSpec, CheckerboardStereoCalibration,
    StereoCalibrationProcessor,
};
pub use capture::{
    CameraOpenRequest, CameraOpener, CameraRoiControl, CameraSelector, OpenedCameraInfo,
};
pub use depth::{DepthSample, DepthSource};
pub use frame::{
    CaptureFormat, Frame, FrameId, FrameMeta, FrameMut, FrameSource, FrameTimestamp, OwnedFrame,
    PixelFormat, SensorKind, TimestampSource,
};
pub use process::{InPlaceFrameProcessor, Processor};
pub use projection::{DepthPointProjection, DepthProjectionMap, ProjectionMapSource};
pub use render::{NoContext, Renderer};
pub use roi::{OrientedBoundingBox, RoiCandidate, RoiProcessor, RoiResult};
pub use view_buffer::{ViewBuffer, ViewBufferMut};
