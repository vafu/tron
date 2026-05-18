pub mod calib;
pub mod capture;
pub mod depth;
pub mod event;
pub mod frame;
pub mod gesture;
pub mod pointer;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub size: Size,
}

impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn as_uvec2(self) -> glam::UVec2 {
        glam::UVec2::new(self.width, self.height)
    }

    pub const fn from_uvec2(size: glam::UVec2) -> Self {
        Self::new(size.x, size.y)
    }
}

impl Rect {
    pub const fn new(x: u32, y: u32, size: Size) -> Self {
        Self { x, y, size }
    }

    pub const fn as_uvec4(self) -> glam::UVec4 {
        glam::UVec4::new(self.x, self.y, self.size.width, self.size.height)
    }

    pub const fn from_uvec4(rect: glam::UVec4) -> Self {
        Self::new(rect.x, rect.y, Size::new(rect.z, rect.w))
    }

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
pub use event::{EventProducer, EventProducerChannels, spawn_event_channels};
pub use frame::{
    CaptureFormat, Frame, FrameId, FrameMeta, FrameMut, FrameSource, FrameTimestamp,
    IterableFrameSource, OwnedFrame, PixelFormat, SensorKind, TimestampSource,
};
pub use gesture::{GestureFrame, HandGesture, PalmPose2d};
pub use pointer::{
    PointerCancelReason, PointerEvent, PointerInput, PointerJoystickVisualization, PointerOutput,
    PointerSink, PointerVisualization,
};
pub use process::{InPlaceFrameProcessor, Processor};
pub use projection::{DepthPointProjection, DepthProjectionMap, ProjectionMapSource};
pub use render::{NoContext, Sink};
pub use roi::{OrientedBoundingBox, RoiCandidate, RoiProcessor, RoiResult};
pub use view_buffer::{ViewBuffer, ViewBufferMut};
