pub mod decode;
pub mod frame;
pub mod present;
pub mod process;
pub mod source;

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

#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    pub meta: frame::FrameMeta,
    pub format: frame::PixelFormat,
    pub size: Size,
    pub stride: usize,
    pub data: &'a [u8],
}

pub use decode::FrameDecoder;
pub use frame::{
    CaptureFormat, CapturedFrame, EncodedFormat, EncodedFrame, Frame, FrameId, FrameMeta, FrameMut,
    FrameTimestamp, OwnedFrame, PixelFormat, SensorKind, TimestampSource,
};
pub use present::{NoContext, Presenter};
pub use process::{InPlaceFrameProcessor, Processor};
pub use source::{CameraOpenRequest, CameraOpener, CameraSelector, FrameSource, OpenedCameraInfo};
