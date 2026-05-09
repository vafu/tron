use std::time::Instant;

pub type FrameId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorKind {
    Rgb,
    Ir,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureFormat {
    Mjpeg,
    Gray8,
    Yuyv422,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodedFormat {
    Mjpeg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Gray8,
    Bgra8,
    Yuyv422,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampSource {
    StartOfExposure,
    EndOfFrame,
    Driver,
    Unknown,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameTimestamp {
    pub camera_monotonic_us: Option<i64>,
    pub source: TimestampSource,
    pub received_at: Instant,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameMeta {
    pub id: FrameId,
    pub sensor: SensorKind,
    pub width: u32,
    pub height: u32,
    pub timestamp: FrameTimestamp,
    pub sequence: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct EncodedFrame<'a> {
    pub meta: FrameMeta,
    pub format: EncodedFormat,
    pub data: &'a [u8],
}

#[derive(Clone, Copy, Debug, derive_more::From)]
pub enum CapturedFrame<'a> {
    Encoded(EncodedFrame<'a>),
    Frame(Frame<'a>),
}

#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    pub meta: FrameMeta,
    pub format: PixelFormat,
    pub stride: usize,
    pub data: &'a [u8],
}

#[derive(Debug)]
pub struct FrameMut<'a> {
    pub meta: FrameMeta,
    pub format: PixelFormat,
    pub stride: usize,
    pub data: &'a mut [u8],
}

impl FrameMut<'_> {
    pub fn as_frame(&self) -> Frame<'_> {
        Frame {
            meta: self.meta,
            format: self.format,
            stride: self.stride,
            data: self.data,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OwnedFrame {
    pub meta: FrameMeta,
    pub format: PixelFormat,
    pub stride: usize,
    pub data: Vec<u8>,
}

impl OwnedFrame {
    pub fn as_frame(&self) -> Frame<'_> {
        Frame {
            meta: self.meta,
            format: self.format,
            stride: self.stride,
            data: &self.data,
        }
    }

    pub fn as_frame_mut(&mut self) -> FrameMut<'_> {
        FrameMut {
            meta: self.meta,
            format: self.format,
            stride: self.stride,
            data: &mut self.data,
        }
    }
}
