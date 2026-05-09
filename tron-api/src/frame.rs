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

impl TryFrom<CaptureFormat> for PixelFormat {
    type Error = anyhow::Error;

    fn try_from(format: CaptureFormat) -> Result<Self, Self::Error> {
        match format {
            CaptureFormat::Gray8 => Ok(Self::Gray8),
            CaptureFormat::Yuyv422 => Ok(Self::Yuyv422),
            CaptureFormat::Mjpeg => {
                anyhow::bail!("MJPEG is encoded and cannot be converted to a pixel format")
            }
        }
    }
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

#[derive(Clone, Copy, Debug)]
pub enum CapturedFrame<'a> {
    Encoded(EncodedFrame<'a>),
    Frame(Frame<'a>),
}

impl<'a> From<EncodedFrame<'a>> for CapturedFrame<'a> {
    fn from(value: EncodedFrame<'a>) -> Self {
        Self::Encoded(value)
    }
}

impl<'a> From<Frame<'a>> for CapturedFrame<'a> {
    fn from(value: Frame<'a>) -> Self {
        Self::Frame(value)
    }
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
