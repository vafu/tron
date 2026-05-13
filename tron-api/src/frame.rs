use std::time::Instant;

use anyhow::Result;
use ndarray::ArrayView3;

use crate::view_buffer::{ViewBuffer, ViewBufferMut};
use crate::{OpenedCameraInfo, Rect, Size};

pub type FrameId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorKind {
    Rgb,
    Ir,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureFormat {
    Mjpeg,
    Gray8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum PixelFormat {
    Gray8 = 1,
    Bgra8 = 4,
}

impl PixelFormat {
    pub const fn channels(self) -> usize {
        self as usize
    }
}

impl TryFrom<CaptureFormat> for PixelFormat {
    type Error = anyhow::Error;

    fn try_from(format: CaptureFormat) -> Result<Self, Self::Error> {
        match format {
            CaptureFormat::Gray8 => Ok(Self::Gray8),
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
    pub size: Size,
    pub timestamp: FrameTimestamp,
    pub sequence: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    pub meta: FrameMeta,
    pub format: PixelFormat,
    pub buffer: ViewBuffer<'a>,
}

impl<'a> Frame<'a> {
    pub fn new(
        meta: FrameMeta,
        format: PixelFormat,
        stride: usize,
        data: &'a [u8],
    ) -> Result<Self> {
        Ok(Self {
            meta,
            format,
            buffer: ViewBuffer::new(format, meta.size, stride, data)?,
        })
    }

    pub fn roi(self, rect: Rect) -> Result<Self> {
        anyhow::ensure!(
            rect.x <= self.meta.size.width
                && rect.y <= self.meta.size.height
                && rect.size.width <= self.meta.size.width.saturating_sub(rect.x)
                && rect.size.height <= self.meta.size.height.saturating_sub(rect.y),
            "ROI {:?} is outside frame size {:?}",
            rect,
            self.meta.size
        );
        Ok(Self {
            meta: FrameMeta {
                size: rect.size,
                ..self.meta
            },
            format: self.format,
            buffer: self.buffer.roi(rect)?,
        })
    }

    pub fn mirrored(self, horizontal: bool, vertical: bool) -> Result<Self> {
        Ok(Self {
            meta: self.meta,
            format: self.format,
            buffer: self.buffer.mirrored(horizontal, vertical)?,
        })
    }

    pub fn view(&self) -> Result<ArrayView3<'a, u8>> {
        self.buffer.view()
    }
}

#[derive(Debug)]
pub struct FrameMut<'a> {
    pub meta: FrameMeta,
    pub format: PixelFormat,
    pub buffer: ViewBufferMut<'a>,
}

impl FrameMut<'_> {
    pub fn as_frame(&self) -> Frame<'_> {
        Frame {
            meta: self.meta,
            format: self.format,
            buffer: self.buffer.as_view_buffer(),
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
        Frame::new(self.meta, self.format, self.stride, &self.data)
            .expect("owned frame metadata must match backing buffer")
    }

    pub fn as_frame_mut(&mut self) -> FrameMut<'_> {
        FrameMut {
            meta: self.meta,
            format: self.format,
            buffer: ViewBufferMut {
                format: self.format,
                size: self.meta.size,
                stride: self.stride,
                data: &mut self.data,
            },
        }
    }
}

pub fn buffer_size(format: PixelFormat, size: Size) -> Result<Size> {
    let width = row_bytes(format, size.width)?;
    anyhow::ensure!(
        width <= u32::MAX as usize,
        "buffer row width {} does not fit u32",
        width
    );
    Ok(Size {
        width: width as u32,
        height: size.height,
    })
}

pub fn row_bytes(format: PixelFormat, width: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(format.channels())
        .ok_or_else(|| anyhow::anyhow!("{format:?} row byte width overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(width: u32, height: u32) -> FrameMeta {
        FrameMeta {
            id: 1,
            sensor: SensorKind::Ir,
            size: Size { width, height },
            timestamp: FrameTimestamp {
                camera_monotonic_us: None,
                source: TimestampSource::Unknown,
                received_at: Instant::now(),
            },
            sequence: None,
        }
    }

    #[test]
    fn frame_roi_exposes_strided_rows_without_copying() {
        let data = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23];
        let frame = Frame::new(meta(4, 3), PixelFormat::Gray8, 4, &data)
            .unwrap()
            .roi(Rect {
                x: 1,
                y: 1,
                size: Size {
                    width: 2,
                    height: 2,
                },
            })
            .unwrap();

        assert_eq!(
            frame.meta.size,
            Size {
                width: 2,
                height: 2,
            }
        );
        assert_eq!(
            frame.buffer.size(),
            Size {
                width: 2,
                height: 2,
            }
        );
        assert_eq!(frame.buffer.stride(), 4);
        let view = frame.view().unwrap();
        assert_eq!(view[[0, 0, 0]], 11);
        assert_eq!(view[[0, 1, 0]], 12);
        assert_eq!(view[[1, 0, 0]], 21);
        assert_eq!(view[[1, 1, 0]], 22);
    }

    #[test]
    fn frame_roi_preserves_pixel_size_for_bgra_view() {
        let data = [0_u8; 4 * 4 * 3];
        let frame = Frame::new(meta(4, 3), PixelFormat::Bgra8, 16, &data)
            .unwrap()
            .roi(Rect {
                x: 1,
                y: 1,
                size: Size {
                    width: 2,
                    height: 2,
                },
            })
            .unwrap();

        assert_eq!(
            frame.meta.size,
            Size {
                width: 2,
                height: 2,
            }
        );
        assert_eq!(
            frame.buffer.size(),
            Size {
                width: 2,
                height: 2,
            }
        );
        assert_eq!(frame.buffer.stride(), 16);
    }

    #[test]
    fn frame_roi_rejects_out_of_bounds_rect() {
        let data = [0; 12];
        let err = Frame::new(meta(4, 3), PixelFormat::Gray8, 4, &data)
            .unwrap()
            .roi(Rect {
                x: 3,
                y: 1,
                size: Size {
                    width: 2,
                    height: 1,
                },
            })
            .unwrap_err();
        assert!(err.to_string().contains("outside frame size"));
    }
}

// TODO: FrameSource will also cover non-camera inputs such as file-backed
// streams. This should become a more general StreamInfo enum that can compose
// OpenedCameraInfo, FileInfo, and future source-specific metadata.
#[async_trait::async_trait]
pub trait FrameSource {
    fn info(&self) -> &OpenedCameraInfo;

    async fn next_frame(&mut self) -> anyhow::Result<Option<Frame<'_>>>;
}
