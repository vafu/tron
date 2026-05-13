use std::time::Instant;

use anyhow::Result;

use crate::view_buffer::{ViewBuffer, ViewBufferMut, ViewRows};
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
    Yuyv422,
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
            buffer: ViewBuffer::new(buffer_size(format, meta.size)?, stride, data)?,
        })
    }

    pub fn row(&self, y: u32) -> Result<&'a [u8]> {
        self.buffer.row(y as usize)
    }

    pub fn rows(&self) -> ViewRows<'a, '_> {
        self.buffer.rows()
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
        let bytes_per_pixel = bytes_per_pixel(self.format)?;
        let x = rect.x as usize * bytes_per_pixel;
        Ok(Self {
            meta: FrameMeta {
                size: rect.size,
                ..self.meta
            },
            format: self.format,
            buffer: self
                .buffer
                .roi(x, rect.y as usize, buffer_size(self.format, rect.size)?)?,
        })
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
                size: buffer_size(self.format, self.meta.size)
                    .expect("owned frame format must have a valid buffer size"),
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
    let width = width as usize;
    match format {
        PixelFormat::Gray8 => Ok(width),
        PixelFormat::Bgra8 => width
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("BGRA8 row byte width overflow")),
        PixelFormat::Yuyv422 => width
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("YUYV422 row byte width overflow")),
    }
}

pub fn bytes_per_pixel(format: PixelFormat) -> Result<usize> {
    match format {
        PixelFormat::Gray8 => Ok(1),
        PixelFormat::Bgra8 => Ok(4),
        PixelFormat::Yuyv422 => {
            anyhow::bail!("YUYV422 does not have an integer per-pixel byte width")
        }
    }
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
            frame.buffer.size,
            Size {
                width: 2,
                height: 2,
            }
        );
        assert_eq!(frame.buffer.stride, 4);
        assert_eq!(frame.row(0).unwrap(), &[11, 12]);
        assert_eq!(frame.row(1).unwrap(), &[21, 22]);
        assert_eq!(
            frame.rows().collect::<Vec<_>>(),
            vec![&[11, 12][..], &[21, 22][..]]
        );
    }

    #[test]
    fn frame_roi_uses_byte_width_for_bgra_buffer_size() {
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
            frame.buffer.size,
            Size {
                width: 8,
                height: 2,
            }
        );
        assert_eq!(frame.buffer.stride, 16);
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
