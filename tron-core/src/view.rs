use anyhow::Result;
use tron_api::{Frame, FrameMeta, PixelFormat, Rect, View};

pub trait IntoView<'a> {
    fn view(self) -> View<'a>;
}

impl<'a> IntoView<'a> for Frame<'a> {
    fn view(self) -> View<'a> {
        View {
            meta: self.meta,
            format: self.format,
            size: self.meta.size,
            stride: self.stride,
            data: self.data,
        }
    }
}

pub trait ViewExt<'a> {
    fn roi(self, rect: Rect) -> Result<View<'a>>;
    fn row(&self, y: u32) -> Result<&'a [u8]>;
    fn rows(&self) -> Rows<'a, '_>;
}

impl<'a> ViewExt<'a> for View<'a> {
    fn roi(self, rect: Rect) -> Result<View<'a>> {
        anyhow::ensure!(
            rect.x <= self.size.width
                && rect.y <= self.size.height
                && rect.size.width <= self.size.width.saturating_sub(rect.x)
                && rect.size.height <= self.size.height.saturating_sub(rect.y),
            "ROI {:?} is outside view size {:?}",
            rect,
            self.size
        );
        let bytes_per_pixel = bytes_per_pixel(self.format)?;
        let x = rect.x as usize;
        let y = rect.y as usize;
        let width = rect.size.width as usize;
        let height = rect.size.height as usize;
        let offset = y
            .checked_mul(self.stride)
            .and_then(|offset| offset.checked_add(x.checked_mul(bytes_per_pixel)?))
            .ok_or_else(|| anyhow::anyhow!("ROI byte offset overflow"))?;
        let last_row_offset = height
            .saturating_sub(1)
            .checked_mul(self.stride)
            .ok_or_else(|| anyhow::anyhow!("ROI row offset overflow"))?;
        let row_len = width
            .checked_mul(bytes_per_pixel)
            .ok_or_else(|| anyhow::anyhow!("ROI row length overflow"))?;
        let required_len = offset
            .checked_add(last_row_offset)
            .and_then(|offset| offset.checked_add(row_len))
            .ok_or_else(|| anyhow::anyhow!("ROI length overflow"))?;
        anyhow::ensure!(
            required_len <= self.data.len(),
            "ROI requires {} bytes but view has {} bytes",
            required_len,
            self.data.len()
        );

        Ok(View {
            meta: FrameMeta {
                size: rect.size,
                ..self.meta
            },
            format: self.format,
            size: rect.size,
            stride: self.stride,
            data: &self.data[offset..],
        })
    }

    fn row(&self, y: u32) -> Result<&'a [u8]> {
        anyhow::ensure!(
            y < self.size.height,
            "row {} is outside view height {}",
            y,
            self.size.height
        );
        let bytes_per_pixel = bytes_per_pixel(self.format)?;
        let row_start = y as usize * self.stride;
        let row_end = row_start + self.size.width as usize * bytes_per_pixel;
        anyhow::ensure!(
            row_end <= self.data.len(),
            "row requires {} bytes but view has {} bytes",
            row_end,
            self.data.len()
        );
        Ok(&self.data[row_start..row_end])
    }

    fn rows(&self) -> Rows<'a, '_> {
        Rows { view: self, y: 0 }
    }
}

pub struct Rows<'a, 'view> {
    view: &'view View<'a>,
    y: u32,
}

impl<'a> Iterator for Rows<'a, '_> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.view.size.height {
            return None;
        }
        let row = self.view.row(self.y).ok()?;
        self.y += 1;
        Some(row)
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
    use std::time::Instant;

    use super::*;
    use tron_api::{FrameMeta, FrameTimestamp, SensorKind, Size, TimestampSource};

    fn frame(data: &[u8]) -> Frame<'_> {
        Frame {
            meta: FrameMeta {
                id: 1,
                sensor: SensorKind::Ir,
                size: Size {
                    width: 4,
                    height: 3,
                },
                timestamp: FrameTimestamp {
                    camera_monotonic_us: None,
                    source: TimestampSource::Unknown,
                    received_at: Instant::now(),
                },
                sequence: None,
            },
            format: PixelFormat::Gray8,
            stride: 4,
            data,
        }
    }

    #[test]
    fn roi_exposes_strided_rows_without_copying() {
        let data = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23];
        let view = frame(&data)
            .view()
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
            view.size,
            Size {
                width: 2,
                height: 2
            }
        );
        assert_eq!(view.stride, 4);
        assert_eq!(view.row(0).unwrap(), &[11, 12]);
        assert_eq!(view.row(1).unwrap(), &[21, 22]);
        assert_eq!(
            view.rows().collect::<Vec<_>>(),
            vec![&[11, 12][..], &[21, 22][..]]
        );
    }

    #[test]
    fn roi_rejects_out_of_bounds_rect() {
        let data = [0; 12];
        let err = frame(&data)
            .view()
            .roi(Rect {
                x: 3,
                y: 1,
                size: Size {
                    width: 2,
                    height: 1,
                },
            })
            .unwrap_err();
        assert!(err.to_string().contains("outside view size"));
    }
}
