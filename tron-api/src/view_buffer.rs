use anyhow::Result;
use ndarray::{ArrayView3, Axis, ShapeBuilder};

use crate::frame::PixelFormat;
use crate::{Rect, Size};

#[derive(Clone, Copy, Debug)]
pub struct ViewBuffer<'a> {
    format: PixelFormat,
    size: Size,
    stride: usize,
    data: &'a [u8],
    horizontal: bool,
    vertical: bool,
}

impl<'a> ViewBuffer<'a> {
    pub fn new(format: PixelFormat, size: Size, stride: usize, data: &'a [u8]) -> Result<Self> {
        let cols = row_bytes(format, size.width)?;
        let rows = size.height as usize;
        anyhow::ensure!(
            cols <= stride,
            "view buffer cols {} exceeds stride {}",
            cols,
            stride
        );
        let required_len = required_len(rows, cols, stride)?;
        anyhow::ensure!(
            required_len <= data.len(),
            "view buffer requires {} bytes but has {} bytes",
            required_len,
            data.len()
        );
        Ok(Self {
            format,
            size,
            stride,
            data,
            horizontal: false,
            vertical: false,
        })
    }

    pub fn mirrored(mut self, horizontal: bool, vertical: bool) -> Result<Self> {
        self.horizontal ^= horizontal;
        self.vertical ^= vertical;
        Ok(self)
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    pub fn size(&self) -> Size {
        self.size
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn is_horizontally_mirrored(&self) -> bool {
        self.horizontal
    }

    pub fn is_vertically_mirrored(&self) -> bool {
        self.vertical
    }

    pub fn view(&self) -> Result<ArrayView3<'a, u8>> {
        let channels = self.format.channels();
        let mut view = ArrayView3::from_shape(
            (
                self.size.height as usize,
                self.size.width as usize,
                channels,
            )
                .strides((self.stride, channels, 1)),
            self.data,
        )?;
        if self.vertical {
            view.slice_axis_inplace(Axis(0), ndarray::Slice::new(0, None, -1));
        }
        if self.horizontal {
            view.slice_axis_inplace(Axis(1), ndarray::Slice::new(0, None, -1));
        }
        Ok(view)
    }

    /// Exposes the original backing bytes, ignoring logical mirror state.
    ///
    /// # Safety
    ///
    /// Callers must not interpret this as logical pixel order when the view is
    /// mirrored. This is intended for APIs such as GPU texture upload that need
    /// contiguous physical storage and apply mirror state separately.
    pub unsafe fn raw(&self) -> &'a [u8] {
        self.data
    }

    pub fn roi(self, rect: Rect) -> Result<Self> {
        ensure_roi(self.size, rect)?;
        let x = rect.x as usize;
        let y = rect.y as usize;
        let cols = rect.size.width as usize;
        let rows = rect.size.height as usize;
        let source_x = if self.horizontal {
            self.size.width as usize - x - cols
        } else {
            x
        };
        let source_y = if self.vertical {
            self.size.height as usize - y - rows
        } else {
            y
        };
        let offset = source_y
            .checked_mul(self.stride)
            .and_then(|offset| offset.checked_add(source_x.checked_mul(self.format.channels())?))
            .ok_or_else(|| anyhow::anyhow!("ROI byte offset overflow"))?;
        anyhow::ensure!(
            offset <= self.data.len(),
            "ROI byte offset {} is outside view buffer len {}",
            offset,
            self.data.len()
        );
        Ok(Self {
            size: rect.size,
            data: &self.data[offset..],
            ..self
        })
    }

    pub unsafe fn detach_lifetime<'b>(self) -> ViewBuffer<'b> {
        ViewBuffer {
            format: self.format,
            size: self.size,
            stride: self.stride,
            data: unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.data.len()) },
            horizontal: self.horizontal,
            vertical: self.vertical,
        }
    }
}

#[derive(Debug)]
pub struct ViewBufferMut<'a> {
    pub format: PixelFormat,
    pub size: Size,
    pub stride: usize,
    pub data: &'a mut [u8],
}

impl ViewBufferMut<'_> {
    pub fn as_view_buffer(&self) -> ViewBuffer<'_> {
        ViewBuffer {
            format: self.format,
            size: self.size,
            stride: self.stride,
            data: self.data,
            horizontal: false,
            vertical: false,
        }
    }
}

fn row_bytes(format: PixelFormat, width: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(format.channels())
        .ok_or_else(|| anyhow::anyhow!("{format:?} row byte width overflow"))
}

fn ensure_roi(bounds: Size, rect: Rect) -> Result<()> {
    let x = rect.x as usize;
    let y = rect.y as usize;
    let cols = rect.size.width as usize;
    let rows = rect.size.height as usize;
    let bounds_cols = bounds.width as usize;
    let bounds_rows = bounds.height as usize;
    anyhow::ensure!(
        x <= bounds_cols
            && y <= bounds_rows
            && cols <= bounds_cols.saturating_sub(x)
            && rows <= bounds_rows.saturating_sub(y),
        "ROI x={} y={} cols={} rows={} is outside view buffer cols={} rows={}",
        x,
        y,
        cols,
        rows,
        bounds.width,
        bounds.height
    );
    Ok(())
}

fn required_len(rows: usize, cols: usize, stride: usize) -> Result<usize> {
    if rows == 0 {
        return Ok(0);
    }
    rows.saturating_sub(1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(cols))
        .ok_or_else(|| anyhow::anyhow!("view buffer length overflow"))
}
