use anyhow::Result;
use ndarray::{ArrayView2, ArrayView3, Axis, ShapeBuilder};

use crate::Size;

#[derive(Clone, Copy, Debug)]
pub struct ViewBuffer<'a> {
    pub size: Size,
    pub stride: usize,
    pub data: &'a [u8],
    kind: ViewBufferKind<'a>,
}

#[derive(Clone, Copy, Debug)]
enum ViewBufferKind<'a> {
    Linear(LinearViewBuffer<'a>),
    Mirrored(MirroredViewBuffer<'a>),
}

#[derive(Clone, Copy, Debug)]
pub struct LinearViewBuffer<'a> {
    pub size: Size,
    pub stride: usize,
    pub data: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct MirroredViewBuffer<'a> {
    pub source: LinearViewBuffer<'a>,
    pub bytes_per_pixel: usize,
    pub horizontal: bool,
    pub vertical: bool,
}

impl<'a> ViewBuffer<'a> {
    pub fn new(size: Size, stride: usize, data: &'a [u8]) -> Result<Self> {
        let linear = LinearViewBuffer::new(size, stride, data)?;
        Ok(Self {
            size,
            stride,
            data,
            kind: ViewBufferKind::Linear(linear),
        })
    }

    pub fn mirrored(
        self,
        bytes_per_pixel: usize,
        horizontal: bool,
        vertical: bool,
    ) -> Result<Self> {
        anyhow::ensure!(
            bytes_per_pixel > 0,
            "mirrored view buffer bytes_per_pixel must be non-zero"
        );
        let (source, old_horizontal, old_vertical) = match self.kind {
            ViewBufferKind::Linear(source) => (source, false, false),
            ViewBufferKind::Mirrored(mirrored) => {
                anyhow::ensure!(
                    mirrored.bytes_per_pixel == bytes_per_pixel,
                    "cannot compose mirrored views with different pixel widths"
                );
                (mirrored.source, mirrored.horizontal, mirrored.vertical)
            }
        };
        anyhow::ensure!(
            source.size.width as usize % bytes_per_pixel == 0,
            "mirrored view buffer width {} is not divisible by pixel width {}",
            source.size.width,
            bytes_per_pixel
        );
        Ok(Self {
            size: source.size,
            stride: source.stride,
            data: source.data,
            kind: ViewBufferKind::Mirrored(MirroredViewBuffer {
                source,
                bytes_per_pixel,
                horizontal: old_horizontal ^ horizontal,
                vertical: old_vertical ^ vertical,
            }),
        })
    }

    pub fn size(&self) -> Size {
        self.linear().size
    }

    pub fn stride(&self) -> usize {
        self.linear().stride
    }

    pub fn data(&self) -> &'a [u8] {
        self.linear().data
    }

    pub fn as_slice(&self) -> Option<&'a [u8]> {
        self.as_linear().map(|linear| linear.data)
    }

    pub fn as_array2(&self) -> Result<ArrayView2<'a, u8>> {
        match self.kind {
            ViewBufferKind::Linear(linear) => linear.as_array2(),
            ViewBufferKind::Mirrored(mirrored) => mirrored.as_array2(),
        }
    }

    pub fn as_array3(&self, bytes_per_pixel: usize) -> Result<ArrayView3<'a, u8>> {
        match self.kind {
            ViewBufferKind::Linear(linear) => linear.as_array3(bytes_per_pixel),
            ViewBufferKind::Mirrored(mirrored) => mirrored.as_array3(bytes_per_pixel),
        }
    }

    pub fn is_horizontally_mirrored(&self) -> bool {
        match self.kind {
            ViewBufferKind::Linear(_) => false,
            ViewBufferKind::Mirrored(mirrored) => mirrored.horizontal,
        }
    }

    pub fn is_vertically_mirrored(&self) -> bool {
        match self.kind {
            ViewBufferKind::Linear(_) => false,
            ViewBufferKind::Mirrored(mirrored) => mirrored.vertical,
        }
    }

    pub fn is_mirrored(&self) -> bool {
        self.is_horizontally_mirrored() || self.is_vertically_mirrored()
    }

    pub fn as_linear(&self) -> Option<LinearViewBuffer<'a>> {
        match self.kind {
            ViewBufferKind::Linear(linear) => Some(linear),
            ViewBufferKind::Mirrored(mirrored) if !mirrored.horizontal && !mirrored.vertical => {
                Some(mirrored.source)
            }
            ViewBufferKind::Mirrored(_) => None,
        }
    }

    pub fn row(&self, y: usize) -> Result<ViewRow<'a>> {
        match self.kind {
            ViewBufferKind::Linear(linear) => linear.row(y).map(ViewRow::Linear),
            ViewBufferKind::Mirrored(mirrored) => mirrored.row(y),
        }
    }

    pub fn rows(&self) -> ViewRows<'a, '_> {
        ViewRows { buffer: self, y: 0 }
    }

    pub fn roi(self, x: usize, y: usize, size: Size) -> Result<Self> {
        match self.kind {
            ViewBufferKind::Linear(linear) => {
                let linear = linear.roi(x, y, size)?;
                Ok(Self {
                    size: linear.size,
                    stride: linear.stride,
                    data: linear.data,
                    kind: ViewBufferKind::Linear(linear),
                })
            }
            ViewBufferKind::Mirrored(mirrored) => {
                let mirrored = mirrored.roi(x, y, size)?;
                Ok(Self {
                    size: mirrored.source.size,
                    stride: mirrored.source.stride,
                    data: mirrored.source.data,
                    kind: ViewBufferKind::Mirrored(mirrored),
                })
            }
        }
    }

    pub unsafe fn detach_lifetime<'b>(self) -> ViewBuffer<'b> {
        match self.kind {
            ViewBufferKind::Linear(linear) => ViewBuffer {
                size: self.size,
                stride: self.stride,
                data: unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.data.len()) },
                kind: ViewBufferKind::Linear(unsafe { linear.detach_lifetime() }),
            },
            ViewBufferKind::Mirrored(mirrored) => ViewBuffer {
                size: self.size,
                stride: self.stride,
                data: unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.data.len()) },
                kind: ViewBufferKind::Mirrored(unsafe { mirrored.detach_lifetime() }),
            },
        }
    }

    fn linear(&self) -> LinearViewBuffer<'a> {
        match self.kind {
            ViewBufferKind::Linear(linear) => linear,
            ViewBufferKind::Mirrored(mirrored) => mirrored.source,
        }
    }
}

impl<'a> LinearViewBuffer<'a> {
    pub fn new(size: Size, stride: usize, data: &'a [u8]) -> Result<Self> {
        let cols = size.width as usize;
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
        Ok(Self { size, stride, data })
    }

    pub fn row(&self, y: usize) -> Result<&'a [u8]> {
        anyhow::ensure!(
            y < self.size.height as usize,
            "row {} is outside view buffer height {}",
            y,
            self.size.height
        );
        let start = y
            .checked_mul(self.stride)
            .ok_or_else(|| anyhow::anyhow!("row byte offset overflow"))?;
        let end = start
            .checked_add(self.size.width as usize)
            .ok_or_else(|| anyhow::anyhow!("row byte length overflow"))?;
        anyhow::ensure!(
            end <= self.data.len(),
            "row requires {} bytes but view buffer has {} bytes",
            end,
            self.data.len()
        );
        Ok(&self.data[start..end])
    }

    pub fn as_array2(&self) -> Result<ArrayView2<'a, u8>> {
        ArrayView2::from_shape(
            (self.size.height as usize, self.size.width as usize).strides((self.stride, 1)),
            self.data,
        )
        .map_err(Into::into)
    }

    pub fn as_array3(&self, bytes_per_pixel: usize) -> Result<ArrayView3<'a, u8>> {
        anyhow::ensure!(bytes_per_pixel > 0, "array pixel width must be non-zero");
        anyhow::ensure!(
            self.size.width as usize % bytes_per_pixel == 0,
            "view buffer width {} is not divisible by pixel width {}",
            self.size.width,
            bytes_per_pixel
        );
        ArrayView3::from_shape(
            (
                self.size.height as usize,
                self.size.width as usize / bytes_per_pixel,
                bytes_per_pixel,
            )
                .strides((self.stride, bytes_per_pixel, 1)),
            self.data,
        )
        .map_err(Into::into)
    }

    pub fn roi(self, x: usize, y: usize, size: Size) -> Result<Self> {
        ensure_roi(self.size, x, y, size)?;
        let offset = y
            .checked_mul(self.stride)
            .and_then(|offset| offset.checked_add(x))
            .ok_or_else(|| anyhow::anyhow!("ROI byte offset overflow"))?;
        Self::new(size, self.stride, &self.data[offset..])
    }

    pub unsafe fn detach_lifetime<'b>(self) -> LinearViewBuffer<'b> {
        LinearViewBuffer {
            size: self.size,
            stride: self.stride,
            data: unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.data.len()) },
        }
    }
}

impl<'a> MirroredViewBuffer<'a> {
    pub fn row(&self, y: usize) -> Result<ViewRow<'a>> {
        let size = self.source.size;
        anyhow::ensure!(
            y < size.height as usize,
            "row {} is outside view buffer height {}",
            y,
            size.height
        );
        let source_y = if self.vertical {
            size.height as usize - 1 - y
        } else {
            y
        };
        let row = self.source.row(source_y)?;
        if self.horizontal {
            Ok(ViewRow::Mirrored {
                data: row,
                bytes_per_pixel: self.bytes_per_pixel,
            })
        } else {
            Ok(ViewRow::Linear(row))
        }
    }

    pub fn as_array2(&self) -> Result<ArrayView2<'a, u8>> {
        anyhow::ensure!(
            self.bytes_per_pixel == 1 || !self.horizontal,
            "horizontal mirror with pixel width {} cannot be represented as a 2D byte ndarray view",
            self.bytes_per_pixel
        );
        let mut view = self.source.as_array2()?;
        if self.vertical {
            view.slice_axis_inplace(Axis(0), ndarray::Slice::new(0, None, -1));
        }
        if self.horizontal {
            view.slice_axis_inplace(Axis(1), ndarray::Slice::new(0, None, -1));
        }
        Ok(view)
    }

    pub fn as_array3(&self, bytes_per_pixel: usize) -> Result<ArrayView3<'a, u8>> {
        anyhow::ensure!(
            bytes_per_pixel == self.bytes_per_pixel,
            "requested pixel width {} does not match mirrored buffer pixel width {}",
            bytes_per_pixel,
            self.bytes_per_pixel
        );
        let mut view = self.source.as_array3(bytes_per_pixel)?;
        if self.vertical {
            view.slice_axis_inplace(Axis(0), ndarray::Slice::new(0, None, -1));
        }
        if self.horizontal {
            view.slice_axis_inplace(Axis(1), ndarray::Slice::new(0, None, -1));
        }
        Ok(view)
    }

    pub fn roi(self, x: usize, y: usize, size: Size) -> Result<Self> {
        ensure_roi(self.source.size, x, y, size)?;
        let cols = size.width as usize;
        let rows = size.height as usize;
        let source_x = if self.horizontal {
            self.source.size.width as usize - x - cols
        } else {
            x
        };
        let source_y = if self.vertical {
            self.source.size.height as usize - y - rows
        } else {
            y
        };
        Ok(Self {
            source: self.source.roi(source_x, source_y, size)?,
            ..self
        })
    }

    pub unsafe fn detach_lifetime<'b>(self) -> MirroredViewBuffer<'b> {
        MirroredViewBuffer {
            source: unsafe { self.source.detach_lifetime() },
            bytes_per_pixel: self.bytes_per_pixel,
            horizontal: self.horizontal,
            vertical: self.vertical,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ViewRow<'a> {
    Linear(&'a [u8]),
    Mirrored {
        data: &'a [u8],
        bytes_per_pixel: usize,
    },
}

impl<'a> ViewRow<'a> {
    pub fn len(&self) -> usize {
        match self {
            Self::Linear(data) | Self::Mirrored { data, .. } => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> Option<&'a [u8]> {
        match self {
            Self::Linear(data) => Some(data),
            Self::Mirrored { .. } => None,
        }
    }

    pub fn byte(&self, index: usize) -> Result<u8> {
        match self {
            Self::Linear(data) => data.get(index).copied().ok_or_else(|| {
                anyhow::anyhow!("byte index {} is outside row len {}", index, data.len())
            }),
            Self::Mirrored {
                data,
                bytes_per_pixel,
            } => {
                anyhow::ensure!(
                    index < data.len(),
                    "byte index {} is outside row len {}",
                    index,
                    data.len()
                );
                let pixel = index / *bytes_per_pixel;
                let channel = index % *bytes_per_pixel;
                let width = data.len() / *bytes_per_pixel;
                let source = (width - 1 - pixel) * *bytes_per_pixel + channel;
                Ok(data[source])
            }
        }
    }

    pub fn pixel(&self, x: usize, bytes_per_pixel: usize) -> Result<&'a [u8]> {
        match self {
            Self::Linear(data) => {
                let start = x
                    .checked_mul(bytes_per_pixel)
                    .ok_or_else(|| anyhow::anyhow!("pixel byte offset overflow"))?;
                let end = start
                    .checked_add(bytes_per_pixel)
                    .ok_or_else(|| anyhow::anyhow!("pixel byte length overflow"))?;
                anyhow::ensure!(
                    end <= data.len(),
                    "pixel {} requires {} bytes but row has {} bytes",
                    x,
                    end,
                    data.len()
                );
                Ok(&data[start..end])
            }
            Self::Mirrored {
                data,
                bytes_per_pixel: mirror_bytes_per_pixel,
            } => {
                anyhow::ensure!(
                    bytes_per_pixel == *mirror_bytes_per_pixel,
                    "requested pixel width {} does not match mirrored row pixel width {}",
                    bytes_per_pixel,
                    mirror_bytes_per_pixel
                );
                let width = data.len() / bytes_per_pixel;
                anyhow::ensure!(x < width, "pixel {} is outside row width {}", x, width);
                let source_x = width - 1 - x;
                let start = source_x * bytes_per_pixel;
                Ok(&data[start..start + bytes_per_pixel])
            }
        }
    }

    pub fn copy_to(&self, output: &mut [u8]) -> Result<()> {
        anyhow::ensure!(
            output.len() == self.len(),
            "row copy length mismatch: output={} row={}",
            output.len(),
            self.len()
        );
        match self {
            Self::Linear(data) => output.copy_from_slice(data),
            Self::Mirrored {
                data,
                bytes_per_pixel,
            } => {
                let width = data.len() / *bytes_per_pixel;
                for x in 0..width {
                    let source_x = width - 1 - x;
                    let src = source_x * *bytes_per_pixel;
                    let dst = x * *bytes_per_pixel;
                    output[dst..dst + *bytes_per_pixel]
                        .copy_from_slice(&data[src..src + *bytes_per_pixel]);
                }
            }
        }
        Ok(())
    }
}

pub struct ViewRows<'a, 'buffer> {
    buffer: &'buffer ViewBuffer<'a>,
    y: usize,
}

impl<'a> Iterator for ViewRows<'a, '_> {
    type Item = ViewRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.buffer.size().height as usize {
            return None;
        }
        let row = self.buffer.row(self.y).ok()?;
        self.y += 1;
        Some(row)
    }
}

#[derive(Debug)]
pub struct ViewBufferMut<'a> {
    pub size: Size,
    pub stride: usize,
    pub data: &'a mut [u8],
}

impl ViewBufferMut<'_> {
    pub fn as_view_buffer(&self) -> ViewBuffer<'_> {
        ViewBuffer {
            size: self.size,
            stride: self.stride,
            data: self.data,
            kind: ViewBufferKind::Linear(LinearViewBuffer {
                size: self.size,
                stride: self.stride,
                data: self.data,
            }),
        }
    }
}

fn ensure_roi(bounds: Size, x: usize, y: usize, size: Size) -> Result<()> {
    let cols = size.width as usize;
    let rows = size.height as usize;
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
