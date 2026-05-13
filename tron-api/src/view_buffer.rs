use anyhow::Result;

use crate::Size;

#[derive(Clone, Copy, Debug)]
pub struct ViewBuffer<'a> {
    pub size: Size,
    pub stride: usize,
    pub data: &'a [u8],
}

impl<'a> ViewBuffer<'a> {
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

    pub fn rows(&self) -> ViewRows<'a, '_> {
        ViewRows { buffer: self, y: 0 }
    }

    pub fn roi(self, x: usize, y: usize, size: Size) -> Result<Self> {
        let cols = size.width as usize;
        let rows = size.height as usize;
        let self_cols = self.size.width as usize;
        let self_rows = self.size.height as usize;
        anyhow::ensure!(
            x <= self_cols
                && y <= self_rows
                && cols <= self_cols.saturating_sub(x)
                && rows <= self_rows.saturating_sub(y),
            "ROI x={} y={} cols={} rows={} is outside view buffer cols={} rows={}",
            x,
            y,
            cols,
            rows,
            self.size.width,
            self.size.height
        );
        let offset = y
            .checked_mul(self.stride)
            .and_then(|offset| offset.checked_add(x))
            .ok_or_else(|| anyhow::anyhow!("ROI byte offset overflow"))?;
        let data = &self.data[offset..];
        Self::new(size, self.stride, data)
    }
}

pub struct ViewRows<'a, 'buffer> {
    buffer: &'buffer ViewBuffer<'a>,
    y: usize,
}

impl<'a> Iterator for ViewRows<'a, '_> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.buffer.size.height as usize {
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
        }
    }
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
