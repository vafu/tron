use anyhow::Result;
use tron_api::{Frame, FrameMeta, OwnedFrame, PixelFormat, ViewRow};

use crate::projection::FrameProjectionMap;

pub(super) fn project_frame(frame: Frame<'_>, map: &FrameProjectionMap) -> Result<OwnedFrame> {
    let output_size = map.output_size;
    let mut data = vec![0_u8; output_size.width as usize * output_size.height as usize];
    for y in 0..output_size.height {
        let dst_row_start = y as usize * output_size.width as usize;
        for x in 0..output_size.width {
            let Some((src_x, src_y)) = map.get(x, y) else {
                continue;
            };
            let row = frame.row(src_y)?;
            data[dst_row_start + x as usize] = gray_at(row, frame.format, src_x as usize)?;
        }
    }

    Ok(OwnedFrame {
        meta: FrameMeta {
            size: output_size,
            ..frame.meta
        },
        format: PixelFormat::Gray8,
        stride: output_size.width as usize,
        data,
    })
}

fn gray_at(row: ViewRow<'_>, format: PixelFormat, x: usize) -> Result<u8> {
    match format {
        PixelFormat::Gray8 => Ok(row.byte(x)?),
        PixelFormat::Bgra8 => {
            let offset = x * 4;
            Ok(((row.byte(offset)? as u16
                + row.byte(offset + 1)? as u16
                + row.byte(offset + 2)? as u16)
                / 3) as u8)
        }
        PixelFormat::Yuyv422 => anyhow::bail!("projection transform does not support YUYV422"),
    }
}
