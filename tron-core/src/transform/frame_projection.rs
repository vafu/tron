use anyhow::Result;
use tron_api::{Frame, FrameMeta, OwnedFrame, PixelFormat};

use crate::projection::FrameProjectionMap;

pub(super) fn project_frame(frame: Frame<'_>, map: &FrameProjectionMap) -> Result<OwnedFrame> {
    let output_size = map.output_size;
    let mut data = vec![0_u8; output_size.width as usize * output_size.height as usize];
    let pixels = frame.view()?;
    let channel_at = |x: usize, y: usize, channel: usize| -> Result<u8> {
        pixels.get([y, x, channel]).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "projection source pixel x={} y={} channel={} outside view shape {:?}",
                x,
                y,
                channel,
                pixels.shape()
            )
        })
    };
    for y in 0..output_size.height {
        let dst_row_start = y as usize * output_size.width as usize;
        for x in 0..output_size.width {
            let Some((src_x, src_y)) = map.get(x, y) else {
                continue;
            };
            data[dst_row_start + x as usize] = match frame.format {
                PixelFormat::Gray8 => channel_at(src_x as usize, src_y as usize, 0)?,
                PixelFormat::Bgra8 => {
                    ((channel_at(src_x as usize, src_y as usize, 0)? as u16
                        + channel_at(src_x as usize, src_y as usize, 1)? as u16
                        + channel_at(src_x as usize, src_y as usize, 2)? as u16)
                        / 3) as u8
                }
            };
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
