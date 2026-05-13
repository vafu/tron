use anyhow::Result;
use tron_api::frame::row_bytes;
use tron_api::{Frame, FrameSource, OpenedCameraInfo, OwnedFrame, PixelFormat};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirrorMode {
    Horizontal,
    Vertical,
    HorizontalAndVertical,
}

impl MirrorMode {
    fn horizontal(self) -> bool {
        matches!(self, Self::Horizontal | Self::HorizontalAndVertical)
    }

    fn vertical(self) -> bool {
        matches!(self, Self::Vertical | Self::HorizontalAndVertical)
    }
}

pub struct MirroredFrameSource<S> {
    source: S,
    info: OpenedCameraInfo,
    mode: MirrorMode,
    current: Option<OwnedFrame>,
}

impl<S> MirroredFrameSource<S>
where
    S: FrameSource,
{
    pub fn new(source: S) -> Self {
        Self::with_mode(source, MirrorMode::Horizontal)
    }

    pub fn with_mode(source: S, mode: MirrorMode) -> Self {
        let info = source.info().clone();
        Self {
            source,
            info,
            mode,
            current: None,
        }
    }
}

#[async_trait::async_trait]
impl<S> FrameSource for MirroredFrameSource<S>
where
    S: FrameSource + Send,
{
    fn info(&self) -> &OpenedCameraInfo {
        &self.info
    }

    async fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        let Some(frame) = self.source.next_frame().await? else {
            return Ok(None);
        };
        mirror_frame_into(frame, self.mode, &mut self.current)?;
        Ok(self.current.as_ref().map(OwnedFrame::as_frame))
    }
}

fn mirror_frame_into(
    frame: Frame<'_>,
    mode: MirrorMode,
    output: &mut Option<OwnedFrame>,
) -> Result<()> {
    let row_len = row_bytes(frame.format, frame.meta.size.width)?;
    let len = row_len
        .checked_mul(frame.meta.size.height as usize)
        .ok_or_else(|| anyhow::anyhow!("mirrored frame size overflow"))?;
    ensure_output(output, frame, row_len, len);
    let output = output.as_mut().expect("mirrored output initialized");
    output.meta = frame.meta;

    match frame.format {
        PixelFormat::Gray8 => mirror_fixed_width_pixels(frame, mode, 1, &mut output.data),
        PixelFormat::Bgra8 => mirror_fixed_width_pixels(frame, mode, 4, &mut output.data),
        PixelFormat::Yuyv422 => mirror_yuyv422(frame, mode, &mut output.data),
    }
}

fn ensure_output(output: &mut Option<OwnedFrame>, frame: Frame<'_>, stride: usize, len: usize) {
    let needs_new = output
        .as_ref()
        .map(|output| {
            output.meta.size != frame.meta.size
                || output.format != frame.format
                || output.stride != stride
                || output.data.len() != len
        })
        .unwrap_or(true);
    if needs_new {
        *output = Some(OwnedFrame {
            meta: frame.meta,
            format: frame.format,
            stride,
            data: vec![0; len],
        });
    }
}

fn mirror_fixed_width_pixels(
    frame: Frame<'_>,
    mode: MirrorMode,
    bytes_per_pixel: usize,
    output: &mut [u8],
) -> Result<()> {
    let width = frame.meta.size.width as usize;
    let height = frame.meta.size.height as usize;
    let row_len = width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| anyhow::anyhow!("mirrored row size overflow"))?;
    for dst_y in 0..height {
        let src_y = source_y(dst_y, height, mode);
        let src = frame.buffer.row(src_y)?;
        let dst = &mut output[dst_y * row_len..(dst_y + 1) * row_len];
        if mode.horizontal() {
            for dst_x in 0..width {
                let src_x = width - 1 - dst_x;
                let src_offset = src_x * bytes_per_pixel;
                let dst_offset = dst_x * bytes_per_pixel;
                dst[dst_offset..dst_offset + bytes_per_pixel]
                    .copy_from_slice(&src[src_offset..src_offset + bytes_per_pixel]);
            }
        } else {
            dst.copy_from_slice(src);
        }
    }
    Ok(())
}

fn mirror_yuyv422(frame: Frame<'_>, mode: MirrorMode, output: &mut [u8]) -> Result<()> {
    let width = frame.meta.size.width as usize;
    let height = frame.meta.size.height as usize;
    anyhow::ensure!(
        width % 2 == 0,
        "YUYV422 mirrored source requires an even frame width"
    );
    let row_len = row_bytes(PixelFormat::Yuyv422, frame.meta.size.width)?;
    for dst_y in 0..height {
        let src_y = source_y(dst_y, height, mode);
        let src = frame.buffer.row(src_y)?;
        let dst = &mut output[dst_y * row_len..(dst_y + 1) * row_len];
        if mode.horizontal() {
            for dst_pair in 0..width / 2 {
                let src_pair = width / 2 - 1 - dst_pair;
                let src_offset = src_pair * 4;
                let dst_offset = dst_pair * 4;
                dst[dst_offset] = src[src_offset + 2];
                dst[dst_offset + 1] = src[src_offset + 1];
                dst[dst_offset + 2] = src[src_offset];
                dst[dst_offset + 3] = src[src_offset + 3];
            }
        } else {
            dst.copy_from_slice(src);
        }
    }
    Ok(())
}

fn source_y(dst_y: usize, height: usize, mode: MirrorMode) -> usize {
    if mode.vertical() {
        height - 1 - dst_y
    } else {
        dst_y
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use tron_api::{FrameMeta, FrameTimestamp, SensorKind, Size, TimestampSource};

    use super::*;

    fn frame(data: &[u8], width: u32, height: u32, format: PixelFormat) -> Frame<'_> {
        let stride = row_bytes(format, width).unwrap();
        Frame::new(
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
            },
            format,
            stride,
            data,
        )
        .unwrap()
    }

    #[test]
    fn mirrors_gray8_horizontally() {
        let data = [1, 2, 3, 4, 5, 6];
        let mut output = None;
        mirror_frame_into(
            frame(&data, 3, 2, PixelFormat::Gray8),
            MirrorMode::Horizontal,
            &mut output,
        )
        .unwrap();

        assert_eq!(output.unwrap().data, vec![3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn mirrors_bgra8_vertically() {
        let data = [
            1, 2, 3, 4, 5, 6, 7, 8, //
            9, 10, 11, 12, 13, 14, 15, 16,
        ];
        let mut output = None;
        mirror_frame_into(
            frame(&data, 2, 2, PixelFormat::Bgra8),
            MirrorMode::Vertical,
            &mut output,
        )
        .unwrap();

        assert_eq!(
            output.unwrap().data,
            vec![9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8]
        );
    }

    #[test]
    fn mirrors_yuyv422_horizontally() {
        let data = [
            10, 20, 30, 40, 50, 60, 70, 80, //
            11, 21, 31, 41, 51, 61, 71, 81,
        ];
        let mut output = None;
        mirror_frame_into(
            frame(&data, 4, 2, PixelFormat::Yuyv422),
            MirrorMode::Horizontal,
            &mut output,
        )
        .unwrap();

        assert_eq!(
            output.unwrap().data,
            vec![
                70, 60, 50, 80, 30, 20, 10, 40, 71, 61, 51, 81, 31, 21, 11, 41
            ]
        );
    }
}
