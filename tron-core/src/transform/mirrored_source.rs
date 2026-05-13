use anyhow::Result;
use tron_api::{Frame, FrameSource, OpenedCameraInfo};

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
        Self { source, info, mode }
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
        Ok(Some(
            frame.mirrored(self.mode.horizontal(), self.mode.vertical())?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use tron_api::frame::row_bytes;
    use tron_api::{FrameMeta, FrameTimestamp, PixelFormat, SensorKind, Size, TimestampSource};

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
    fn mirrors_gray8_horizontally_without_copying() {
        let data = [1, 2, 3, 4, 5, 6];
        let frame = frame(&data, 3, 2, PixelFormat::Gray8)
            .mirrored(true, false)
            .unwrap();

        assert_eq!(frame.buffer.data.as_ptr(), data.as_ptr());
        assert_eq!(frame.row(0).unwrap().byte(0).unwrap(), 3);
        assert_eq!(frame.row(0).unwrap().byte(2).unwrap(), 1);
        assert_eq!(frame.gray8_view().unwrap()[[0, 0]], 3);
        assert_eq!(frame.gray8_view().unwrap()[[1, 2]], 4);
    }

    #[test]
    fn mirrors_bgra8_horizontally_without_copying() {
        let data = [
            1, 2, 3, 4, 5, 6, 7, 8, //
            9, 10, 11, 12, 13, 14, 15, 16,
        ];
        let frame = frame(&data, 2, 2, PixelFormat::Bgra8)
            .mirrored(true, false)
            .unwrap();
        let view = frame.bgra8_view().unwrap();

        assert_eq!(frame.buffer.data.as_ptr(), data.as_ptr());
        assert_eq!(view[[0, 0, 0]], 5);
        assert_eq!(view[[0, 1, 0]], 1);
        assert_eq!(view[[1, 0, 2]], 15);
    }
}
