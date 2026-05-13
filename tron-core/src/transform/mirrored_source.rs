use anyhow::Result;
use tron_api::{Frame, FrameSource, OpenedCameraInfo};

pub struct MirroredFrameSource<S> {
    source: S,
    horisontal: bool,
    vertical: bool,
}

impl<S> MirroredFrameSource<S>
where
    S: FrameSource,
{
    pub fn both(source: S) -> Self {
        Self {
            source,
            horisontal: true,
            vertical: true,
        }
    }

    pub fn horizontal(source: S) -> Self {
        Self {
            source,
            horisontal: true,
            vertical: false,
        }
    }

    pub fn vertical(source: S) -> Self {
        Self {
            source,
            horisontal: false,
            vertical: true,
        }
    }
}

#[async_trait::async_trait]
impl<S> FrameSource for MirroredFrameSource<S>
where
    S: FrameSource + Send,
{
    fn info(&self) -> &OpenedCameraInfo {
        &self.source.info()
    }

    async fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        let Some(frame) = self.source.next_frame().await? else {
            return Ok(None);
        };
        Ok(Some(frame.mirrored(self.horisontal, self.vertical)?))
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

        // SAFETY: test asserts the mirror view reuses the original backing
        // storage; logical reads below still go through view APIs.
        assert_eq!(unsafe { frame.buffer.raw() }.as_ptr(), data.as_ptr());
        assert_eq!(frame.view().unwrap()[[0, 0, 0]], 3);
        assert_eq!(frame.view().unwrap()[[0, 2, 0]], 1);
        assert_eq!(frame.view().unwrap()[[1, 2, 0]], 4);
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
        let view = frame.view().unwrap();

        // SAFETY: test asserts the mirror view reuses the original backing
        // storage; logical reads below still go through view APIs.
        assert_eq!(unsafe { frame.buffer.raw() }.as_ptr(), data.as_ptr());
        assert_eq!(view[[0, 0, 0]], 5);
        assert_eq!(view[[0, 1, 0]], 1);
        assert_eq!(view[[1, 0, 2]], 15);
    }
}
