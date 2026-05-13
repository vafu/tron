use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{Context, Result};
use tron_api::{Frame, FrameSource, OpenedCameraInfo, OwnedFrame};

#[derive(Clone)]
pub struct LatestFrameSource {
    info: OpenedCameraInfo,
    latest: Arc<Mutex<Option<Arc<OwnedFrame>>>>,
    current: Option<Arc<OwnedFrame>>,
}

impl LatestFrameSource {
    pub fn spawn(name: &'static str, mut source: Box<dyn FrameSource + Send + 'static>) -> Self {
        let info = source.info().clone();
        let latest = Arc::new(Mutex::new(None));
        let thread_latest = latest.clone();

        thread::spawn(move || {
            let mut pool = Vec::new();
            loop {
                match pollster::block_on(source.next_frame()) {
                    Ok(Some(frame)) => {
                        let mut owned = pool.pop().unwrap_or_else(|| empty_frame(frame));
                        if let Err(err) = copy_frame_into(frame, &mut owned) {
                            eprintln!("capture {name}: stopped: {err:#}");
                            return;
                        }
                        let owned = Arc::new(owned);
                        if let Ok(mut latest) = thread_latest.lock() {
                            let previous = latest.replace(owned);
                            if let Some(previous) = previous
                                && let Ok(previous) = Arc::try_unwrap(previous)
                            {
                                pool.push(previous);
                            }
                        }
                    }
                    Ok(None) => thread::yield_now(),
                    Err(err) => {
                        eprintln!("capture {name}: stopped: {err:#}");
                        return;
                    }
                }
            }
        });

        Self {
            info,
            latest,
            current: None,
        }
    }

    pub fn next_frame(&self) -> Result<Option<Arc<OwnedFrame>>> {
        Ok(self.latest.lock().ok().and_then(|latest| latest.clone()))
    }
}

#[async_trait::async_trait]
impl FrameSource for LatestFrameSource {
    fn info(&self) -> &OpenedCameraInfo {
        &self.info
    }

    async fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        self.current = self.latest.lock().ok().and_then(|latest| latest.clone());
        Ok(self.current.as_ref().map(|frame| frame.as_frame()))
    }
}

fn empty_frame(frame: Frame<'_>) -> OwnedFrame {
    let row_len = frame.meta.size.width as usize * frame.format.channels();
    OwnedFrame {
        meta: frame.meta,
        format: frame.format,
        stride: row_len,
        data: Vec::new(),
    }
}

fn copy_frame_into(frame: Frame<'_>, output: &mut OwnedFrame) -> Result<()> {
    let row_len = frame.meta.size.width as usize * frame.format.channels();
    let len = row_len
        .checked_mul(frame.meta.size.height as usize)
        .context("latest frame buffer size overflow")?;
    output.meta = frame.meta;
    output.format = frame.format;
    output.stride = row_len;
    output.data.resize(len, 0);

    if frame.buffer.stride() == row_len
        && !frame.buffer.is_horizontally_mirrored()
        && !frame.buffer.is_vertically_mirrored()
    {
        // SAFETY: this fast path only copies physical storage when logical and
        // physical pixel order match and rows are tightly packed.
        let raw = unsafe { frame.buffer.raw() };
        anyhow::ensure!(
            raw.len() >= len,
            "latest frame raw storage len {} shorter than expected {}",
            raw.len(),
            len
        );
        output.data.copy_from_slice(&raw[..len]);
        return Ok(());
    }

    let view = frame.view().context("view latest frame")?;
    for y in 0..frame.meta.size.height as usize {
        for x in 0..frame.meta.size.width as usize {
            for channel in 0..frame.format.channels() {
                output.data[y * row_len + x * frame.format.channels() + channel] =
                    view[[y, x, channel]];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use tron_api::{
        FrameMeta, FrameTimestamp, PixelFormat, SensorKind, Size, TimestampSource, frame::row_bytes,
    };

    use super::*;

    fn meta(width: u32, height: u32) -> FrameMeta {
        FrameMeta {
            id: 7,
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
    fn copy_frame_into_preserves_logical_mirror_order() {
        let data = [1, 2, 3, 4, 5, 6];
        let frame = Frame::new(
            meta(3, 2),
            PixelFormat::Gray8,
            row_bytes(PixelFormat::Gray8, 3).unwrap(),
            &data,
        )
        .unwrap()
        .mirrored(true, false)
        .unwrap();
        let mut owned = empty_frame(frame);

        copy_frame_into(frame, &mut owned).unwrap();

        assert_eq!(owned.data, [3, 2, 1, 6, 5, 4]);
    }
}
