use std::time::Duration;

use anyhow::Result;
use tokio::time::Instant;
use tron_api::{Frame, FrameSource, OpenedCameraInfo};

pub struct FpsThrottledFrameSource<S> {
    source: S,
    min_interval: Option<Duration>,
    last_frame_at: Option<Instant>,
}

impl<S> FpsThrottledFrameSource<S> {
    pub fn new(source: S, max_fps: Option<f64>) -> Result<Self> {
        let min_interval = max_fps
            .map(|fps| {
                anyhow::ensure!(
                    fps.is_finite() && fps > 0.0,
                    "FPS throttle must be a positive finite value"
                );
                Ok(Duration::from_secs_f64(1.0 / fps))
            })
            .transpose()?;
        Ok(Self {
            source,
            min_interval,
            last_frame_at: None,
        })
    }

    pub fn unlimited(source: S) -> Self {
        Self {
            source,
            min_interval: None,
            last_frame_at: None,
        }
    }

    pub fn into_inner(self) -> S {
        self.source
    }
}

#[async_trait::async_trait]
impl<S> FrameSource for FpsThrottledFrameSource<S>
where
    S: FrameSource + Send,
{
    fn info(&self) -> &OpenedCameraInfo {
        self.source.info()
    }

    async fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        if let (Some(min_interval), Some(last_frame_at)) = (self.min_interval, self.last_frame_at) {
            let next_allowed = last_frame_at + min_interval;
            if Instant::now() < next_allowed {
                let _ = self.source.next_frame().await?;
                return Ok(None);
            }
        }

        let frame = self.source.next_frame().await?;
        if frame.is_some() {
            self.last_frame_at = Some(Instant::now());
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant as StdInstant};

    use tron_api::{
        CaptureFormat, FrameMeta, FrameTimestamp, PixelFormat, SensorKind, Size, TimestampSource,
    };

    use super::*;

    struct TestFrameSource {
        info: OpenedCameraInfo,
        data: [u8; 1],
        next_id: u64,
    }

    impl TestFrameSource {
        fn new() -> Self {
            Self {
                info: OpenedCameraInfo {
                    id: "test".to_owned(),
                    sensor: SensorKind::Rgb,
                    format: CaptureFormat::Gray8,
                    size: Size {
                        width: 1,
                        height: 1,
                    },
                },
                data: [0],
                next_id: 0,
            }
        }
    }

    #[async_trait::async_trait]
    impl FrameSource for TestFrameSource {
        fn info(&self) -> &OpenedCameraInfo {
            &self.info
        }

        async fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
            let id = self.next_id;
            self.next_id += 1;
            Ok(Some(Frame::new(
                FrameMeta {
                    id,
                    sensor: SensorKind::Rgb,
                    size: self.info.size,
                    timestamp: FrameTimestamp {
                        camera_monotonic_us: None,
                        source: TimestampSource::Unknown,
                        received_at: StdInstant::now(),
                    },
                    sequence: None,
                },
                PixelFormat::Gray8,
                1,
                &self.data,
            )?))
        }
    }

    #[tokio::test]
    async fn throttles_frame_interval() {
        let mut source = FpsThrottledFrameSource::new(TestFrameSource::new(), Some(20.0)).unwrap();
        assert!(source.next_frame().await.unwrap().is_some());
        let start = StdInstant::now();
        assert!(source.next_frame().await.unwrap().is_none());
        assert!(start.elapsed() < Duration::from_millis(45));
        tokio::time::sleep(Duration::from_millis(50)).await;
        let frame = source.next_frame().await.unwrap().unwrap();
        assert!(frame.meta.id > 1);
    }

    #[test]
    fn rejects_invalid_fps() {
        assert!(FpsThrottledFrameSource::new(TestFrameSource::new(), Some(0.0)).is_err());
        assert!(FpsThrottledFrameSource::new(TestFrameSource::new(), Some(f64::NAN)).is_err());
    }
}
