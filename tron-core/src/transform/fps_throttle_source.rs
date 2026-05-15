use std::time::Duration;

use anyhow::Result;
use tron_api::{Frame, FrameSource, OpenedCameraInfo};

pub struct FpsThrottledFrameSource<S> {
    source: S,
    min_interval: Duration,
    last_emitted_camera_monotonic_us: Option<i64>,
}

impl<S> FpsThrottledFrameSource<S> {
    pub fn new(source: S, max_fps: f64) -> Result<Self> {
        anyhow::ensure!(
            max_fps.is_finite() && max_fps > 0.0,
            "FPS throttle must be a positive finite value"
        );
        Ok(Self {
            source,
            min_interval: Duration::from_secs_f64(1.0 / max_fps),
            last_emitted_camera_monotonic_us: None,
        })
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
        let frame = self.source.next_frame().await?;
        let Some(frame) = frame else {
            return Ok(None);
        };

        let Some(current_us) = frame.meta.timestamp.camera_monotonic_us else {
            return Ok(None);
        };
        let Some(previous_us) = self.last_emitted_camera_monotonic_us else {
            self.last_emitted_camera_monotonic_us = Some(current_us);
            return Ok(Some(frame));
        };
        if current_us.saturating_sub(previous_us)
            < self.min_interval.as_micros().min(i64::MAX as u128) as i64
        {
            return Ok(None);
        }
        self.last_emitted_camera_monotonic_us = Some(current_us);
        Ok(Some(frame))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant as StdInstant;

    use tron_api::{
        CaptureFormat, FrameMeta, FrameTimestamp, PixelFormat, SensorKind, Size, TimestampSource,
    };

    use super::*;

    struct TestFrameSource {
        info: OpenedCameraInfo,
        data: [u8; 1],
        next_id: u64,
        camera_monotonic_us: Option<i64>,
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
                camera_monotonic_us: None,
            }
        }

        fn with_camera_timestamps() -> Self {
            Self {
                camera_monotonic_us: Some(0),
                ..Self::new()
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
            let camera_monotonic_us = self.camera_monotonic_us;
            if let Some(timestamp) = self.camera_monotonic_us.as_mut() {
                *timestamp = timestamp.saturating_add(33_333);
            }
            Ok(Some(Frame::new(
                FrameMeta {
                    id,
                    sensor: SensorKind::Rgb,
                    size: self.info.size,
                    timestamp: FrameTimestamp {
                        camera_monotonic_us,
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
    async fn drops_frames_without_camera_timestamp() {
        let mut source = FpsThrottledFrameSource::new(TestFrameSource::new(), 20.0).unwrap();
        assert!(source.next_frame().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn throttles_by_camera_timestamp_when_available() {
        let mut source =
            FpsThrottledFrameSource::new(TestFrameSource::with_camera_timestamps(), 8.0).unwrap();
        assert_eq!(source.next_frame().await.unwrap().unwrap().meta.id, 0);
        assert!(source.next_frame().await.unwrap().is_none());
        assert!(source.next_frame().await.unwrap().is_none());
        assert!(source.next_frame().await.unwrap().is_none());
        assert_eq!(source.next_frame().await.unwrap().unwrap().meta.id, 4);
    }

    #[test]
    fn rejects_invalid_fps() {
        assert!(FpsThrottledFrameSource::new(TestFrameSource::new(), 0.0).is_err());
        assert!(FpsThrottledFrameSource::new(TestFrameSource::new(), f64::NAN).is_err());
    }
}
