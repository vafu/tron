use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
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
            loop {
                match pollster::block_on(source.next_frame()) {
                    Ok(Some(frame)) => {
                        let owned = Arc::new(own_frame(frame));
                        if let Ok(mut latest) = thread_latest.lock() {
                            *latest = Some(owned);
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

fn own_frame(frame: Frame<'_>) -> OwnedFrame {
    OwnedFrame {
        meta: frame.meta,
        format: frame.format,
        stride: frame.buffer.stride,
        data: frame.buffer.data.to_vec(),
    }
}
