use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
use tron_api::{Frame, OwnedFrame};
use tron_core::pipeline::FrameStream;

#[derive(Clone)]
pub struct LatestFrameSource {
    latest: Arc<Mutex<Option<Arc<OwnedFrame>>>>,
}

impl LatestFrameSource {
    pub fn spawn(name: &'static str, mut stream: Box<dyn FrameStream + Send + 'static>) -> Self {
        let latest = Arc::new(Mutex::new(None));
        let thread_latest = latest.clone();

        thread::spawn(move || {
            loop {
                match stream.next_frame() {
                    Ok(Some(frame)) => {
                        let owned = Arc::new(own_frame(frame));
                        if let Ok(mut latest) = thread_latest.lock() {
                            *latest = Some(owned);
                        }
                    }
                    Ok(None) => thread::yield_now(),
                    Err(err) => {
                        eprintln!("playground capture {name}: stopped: {err:#}");
                        return;
                    }
                }
            }
        });

        Self { latest }
    }

    pub fn next_frame(&self) -> Result<Option<Arc<OwnedFrame>>> {
        Ok(self.latest.lock().ok().and_then(|latest| latest.clone()))
    }
}

fn own_frame(frame: Frame<'_>) -> OwnedFrame {
    OwnedFrame {
        meta: frame.meta,
        format: frame.format,
        stride: frame.stride,
        data: frame.data.to_vec(),
    }
}
