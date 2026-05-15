use std::time::{Duration, Instant};

use anyhow::Result;
use tron_api::{CameraRoiControl, Rect, Sink};

use crate::aggregate::Aggregate;

pub struct CameraRoiSink {
    control: Box<dyn CameraRoiControl>,
    update_interval: Option<Duration>,
    last_requested_rect: Option<Rect>,
    actual_rect: Option<Rect>,
    last_update: Option<Instant>,
}

impl CameraRoiSink {
    pub fn new(control: Box<dyn CameraRoiControl>, update_interval: Option<Duration>) -> Self {
        Self {
            control,
            update_interval,
            last_requested_rect: None,
            actual_rect: None,
            last_update: None,
        }
    }
}

#[async_trait::async_trait(?Send)]
impl<'a> Sink<&'a Aggregate<'a>> for CameraRoiSink {
    async fn consume(&mut self, aggregate: &'a Aggregate<'a>) -> Result<()> {
        let Some(rect) = aggregate.camera_roi else {
            return Ok(());
        };
        if self.last_requested_rect == Some(rect) {
            return Ok(());
        }
        if let Some(update_interval) = self.update_interval
            && let Some(last_update) = self.last_update
            && last_update.elapsed() < update_interval
        {
            return Ok(());
        }

        self.control.set_roi_rect(rect)?;
        self.actual_rect = Some(self.control.roi_rect()?);
        self.last_requested_rect = Some(rect);
        self.last_update = Some(Instant::now());
        Ok(())
    }
}
