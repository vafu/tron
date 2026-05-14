use std::time::{Duration, Instant};

use anyhow::Result;
use tron_api::{CameraRoiControl, NoContext, Processor, Rect, RoiResult, Size};
use tron_core::roi::camera::{
    CameraRoiFollowConfig, CameraRoiFollowInput, CameraRoiFollowProcessor,
};

#[derive(Clone, Debug)]
pub struct CameraRoiConfig {
    pub min_edge: u32,
    pub update_interval: Option<Duration>,
}

pub struct CameraRoiDriver {
    config: CameraRoiConfig,
    control: Box<dyn CameraRoiControl>,
    follow: CameraRoiFollowProcessor,
    actual_rect: Option<Rect>,
    last_requested_rect: Option<Rect>,
    last_update: Option<Instant>,
}

impl CameraRoiDriver {
    pub fn new(config: CameraRoiConfig, control: Box<dyn CameraRoiControl>) -> Self {
        Self {
            follow: CameraRoiFollowProcessor::new(CameraRoiFollowConfig {
                min_edge: config.min_edge,
            }),
            config,
            control,
            actual_rect: None,
            last_requested_rect: None,
            last_update: None,
        }
    }

    pub fn current_rect(&self) -> Option<Rect> {
        self.actual_rect
    }

    pub fn update(
        &mut self,
        roi: Option<RoiResult>,
        allowed_bounds: Option<Rect>,
        frame_size: Size,
    ) -> Result<()> {
        let Some(roi) = roi else {
            return Ok(());
        };
        let allowed_bounds = allowed_bounds
            .map(|bounds| clamp_rect(bounds, frame_size))
            .unwrap_or(Rect {
                x: 0,
                y: 0,
                size: frame_size,
            });
        let Some(rect) = self.follow.process(
            CameraRoiFollowInput {
                roi: Some(roi),
                allowed_bounds: Some(allowed_bounds),
                source_size: frame_size,
                target_size: frame_size,
            },
            NoContext,
        )?
        else {
            return Ok(());
        };
        if self.last_requested_rect == Some(rect) {
            return Ok(());
        }
        if let Some(update_interval) = self.config.update_interval
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

fn clamp_rect(rect: Rect, frame_size: Size) -> Rect {
    let x = rect.x.min(frame_size.width);
    let y = rect.y.min(frame_size.height);
    let width = rect.size.width.min(frame_size.width.saturating_sub(x));
    let height = rect.size.height.min(frame_size.height.saturating_sub(y));
    Rect {
        x,
        y,
        size: Size { width, height },
    }
}
