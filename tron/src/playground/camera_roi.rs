use std::time::{Duration, Instant};

use anyhow::Result;
use tron_api::{CameraRoiControl, Rect, RoiResult, Size};

#[derive(Clone, Debug)]
pub struct CameraRoiConfig {
    pub min_edge: u32,
    pub update_interval: Option<Duration>,
}

pub struct CameraRoiDriver {
    config: CameraRoiConfig,
    control: Box<dyn CameraRoiControl>,
    actual_rect: Option<Rect>,
    last_requested_rect: Option<Rect>,
    last_update: Option<Instant>,
}

impl CameraRoiDriver {
    pub fn new(config: CameraRoiConfig, control: Box<dyn CameraRoiControl>) -> Self {
        Self {
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
        let rect = expand_to_min_edge(roi.rect, self.config.min_edge, allowed_bounds);
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

fn expand_to_min_edge(rect: Rect, min_edge: u32, bounds: Rect) -> Rect {
    let width = rect.size.width.max(min_edge).min(bounds.size.width).max(1);
    let height = rect
        .size
        .height
        .max(min_edge)
        .min(bounds.size.height)
        .max(1);
    let bx1 = bounds.x.saturating_add(bounds.size.width);
    let by1 = bounds.y.saturating_add(bounds.size.height);
    let cx = rect
        .x
        .saturating_add(rect.size.width / 2)
        .clamp(bounds.x, bx1);
    let cy = rect
        .y
        .saturating_add(rect.size.height / 2)
        .clamp(bounds.y, by1);
    let x = cx
        .saturating_sub(width / 2)
        .clamp(bounds.x, bx1.saturating_sub(width));
    let y = cy
        .saturating_sub(height / 2)
        .clamp(bounds.y, by1.saturating_sub(height));
    Rect {
        x,
        y,
        size: Size { width, height },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_roi_inside_allowed_bounds() {
        let rect = expand_to_min_edge(
            Rect {
                x: 95,
                y: 95,
                size: Size {
                    width: 5,
                    height: 5,
                },
            },
            40,
            Rect {
                x: 80,
                y: 80,
                size: Size {
                    width: 50,
                    height: 50,
                },
            },
        );

        assert_eq!(
            rect,
            Rect {
                x: 80,
                y: 80,
                size: Size {
                    width: 40,
                    height: 40
                }
            }
        );
    }
}
