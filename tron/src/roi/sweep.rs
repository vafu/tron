use crate::roi::RoiController;
use anyhow::Result;
use std::time::{Duration, Instant};
use tron_api::{Frame, PixelFormat, Rect, Size};

const LOG_INTERVAL: Duration = Duration::from_millis(250);
const MIN_SPEED: f32 = 1.0;

pub struct RoiSweep {
    enabled: bool,
    speed_px_per_sec: f32,
    direction: i32,
    last_update: Option<Instant>,
    last_log: Instant,
}

impl RoiSweep {
    pub fn new(speed_px_per_sec: f32) -> Self {
        Self {
            enabled: false,
            speed_px_per_sec: speed_px_per_sec.max(MIN_SPEED),
            direction: 1,
            last_update: None,
            last_log: Instant::now(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn speed(&self) -> f32 {
        self.speed_px_per_sec
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        self.last_update = None;
    }

    pub fn adjust_speed(&mut self, delta: f32) {
        self.speed_px_per_sec = (self.speed_px_per_sec + delta).max(MIN_SPEED);
    }

    pub fn update(
        &mut self,
        controller: &mut RoiController,
        frame_size: Size,
        now: Instant,
    ) -> Result<()> {
        if !self.enabled {
            self.last_update = Some(now);
            return Ok(());
        }

        let Some(last_update) = self.last_update.replace(now) else {
            return Ok(());
        };
        let dt = now.duration_since(last_update).as_secs_f32();
        let movement = (self.speed_px_per_sec * dt).round() as u32;
        if movement == 0 {
            return Ok(());
        }

        let mut rect = controller
            .rect()
            .non_empty_or(Size {
                width: 80,
                height: 80,
            })
            .clamp_to(frame_size);
        let max_x = frame_size.width.saturating_sub(rect.size.width);
        if self.direction >= 0 {
            rect.x = rect.x.saturating_add(movement);
            if rect.x >= max_x {
                rect.x = max_x;
                self.direction = -1;
            }
        } else {
            let next_x = rect.x.saturating_sub(movement);
            rect.x = next_x;
            if rect.x == 0 {
                self.direction = 1;
            }
        }
        controller.set_rect(rect, frame_size)
    }

    pub fn maybe_log(&mut self, frame: Frame<'_>, roi: Rect, now: Instant) {
        if !self.enabled || now.duration_since(self.last_log) < LOG_INTERVAL {
            return;
        }
        self.last_log = now;
        let Some(stats) = BrightnessStats::measure(frame, roi) else {
            return;
        };
        eprintln!(
            "roi-sweep: speed={:.1}px/s roi={}x{}@{},{} roi_mean={:.1} frame_mean={:.1}",
            self.speed_px_per_sec,
            roi.size.width,
            roi.size.height,
            roi.x,
            roi.y,
            stats.roi_mean,
            stats.frame_mean
        );
    }
}

struct BrightnessStats {
    roi_mean: f32,
    frame_mean: f32,
}

impl BrightnessStats {
    fn measure(frame: Frame<'_>, roi: Rect) -> Option<Self> {
        match frame.format {
            PixelFormat::Gray8 => gray8_stats(frame, roi),
            PixelFormat::Bgra8 | PixelFormat::Yuyv422 => None,
        }
    }
}

fn gray8_stats(frame: Frame<'_>, roi: Rect) -> Option<BrightnessStats> {
    let size = frame.meta.size;
    let roi = roi
        .non_empty_or(Size {
            width: 80,
            height: 80,
        })
        .clamp_to(size);
    let frame_pixels = size.width as usize * size.height as usize;
    if frame.stride < size.width as usize || frame.data.len() < frame.stride * size.height as usize
    {
        return None;
    }

    let frame_sum = frame
        .data
        .iter()
        .take(frame_pixels)
        .map(|v| *v as u64)
        .sum::<u64>();
    let mut roi_sum = 0_u64;
    for y in roi.y..roi.y + roi.size.height {
        let row_start = y as usize * frame.stride + roi.x as usize;
        let row_end = row_start + roi.size.width as usize;
        roi_sum += frame.data[row_start..row_end]
            .iter()
            .map(|v| *v as u64)
            .sum::<u64>();
    }

    Some(BrightnessStats {
        roi_mean: roi_sum as f32 / (roi.size.width * roi.size.height) as f32,
        frame_mean: frame_sum as f32 / frame_pixels as f32,
    })
}
