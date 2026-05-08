use crate::types::{FrameContext, HandLandmarks, IrDepthMetrics, RectNorm};

pub struct DepthCueContext<'a> {
    pub frame: &'a FrameContext,
    pub roi: RectNorm,
    pub landmarks: &'a HandLandmarks,
}

pub trait DepthCueEstimator: Send {
    fn estimate(&mut self, ctx: DepthCueContext<'_>) -> Option<IrDepthMetrics>;
}

pub struct IrBrightnessDepthEstimator {
    last_corrected: Option<f32>,
}

impl IrBrightnessDepthEstimator {
    pub fn new() -> Self {
        Self {
            last_corrected: None,
        }
    }
}

impl Default for IrBrightnessDepthEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl DepthCueEstimator for IrBrightnessDepthEstimator {
    fn estimate(&mut self, ctx: DepthCueContext<'_>) -> Option<IrDepthMetrics> {
        let diff = ctx.frame.ir_diff.as_ref()?;
        if diff.width == 0
            || diff.height == 0
            || ctx.frame.ir.width != diff.width
            || ctx.frame.ir.height != diff.height
        {
            return None;
        }

        let hand_rect = crate::calib::current().unmap_rect(ctx.roi).clamped();
        if hand_rect.w <= 0.0 || hand_rect.h <= 0.0 {
            return None;
        }

        let bounds = PixelBounds::from_rect(hand_rect, diff.width, diff.height)?;
        let mut hand_diff = Vec::with_capacity(bounds.area().min(4096));
        let mut bg_diff = Vec::with_capacity((diff.pixel_count() / 8).min(8192));
        let mut raw_sum = 0u64;
        let mut clipped = 0usize;

        let diff_stride = diff.format.bytes_per_pixel();
        let raw_stride = ctx.frame.ir.format.bytes_per_pixel();
        let bg_step = 4usize;

        for y in 0..diff.height as usize {
            for x in 0..diff.width as usize {
                let in_hand = bounds.contains(x, y);
                let diff_v = diff.data[(y * diff.width as usize + x) * diff_stride];
                if in_hand {
                    hand_diff.push(diff_v);
                    let raw_v =
                        ctx.frame.ir.data[(y * ctx.frame.ir.width as usize + x) * raw_stride];
                    raw_sum += raw_v as u64;
                    if raw_v >= 245 {
                        clipped += 1;
                    }
                } else if x % bg_step == 0 && y % bg_step == 0 {
                    bg_diff.push(diff_v);
                }
            }
        }

        if hand_diff.is_empty() || bg_diff.is_empty() {
            return None;
        }

        let hand_mean = mean_u8(&hand_diff);
        let bg_mean = mean_u8(&bg_diff);
        let hand_median = median_u8(&mut hand_diff);
        let bg_median = median_u8(&mut bg_diff);
        let raw_hand_mean = raw_sum as f32 / bounds.area().max(1) as f32;
        let clip_fraction = clipped as f32 / bounds.area().max(1) as f32;
        let corrected_signal = (hand_mean - bg_median).max(0.0) / bg_mean.max(8.0);
        let delta = self
            .last_corrected
            .map(|last| corrected_signal - last)
            .unwrap_or(0.0);
        self.last_corrected = Some(corrected_signal);

        let clip_conf = (1.0 - clip_fraction * 2.0).clamp(0.0, 1.0);
        let signal_conf = ((hand_mean - bg_mean).max(0.0) / 32.0).clamp(0.0, 1.0);
        let presence_conf = ctx.landmarks.presence.clamp(0.0, 1.0);

        Some(IrDepthMetrics {
            hand_diff_mean: hand_mean,
            hand_diff_median: hand_median,
            background_diff_mean: bg_mean,
            background_diff_median: bg_median,
            raw_hand_mean,
            clip_fraction,
            corrected_signal,
            delta,
            confidence: clip_conf * signal_conf * presence_conf,
        })
    }
}

trait RectExt {
    fn clamped(self) -> Self;
}

impl RectExt for RectNorm {
    fn clamped(self) -> Self {
        let x0 = self.x.clamp(0.0, 1.0);
        let y0 = self.y.clamp(0.0, 1.0);
        let x1 = (self.x + self.w).clamp(0.0, 1.0);
        let y1 = (self.y + self.h).clamp(0.0, 1.0);
        RectNorm {
            x: x0,
            y: y0,
            w: (x1 - x0).max(0.0),
            h: (y1 - y0).max(0.0),
        }
    }
}

struct PixelBounds {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl PixelBounds {
    fn from_rect(r: RectNorm, width: u32, height: u32) -> Option<Self> {
        let w = width as f32;
        let h = height as f32;
        let x0 = (r.x * w).floor().clamp(0.0, w) as usize;
        let y0 = (r.y * h).floor().clamp(0.0, h) as usize;
        let x1 = ((r.x + r.w) * w).ceil().clamp(0.0, w) as usize;
        let y1 = ((r.y + r.h) * h).ceil().clamp(0.0, h) as usize;
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some(Self { x0, y0, x1, y1 })
    }

    fn contains(&self, x: usize, y: usize) -> bool {
        (self.x0..self.x1).contains(&x) && (self.y0..self.y1).contains(&y)
    }

    fn area(&self) -> usize {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }
}

fn mean_u8(values: &[u8]) -> f32 {
    let sum: u64 = values.iter().map(|&v| v as u64).sum();
    sum as f32 / values.len().max(1) as f32
}

fn median_u8(values: &mut [u8]) -> f32 {
    values.sort_unstable();
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] as f32 + values[mid] as f32) * 0.5
    } else {
        values[mid] as f32
    }
}
