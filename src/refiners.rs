//! Pre-detection passes that mutate a `FrameContext` in place.
//!
//! Each refiner owns one concern: deciding whether the IR flashlight is on,
//! producing the IR foreground signal, masking the RGB image with that signal,
//! and so on. The pipeline runs them in order before ROI/landmark stages, so
//! their order in `main.rs` defines causality.

use crate::pipeline::FrameContext;
use crate::types::{Image, PixelFormat};

pub trait FrameContextRefiner: Send {
    fn refine(&mut self, ctx: &mut FrameContext);
}

/// Decides whether the IR flashlight is currently on by comparing this frame's
/// IR mean intensity against a long-running average. A sudden dip → off.
pub struct FlashlightDetectorRefiner {
    avg_intensity: f32,
}

impl FlashlightDetectorRefiner {
    pub fn new() -> Self {
        Self { avg_intensity: 0.0 }
    }
}

impl Default for FlashlightDetectorRefiner {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameContextRefiner for FlashlightDetectorRefiner {
    fn refine(&mut self, ctx: &mut FrameContext) {
        let mean = ir_mean(&ctx.ir);
        self.avg_intensity = if self.avg_intensity == 0.0 {
            mean
        } else {
            self.avg_intensity * 0.9 + mean * 0.1
        };
        ctx.ir_flashlight_on = mean >= self.avg_intensity * 0.95;
    }
}

fn ir_mean(img: &Image) -> f32 {
    let sum: u64 = img.grey_iter().map(|g| g as u64).sum();
    sum as f32 / img.pixel_count().max(1) as f32
}

/// Subtracts a slowly-updated IR background from the current IR frame to
/// expose the moving foreground. Stores the result in `ctx.ir_diff` (R8).
pub struct TemporalSubtractionRefiner {
    last_bg: Option<opencv::core::Mat>,
}

impl TemporalSubtractionRefiner {
    pub fn new() -> Self {
        Self { last_bg: None }
    }
}

impl Default for TemporalSubtractionRefiner {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameContextRefiner for TemporalSubtractionRefiner {
    fn refine(&mut self, ctx: &mut FrameContext) {
        use opencv::core::{Mat, CV_8UC1};
        use opencv::prelude::*;

        let w = ctx.ir.width as i32;
        let h = ctx.ir.height as i32;

        let mut grey: Vec<u8> = ctx.ir.grey_iter().collect();

        let current = unsafe {
            Mat::new_rows_cols_with_data_unsafe_def(h, w, CV_8UC1, grey.as_mut_ptr() as *mut _)
                .unwrap()
        }
        .clone();

        if !ctx.ir_flashlight_on {
            self.last_bg = Some(current);
            ctx.ir_diff = None;
            return;
        }

        let Some(bg) = &self.last_bg else { return };
        let mut diff = Mat::default();
        if opencv::core::subtract(&current, bg, &mut diff, &opencv::core::no_array(), -1).is_err() {
            return;
        }
        let mut diff_data = vec![0u8; ctx.ir.pixel_count()];
        if let Ok(data) = diff.data_bytes() {
            diff_data.copy_from_slice(data);
        }
        ctx.ir_diff = Some(Image {
            data: diff_data,
            width: ctx.ir.width,
            height: ctx.ir.height,
            format: PixelFormat::R8,
            timestamp: ctx.ir.timestamp,
            seq: ctx.ir.seq,
        });
    }
}

/// Multiplies RGB by the IR foreground signal so background pixels are
/// dimmed and the moving hand stands out. Caches the most recent diff so
/// flashlight-off frames can reuse it instead of producing an unmasked frame.
pub struct RgbMaskingRefiner {
    last_diff: Option<Image>,
}

impl RgbMaskingRefiner {
    pub fn new() -> Self {
        Self { last_diff: None }
    }
}

impl Default for RgbMaskingRefiner {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameContextRefiner for RgbMaskingRefiner {
    fn refine(&mut self, ctx: &mut FrameContext) {
        if let Some(d) = &ctx.ir_diff {
            self.last_diff = Some(d.clone());
        }
        let Some(diff) = ctx.ir_diff.as_ref().or(self.last_diff.as_ref()) else {
            return;
        };

        let calib = crate::calib::current();
        let w = ctx.rgb.width as f32;
        let h = ctx.rgb.height as f32;
        let dw = diff.width as usize;
        let dh = diff.height as usize;

        for y in 0..ctx.rgb.height {
            let ny = (y as f32 + 0.5) / h;
            for x in 0..ctx.rgb.width {
                let nx = (x as f32 + 0.5) / w;
                let nir_x = (nx - calib.offset_x) / calib.scale_x;
                let nir_y = (ny - calib.offset_y) / calib.scale_y;

                let mask = if (0.0..1.0).contains(&nir_x) && (0.0..1.0).contains(&nir_y) {
                    let ix = ((nir_x * dw as f32) as usize).min(dw - 1);
                    let iy = ((nir_y * dh as f32) as usize).min(dh - 1);
                    let v = diff.data[iy * dw + ix] as f32 / 255.0;
                    0.2 + 0.8 * v
                } else {
                    0.2
                };

                let i = (y * ctx.rgb.width + x) as usize * 4;
                ctx.rgb.data[i]     = (ctx.rgb.data[i] as f32 * mask) as u8;
                ctx.rgb.data[i + 1] = (ctx.rgb.data[i + 1] as f32 * mask) as u8;
                ctx.rgb.data[i + 2] = (ctx.rgb.data[i + 2] as f32 * mask) as u8;
            }
        }
    }
}
