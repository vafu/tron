use crate::camera::SharedImage;
use crate::filter::LandmarkFilter;
use crate::gestures::GestureClassifier;
use crate::landmarker::HandLandmarker;
use crate::proximity::SharedProx;
use crate::roi::RoiHinter;
pub use crate::types::{HandLandmarks, HandState, Image, FrameContext};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub type SharedHand = Arc<Mutex<Option<HandState>>>;
pub type SharedMask = Arc<Mutex<Option<Image>>>;

pub struct PipelineOutputs {
    pub hand: SharedHand,
    pub mask: SharedMask,
}

pub struct StepOutput {
    pub state: Option<HandState>,
    /// Final IR-diff grayscale image visible to the UI (post-refiner).
    pub ir_diff: Option<Image>,
}

pub trait FrameContextRefiner: Send {
    fn refine(&mut self, ctx: &mut FrameContext);
}

pub struct GesturePipeline {
    pub refiners: Vec<Box<dyn FrameContextRefiner>>,
    pub roi:      Box<dyn RoiHinter>,
    pub lm:       Box<dyn HandLandmarker>,
    pub filter:   Box<dyn LandmarkFilter>,
    pub gestures: Box<dyn GestureClassifier>,
    last_outcome: StepOutcome,
    frame: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StepOutcome { Ok, NoRoi, NoLandmarks }

impl GesturePipeline {
    pub fn new(
        refiners: Vec<Box<dyn FrameContextRefiner>>,
        roi: Box<dyn RoiHinter>,
        lm: Box<dyn HandLandmarker>,
        filter: Box<dyn LandmarkFilter>,
        gestures: Box<dyn GestureClassifier>,
    ) -> Self {
        Self { refiners, roi, lm, filter, gestures, last_outcome: StepOutcome::Ok, frame: 0 }
    }

    pub fn step(&mut self, mut ctx: FrameContext) -> StepOutput {
        self.frame = self.frame.wrapping_add(1);
        let had_last = ctx.last.is_some();

        for r in &mut self.refiners {
            r.refine(&mut ctx);
        }
        let ir_diff = ctx.ir_diff.clone();

        let (roi, _debug) = self.roi.hint(&ctx);
        let Some(roi) = roi else {
            self.transition(StepOutcome::NoRoi, had_last, &ctx);
            return StepOutput { state: None, ir_diff };
        };

        let raw = match self.lm.run(&ctx, Some(roi)) {
            Some(r) => r,
            None => {
                self.transition(StepOutcome::NoLandmarks, had_last, &ctx);
                if self.frame % 30 == 0 {
                    eprintln!(
                        "pipeline: lm gated frame={} roi=[{:.2},{:.2} {:.2}x{:.2}] had_last={}",
                        self.frame, roi.x, roi.y, roi.w, roi.h, had_last
                    );
                }
                return StepOutput { state: None, ir_diff };
            }
        };
        let smoothed = self.filter.apply(raw);
        let gesture = self.gestures.classify(&ctx, &smoothed);

        self.transition(StepOutcome::Ok, had_last, &ctx);
        StepOutput {
            state: Some(HandState {
                roi,
                landmarks: smoothed,
                gesture,
                debug_image: Some(ctx.rgb.clone()),
            }),
            ir_diff,
        }
    }

    fn transition(&mut self, now: StepOutcome, had_last: bool, ctx: &FrameContext) {
        if now != self.last_outcome {
            let from = match self.last_outcome { StepOutcome::Ok => "ok", StepOutcome::NoRoi => "no-roi", StepOutcome::NoLandmarks => "no-lm" };
            let to   = match now                { StepOutcome::Ok => "ok", StepOutcome::NoRoi => "no-roi", StepOutcome::NoLandmarks => "no-lm" };
            eprintln!(
                "pipeline: {from}->{to} frame={} had_last={} flashlight={} ir_diff={}",
                self.frame, had_last, ctx.ir_flashlight_on, ctx.ir_diff.is_some()
            );
            self.last_outcome = now;
        }
    }
}

pub fn spawn(
    rgb: SharedImage,
    ir: SharedImage,
    prox: SharedProx,
    mut pipeline: GesturePipeline,
) -> PipelineOutputs {
    let out: SharedHand = Arc::new(Mutex::new(None));
    let mask: SharedMask = Arc::new(Mutex::new(None));
    let publish = out.clone();
    let publish_mask = mask.clone();

    thread::Builder::new()
        .name("gesture".into())
        .spawn(move || {
            let mut last_rgb_seq: u64 = u64::MAX;
            let mut last: Option<HandLandmarks> = None;
            let mut ir_avg_intensity: f32 = 0.0;

            loop {
                let (rgb_img, ir_img, prox_v) = {
                    let r = rgb.lock().unwrap().clone();
                    let i = ir.lock().unwrap().clone();
                    let p = *prox.lock().unwrap();
                    (r, i, p)
                };

                let advanced = match &rgb_img {
                    Some(img) => img.seq != last_rgb_seq,
                    None => false,
                };

                if advanced && rgb_img.is_some() && ir_img.is_some() {
                    let rgb_img = rgb_img.unwrap();
                    let ir_img = ir_img.unwrap();
                    last_rgb_seq = rgb_img.seq;
                    
                    let mut flashlight_on = true;
                    let mean = ir_mean(&ir_img);
                    if ir_avg_intensity == 0.0 {
                        ir_avg_intensity = mean;
                    } else {
                        ir_avg_intensity = ir_avg_intensity * 0.9 + mean * 0.1;
                    }
                    if mean < ir_avg_intensity * 0.95 {
                        flashlight_on = false;
                    }

                    let ctx = FrameContext {
                        rgb: rgb_img,
                        ir: ir_img,
                        ir_diff: None,
                        ir_flashlight_on: flashlight_on,
                        proximity: prox_v,
                        last: last.clone(),
                        now: Instant::now(),
                    };

                    let StepOutput { state, ir_diff } = pipeline.step(ctx);
                    *publish_mask.lock().unwrap() = ir_diff;
                    if let Some(state) = state {
                        last = Some(state.landmarks.clone());
                        *publish.lock().unwrap() = Some(state);
                    } else {
                        *publish.lock().unwrap() = None;
                    }
                } else {
                    thread::sleep(Duration::from_millis(2));
                }
            }
        })
        .expect("spawn gesture thread");
    PipelineOutputs { hand: out, mask }
}

fn ir_mean(img: &Image) -> f32 {
    let mut sum: u64 = 0;
    match img.format {
        crate::types::PixelFormat::Rgba8 => {
            for i in 0..(img.data.len() / 4) {
                sum += img.data[i * 4] as u64;
            }
        }
        crate::types::PixelFormat::R8 => {
            for &g in &img.data {
                sum += g as u64;
            }
        }
    }
    sum as f32 / (img.width * img.height) as f32
}

// --- Implementation of Refiners ---

pub struct TemporalSubtractionRefiner {
    last_bg: Option<opencv::core::Mat>,
}

impl TemporalSubtractionRefiner {
    pub fn new() -> Self {
        Self { last_bg: None }
    }
}

impl FrameContextRefiner for TemporalSubtractionRefiner {
    fn refine(&mut self, ctx: &mut FrameContext) {
        use opencv::prelude::*;
        use opencv::core::{Mat, CV_8UC1};

        let w = ctx.ir.width as i32;
        let h = ctx.ir.height as i32;
        
        let mut grey = vec![0u8; (w * h) as usize];
        match ctx.ir.format {
            crate::types::PixelFormat::Rgba8 => {
                for (i, g) in grey.iter_mut().enumerate() {
                    *g = ctx.ir.data[i * 4];
                }
            }
            crate::types::PixelFormat::R8 => {
                grey.copy_from_slice(&ctx.ir.data);
            }
        }
        
        let current = unsafe {
            Mat::new_rows_cols_with_data_unsafe_def(h, w, CV_8UC1, grey.as_mut_ptr() as *mut _).unwrap()
        }.clone();

        if !ctx.ir_flashlight_on {
            self.last_bg = Some(current);
            ctx.ir_diff = None;
            return;
        }

        if let Some(bg) = &self.last_bg {
            let mut diff = Mat::default();
            if opencv::core::subtract(&current, bg, &mut diff, &opencv::core::no_array(), -1).is_ok() {
                let mut diff_data = vec![0u8; (w * h) as usize];
                if let Ok(data) = diff.data_bytes() {
                    diff_data.copy_from_slice(data);
                }
                ctx.ir_diff = Some(Image {
                    data: diff_data,
                    width: ctx.ir.width,
                    height: ctx.ir.height,
                    format: crate::types::PixelFormat::R8,
                    timestamp: ctx.ir.timestamp,
                    seq: ctx.ir.seq,
                });
            }
        }
    }
}

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
        let diff = match ctx.ir_diff.as_ref().or(self.last_diff.as_ref()) {
            Some(d) => d,
            None => return,
        };
        
        let calib = crate::calib::current();
        let w = ctx.rgb.width as f32;
        let h = ctx.rgb.height as f32;
        
        for y in 0..ctx.rgb.height {
            let ny = (y as f32 + 0.5) / h;
            for x in 0..ctx.rgb.width {
                let nx = (x as f32 + 0.5) / w;
                
                let nir_x = (nx - calib.offset_x) / calib.scale_x;
                let nir_y = (ny - calib.offset_y) / calib.scale_y;
                
                let mut mask = 0.2f32;
                if nir_x >= 0.0 && nir_x < 1.0 && nir_y >= 0.0 && nir_y < 1.0 {
                    let ix = (nir_x * diff.width as f32) as usize;
                    let iy = (nir_y * diff.height as f32) as usize;
                    let ix = ix.min(diff.width as usize - 1);
                    let iy = iy.min(diff.height as usize - 1);
                    
                    let ir_val = diff.data[iy * diff.width as usize + ix] as f32 / 255.0;
                    mask = 0.2 + 0.8 * ir_val;
                }
                
                let i = (y * ctx.rgb.width + x) as usize * 4;
                ctx.rgb.data[i]     = (ctx.rgb.data[i] as f32 * mask) as u8;
                ctx.rgb.data[i + 1] = (ctx.rgb.data[i + 1] as f32 * mask) as u8;
                ctx.rgb.data[i + 2] = (ctx.rgb.data[i + 2] as f32 * mask) as u8;
            }
        }
    }
}
