use crate::camera::SharedImage;
use crate::filter::LandmarkFilter;
use crate::gestures::GestureClassifier;
use crate::landmarker::HandLandmarker;
use crate::proximity::SharedProx;
use crate::refiners::FrameContextRefiner;
use crate::roi::RoiHinter;
pub use crate::types::{FrameContext, HandLandmarks, HandState, Image};
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

/// Orchestrates one frame through replaceable stages:
///
/// 1. `FrameContextRefiner`: enrich or mutate frame inputs before inference.
/// 2. `RoiHinter`: acquire or track the hand crop.
/// 3. `HandLandmarker`: extract landmarks from the selected crop.
/// 4. `LandmarkFilter`: smooth or otherwise post-process landmarks.
/// 5. `GestureClassifier`: classify the final landmark state.
///
/// Keep device I/O, model runtime setup, and rendering outside this type; this
/// layer should remain a small composition root for tracking/extraction logic.
pub struct GesturePipeline {
    pub refiners: Vec<Box<dyn FrameContextRefiner>>,
    pub roi: Box<dyn RoiHinter>,
    pub lm: Box<dyn HandLandmarker>,
    pub filter: Box<dyn LandmarkFilter>,
    pub gestures: Box<dyn GestureClassifier>,
    last_outcome: StepOutcome,
    frame: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StepOutcome {
    Ok,
    NoRoi,
    NoLandmarks,
}

impl GesturePipeline {
    pub fn new(
        refiners: Vec<Box<dyn FrameContextRefiner>>,
        roi: Box<dyn RoiHinter>,
        lm: Box<dyn HandLandmarker>,
        filter: Box<dyn LandmarkFilter>,
        gestures: Box<dyn GestureClassifier>,
    ) -> Self {
        Self {
            refiners,
            roi,
            lm,
            filter,
            gestures,
            last_outcome: StepOutcome::Ok,
            frame: 0,
        }
    }

    pub fn step(&mut self, mut ctx: FrameContext) -> StepOutput {
        self.frame = self.frame.wrapping_add(1);
        let had_last = ctx.last.is_some();
        let trace = self.frame % 30 == 0;
        let t_step = Instant::now();

        // Per-refiner timings (only if tracing this frame).
        let mut refiner_us: [u32; 8] = [0; 8];
        for (i, r) in self.refiners.iter_mut().enumerate() {
            let t = Instant::now();
            r.refine(&mut ctx);
            if trace && i < refiner_us.len() {
                refiner_us[i] = t.elapsed().as_micros() as u32;
            }
        }
        let ir_diff = ctx.ir_diff.clone();

        let t_roi = Instant::now();
        let (roi, _debug) = self.roi.hint(&ctx);
        let roi_us = t_roi.elapsed().as_micros() as u32;
        let Some(roi) = roi else {
            self.transition(StepOutcome::NoRoi, had_last, &ctx);
            if trace {
                self.log_trace("no-roi", t_step.elapsed(), &refiner_us, roi_us, 0, 0, 0);
            }
            return StepOutput {
                state: None,
                ir_diff,
            };
        };

        let t_lm = Instant::now();
        let raw = self.lm.run(&ctx, Some(roi));
        let lm_us = t_lm.elapsed().as_micros() as u32;
        let Some(raw) = raw else {
            self.transition(StepOutcome::NoLandmarks, had_last, &ctx);
            if trace {
                self.log_trace("no-lm", t_step.elapsed(), &refiner_us, roi_us, lm_us, 0, 0);
            }
            return StepOutput {
                state: None,
                ir_diff,
            };
        };

        let t_filter = Instant::now();
        let smoothed = self.filter.apply(raw);
        let filter_us = t_filter.elapsed().as_micros() as u32;
        let t_gest = Instant::now();
        let gesture = self.gestures.classify(&ctx, &smoothed);
        let gest_us = t_gest.elapsed().as_micros() as u32;

        self.transition(StepOutcome::Ok, had_last, &ctx);
        if trace {
            self.log_trace(
                "ok",
                t_step.elapsed(),
                &refiner_us,
                roi_us,
                lm_us,
                filter_us,
                gest_us,
            );
        }
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

    fn log_trace(
        &self,
        outcome: &str,
        total: Duration,
        refiners_us: &[u32; 8],
        roi_us: u32,
        lm_us: u32,
        filter_us: u32,
        gest_us: u32,
    ) {
        let n = self.refiners.len().min(refiners_us.len());
        // Format active refiners as e.g. "[123,4500,8900]"
        let mut refs = String::with_capacity(32);
        refs.push('[');
        for (i, us) in refiners_us[..n].iter().enumerate() {
            if i > 0 {
                refs.push(',');
            }
            refs.push_str(&format!("{}", us));
        }
        refs.push(']');
        eprintln!(
            "trace frame={} {} total={:.1}ms refiners={}us roi={:.1}ms lm={:.1}ms filter={}us gest={}us",
            self.frame,
            outcome,
            total.as_secs_f32() * 1000.0,
            refs,
            roi_us as f32 / 1000.0,
            lm_us as f32 / 1000.0,
            filter_us,
            gest_us,
        );
    }

    fn transition(&mut self, now: StepOutcome, had_last: bool, ctx: &FrameContext) {
        if now != self.last_outcome {
            let from = match self.last_outcome {
                StepOutcome::Ok => "ok",
                StepOutcome::NoRoi => "no-roi",
                StepOutcome::NoLandmarks => "no-lm",
            };
            let to = match now {
                StepOutcome::Ok => "ok",
                StepOutcome::NoRoi => "no-roi",
                StepOutcome::NoLandmarks => "no-lm",
            };
            eprintln!(
                "pipeline: {from}->{to} frame={} had_last={} flashlight={} ir_diff={}",
                self.frame,
                had_last,
                ctx.ir_flashlight_on,
                ctx.ir_diff.is_some()
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
            let mut frame: u64 = 0;
            let mut last_published = Instant::now();

            loop {
                let t_lock = Instant::now();
                let rgb_img = rgb.lock().unwrap().clone();
                let ir_img = ir.lock().unwrap().clone();
                let prox_v = *prox.lock().unwrap();
                let lock_us = t_lock.elapsed().as_micros() as u32;

                let (Some(rgb_img), Some(ir_img)) = (rgb_img, ir_img) else {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                };
                if rgb_img.seq == last_rgb_seq {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                }
                // Frame age — how stale was this frame when we picked it up?
                let frame_age_ms = rgb_img.timestamp.elapsed().as_secs_f32() * 1000.0;
                let interval_ms = last_published.elapsed().as_secs_f32() * 1000.0;
                last_published = Instant::now();
                last_rgb_seq = rgb_img.seq;
                frame = frame.wrapping_add(1);

                let ctx = FrameContext {
                    rgb: rgb_img,
                    ir: ir_img,
                    ir_diff: None,
                    // Refiners decide the real value; this is just an init.
                    ir_flashlight_on: true,
                    proximity: prox_v,
                    last: last.clone(),
                    now: Instant::now(),
                };

                let StepOutput { state, ir_diff } = pipeline.step(ctx);
                *publish_mask.lock().unwrap() = ir_diff;
                last = state.as_ref().map(|s| s.landmarks.clone()).or(last);
                *publish.lock().unwrap() = state;

                if frame % 30 == 0 {
                    eprintln!(
                        "spawn: frame={} interval={:.1}ms input_age={:.1}ms lock={}us",
                        frame, interval_ms, frame_age_ms, lock_us
                    );
                }
            }
        })
        .expect("spawn gesture thread");
    PipelineOutputs { hand: out, mask }
}
