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

            loop {
                let rgb_img = rgb.lock().unwrap().clone();
                let ir_img = ir.lock().unwrap().clone();
                let prox_v = *prox.lock().unwrap();

                let (Some(rgb_img), Some(ir_img)) = (rgb_img, ir_img) else {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                };
                if rgb_img.seq == last_rgb_seq {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                }
                last_rgb_seq = rgb_img.seq;

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
            }
        })
        .expect("spawn gesture thread");
    PipelineOutputs { hand: out, mask }
}
