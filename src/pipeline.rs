use crate::camera::SharedImage;
use crate::filter::LandmarkFilter;
use crate::gestures::GestureClassifier;
use crate::landmarker::HandLandmarker;
use crate::proximity::SharedProx;
use crate::roi::RoiHinter;
use crate::types::{HandLandmarks, HandState, Image};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub type SharedHand = Arc<Mutex<Option<HandState>>>;

/// Snapshot of inputs handed to every pipeline stage on each tick.
pub struct FrameContext<'a> {
    pub rgb: Option<&'a Image>,
    pub ir:  Option<&'a Image>,
    pub proximity: Option<i64>,
    /// Previous frame's smoothed landmarks — for ROI tracking and temporal
    /// gestures.
    pub last: Option<&'a HandLandmarks>,
    pub now: Instant,
}

pub struct GesturePipeline {
    pub roi:      Box<dyn RoiHinter>,
    pub lm:       Box<dyn HandLandmarker>,
    pub filter:   Box<dyn LandmarkFilter>,
    pub gestures: Box<dyn GestureClassifier>,
}

impl GesturePipeline {
    pub fn step(&mut self, ctx: &FrameContext) -> Option<HandState> {
        let roi = self.roi.hint(ctx);
        let raw = self.lm.run(ctx, roi)?;
        let smoothed = self.filter.apply(raw);
        let gesture = self.gestures.classify(ctx, &smoothed);
        Some(HandState { landmarks: smoothed, gesture })
    }
}

/// Spawn the pipeline thread. Reads the latest snapshot from each input source
/// and publishes the latest `HandState` for the renderer.
pub fn spawn(
    rgb: SharedImage,
    ir: SharedImage,
    prox: SharedProx,
    mut pipeline: GesturePipeline,
) -> SharedHand {
    let out: SharedHand = Arc::new(Mutex::new(None));
    let publish = out.clone();

    thread::Builder::new()
        .name("gesture".into())
        .spawn(move || {
            let mut last_rgb_seq: u64 = u64::MAX;
            let mut last: Option<HandLandmarks> = None;

            loop {
                let rgb_img = rgb.lock().unwrap().clone();
                let ir_img = ir.lock().unwrap().clone();
                let prox_v = *prox.lock().unwrap();

                let advanced = match &rgb_img {
                    Some(img) => img.seq != last_rgb_seq,
                    None => false,
                };

                if advanced {
                    last_rgb_seq = rgb_img.as_ref().unwrap().seq;
                    let ctx = FrameContext {
                        rgb: rgb_img.as_ref(),
                        ir: ir_img.as_ref(),
                        proximity: prox_v,
                        last: last.as_ref(),
                        now: Instant::now(),
                    };
                    if let Some(state) = pipeline.step(&ctx) {
                        last = Some(state.landmarks.clone());
                        *publish.lock().unwrap() = Some(state);
                    } else {
                        last = None;
                        *publish.lock().unwrap() = None;
                    }
                } else {
                    thread::sleep(Duration::from_millis(2));
                }
            }
        })
        .expect("spawn gesture thread");
    out
}
