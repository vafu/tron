use std::sync::Arc;
use std::time::Instant;
use std::{fs::File, path::PathBuf};

use anyhow::{Context, Result};
use tron_api::{
    CheckerboardDetection, CheckerboardSample, CheckerboardSpec, OwnedFrame, Presenter, Processor,
    Size,
};
use tron_core::calib::checkerboard::{
    CheckerboardSampleBuilder, OpenCvCheckerboardConfig, OpenCvCheckerboardDetector,
    calibrate_stereo_checkerboard, calibration_frame_side,
};
use tron_core::pipeline::{FrameStream, FrameSynchronizer};
use tron_core::view::IntoView;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

use crate::latency::{CalibrationLatencyLog, CalibrationLatencySample};
use crate::presenter::{CalibrationPresenter, CalibrationView};

pub struct CalibrationRunConfig {
    pub checkerboard: CheckerboardSpec,
    pub min_samples: usize,
    pub max_sync_delta_us: i64,
    pub output: PathBuf,
}

pub fn run<R, I>(rgb: R, ir: I, config: CalibrationRunConfig) -> Result<()>
where
    R: FrameStream,
    I: FrameStream,
{
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(rgb, ir, config);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp<R, I> {
    synchronizer: FrameSynchronizer<R, I>,
    rgb_checkerboard: CheckerboardDetectionState,
    ir_checkerboard: CheckerboardDetectionState,
    sample_builder: CheckerboardSampleBuilder,
    samples: Vec<CheckerboardSample>,
    last_sample_pair: Option<(u64, u64)>,
    capture_requested: bool,
    min_samples: usize,
    output: PathBuf,
    window_id: Option<WindowId>,
    presenter: Option<CalibrationPresenter>,
    window: Option<Arc<winit::window::Window>>,
    latency: CalibrationLatencyLog,
    result: Result<()>,
}

impl<R, I> WindowApp<R, I>
where
    R: FrameStream,
    I: FrameStream,
{
    fn new(rgb: R, ir: I, config: CalibrationRunConfig) -> Self {
        let checkerboard = OpenCvCheckerboardConfig::new(config.checkerboard);
        Self {
            synchronizer: FrameSynchronizer::new(rgb, ir, config.max_sync_delta_us),
            rgb_checkerboard: CheckerboardDetectionState::new(checkerboard),
            ir_checkerboard: CheckerboardDetectionState::new(checkerboard),
            sample_builder: CheckerboardSampleBuilder::new(config.checkerboard),
            samples: Vec::new(),
            last_sample_pair: None,
            capture_requested: false,
            min_samples: config.min_samples,
            output: config.output,
            window_id: None,
            window: None,
            presenter: None,
            latency: CalibrationLatencyLog::default(),
            result: Ok(()),
        }
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }

    fn capture_sample(
        &mut self,
        rgb: Option<&OwnedFrame>,
        ir: Option<&OwnedFrame>,
        rgb_detection: Option<&CheckerboardDetection>,
        ir_detection: Option<&CheckerboardDetection>,
    ) -> Result<()> {
        let Some(rgb) = rgb else {
            eprintln!("tron-calibration: cannot capture sample, RGB frame is not available");
            return Ok(());
        };
        let Some(ir) = ir else {
            eprintln!("tron-calibration: cannot capture sample, IR frame is not available");
            return Ok(());
        };
        let pair = (rgb.meta.id, ir.meta.id);
        if self.last_sample_pair == Some(pair) {
            eprintln!(
                "tron-calibration: skipped duplicate sample for frames {} / {}",
                pair.0, pair.1
            );
            return Ok(());
        }

        let sample = self.sample_builder.process(
            (
                rgb_detection.map(|detection| calibration_frame_side(rgb.meta, detection)),
                ir_detection.map(|detection| calibration_frame_side(ir.meta, detection)),
            ),
            tron_api::NoContext,
        )?;

        let Some(sample) = sample else {
            eprintln!(
                "tron-calibration: cannot capture sample, checkerboard is not detected in both feeds"
            );
            return Ok(());
        };

        self.samples.push(sample);
        self.last_sample_pair = Some(pair);
        eprintln!(
            "tron-calibration: captured sample {} from frames {} / {}",
            self.samples.len(),
            pair.0,
            pair.1
        );
        Ok(())
    }

    fn write_calibration(&self) -> Result<()> {
        anyhow::ensure!(
            self.samples.len() >= self.min_samples,
            "need at least {} samples to calibrate, have {}",
            self.min_samples,
            self.samples.len()
        );
        let calibration = calibrate_stereo_checkerboard(&self.samples)?;
        let file = File::create(&self.output)
            .with_context(|| format!("create {}", self.output.display()))?;
        serde_json::to_writer_pretty(file, &calibration)
            .with_context(|| format!("write {}", self.output.display()))?;
        eprintln!(
            "tron-calibration: wrote {} samples to {} (stereo reprojection error {:.4})",
            self.samples.len(),
            self.output.display(),
            calibration.stereo_reprojection_error
        );
        Ok(())
    }
}

impl<R, I> ApplicationHandler for WindowApp<R, I>
where
    R: FrameStream,
    I: FrameStream,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.presenter.is_some() {
            return;
        }

        let attrs = WindowAttributes::default().with_title("tron calibration");
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.set_error(event_loop, anyhow::Error::new(err).context("create window"));
                return;
            }
        };
        self.window_id = Some(window.id());
        let size = window.inner_size();
        match pollster::block_on(CalibrationPresenter::new(
            window.clone(),
            Size {
                width: size.width,
                height: size.height,
            },
        )) {
            Ok(presenter) => {
                self.window = Some(window);
                self.presenter = Some(presenter);
            }
            Err(err) => self.set_error(event_loop, err),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(presenter) = self.presenter.as_mut() {
                    presenter.resize(Size {
                        width: size.width,
                        height: size.height,
                    });
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed || event.repeat {
                    return;
                }
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Space) => {
                        self.capture_requested = true;
                    }
                    Key::Character(key) if key.eq_ignore_ascii_case("c") => {
                        if let Err(err) = self.write_calibration() {
                            eprintln!("tron-calibration: {err:#}");
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                let redraw_start = Instant::now();
                let sync_start = Instant::now();
                // Timestamp pairing is intentionally disabled while we diagnose
                // calibration window startup behavior. Keep FrameSynchronizer
                // in place, but only use it as a temporary owner for both
                // streams.
                // let pair = match tron_core::pipeline::FramePairStream::next_pair(&mut self.synchronizer) {
                let pair = match self.synchronizer.next_unsynchronized_pair() {
                    Ok(pair) => pair,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };
                let latest = sync_start.elapsed();
                let Some(pair) = pair else {
                    self.rgb_checkerboard.clear();
                    self.ir_checkerboard.clear();
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                    return;
                };
                let sync_delta_us = pair.delta_us;
                let rgb = pair.left;
                let ir = pair.right;

                let rgb_detect_start = Instant::now();
                if let Err(err) = self.rgb_checkerboard.update(Some(&rgb)) {
                    self.set_error(event_loop, err);
                    return;
                }
                let rgb_detect = rgb_detect_start.elapsed();

                let ir_detect_start = Instant::now();
                if let Err(err) = self.ir_checkerboard.update(Some(&ir)) {
                    self.set_error(event_loop, err);
                    return;
                }
                let ir_detect = ir_detect_start.elapsed();

                if self.capture_requested {
                    self.capture_requested = false;
                    let rgb_checkerboard = self.rgb_checkerboard.detection().cloned();
                    let ir_checkerboard = self.ir_checkerboard.detection().cloned();
                    if let Err(err) = self.capture_sample(
                        Some(&rgb),
                        Some(&ir),
                        rgb_checkerboard.as_ref(),
                        ir_checkerboard.as_ref(),
                    ) {
                        self.set_error(event_loop, err);
                        return;
                    }
                }

                let rgb_checkerboard = self.rgb_checkerboard.detection();
                let ir_checkerboard = self.ir_checkerboard.detection();

                let present_start = Instant::now();
                let Some(presenter) = self.presenter.as_mut() else {
                    return;
                };
                if let Err(err) = presenter.present(CalibrationView {
                    rgb: Some(rgb.as_frame()),
                    ir: Some(ir.as_frame()),
                    rgb_checkerboard,
                    ir_checkerboard,
                }) {
                    self.set_error(event_loop, err);
                    return;
                }
                let present = present_start.elapsed();
                let finished_at = Instant::now();
                self.latency.record(CalibrationLatencySample {
                    latest,
                    rgb_detect,
                    ir_detect,
                    present,
                    total: finished_at.saturating_duration_since(redraw_start),
                    finished_at,
                    rgb: Some(&rgb),
                    ir: Some(&ir),
                    sync_delta_us: Some(sync_delta_us),
                    rgb_detected: rgb_checkerboard.is_some(),
                    ir_detected: ir_checkerboard.is_some(),
                });
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

struct CheckerboardDetectionState {
    detector: OpenCvCheckerboardDetector,
    last_frame_id: Option<u64>,
    detection: Option<CheckerboardDetection>,
}

impl CheckerboardDetectionState {
    fn new(config: OpenCvCheckerboardConfig) -> Self {
        Self {
            detector: OpenCvCheckerboardDetector::new(config),
            last_frame_id: None,
            detection: None,
        }
    }

    fn update(&mut self, frame: Option<&OwnedFrame>) -> Result<()> {
        let Some(frame) = frame else {
            self.detection = None;
            return Ok(());
        };
        let frame_id = frame.meta.id;
        if self.last_frame_id == Some(frame_id) {
            return Ok(());
        }

        self.last_frame_id = Some(frame_id);
        self.detection = self
            .detector
            .process(frame.as_frame().view(), tron_api::NoContext)?;
        Ok(())
    }

    fn detection(&self) -> Option<&CheckerboardDetection> {
        self.detection.as_ref()
    }

    fn clear(&mut self) {
        self.detection = None;
    }
}
