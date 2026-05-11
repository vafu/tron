use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tron::latest::LatestFrameSource;
use tron_api::{CheckerboardDetection, CheckerboardSpec, OwnedFrame, Presenter, Processor, Size};
use tron_core::calib::checkerboard::{OpenCvCheckerboardConfig, OpenCvCheckerboardDetector};
use tron_core::view::IntoView;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use crate::latency::{CalibrationLatencyLog, CalibrationLatencySample};
use crate::presenter::{CalibrationPresenter, CalibrationView};

pub fn run(
    rgb: LatestFrameSource,
    ir: LatestFrameSource,
    checkerboard: CheckerboardSpec,
    checkerboard_detect_interval: Duration,
) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(rgb, ir, checkerboard, checkerboard_detect_interval);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp {
    rgb: LatestFrameSource,
    ir: LatestFrameSource,
    rgb_checkerboard: CheckerboardDetectionState,
    ir_checkerboard: CheckerboardDetectionState,
    window_id: Option<WindowId>,
    window: Option<Arc<winit::window::Window>>,
    presenter: Option<CalibrationPresenter>,
    latency: CalibrationLatencyLog,
    result: Result<()>,
}

impl WindowApp {
    fn new(
        rgb: LatestFrameSource,
        ir: LatestFrameSource,
        checkerboard: CheckerboardSpec,
        checkerboard_detect_interval: Duration,
    ) -> Self {
        let config = OpenCvCheckerboardConfig::new(checkerboard);
        Self {
            rgb,
            ir,
            rgb_checkerboard: CheckerboardDetectionState::new(config, checkerboard_detect_interval),
            ir_checkerboard: CheckerboardDetectionState::new(config, checkerboard_detect_interval),
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
}

impl ApplicationHandler for WindowApp {
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
        let Some(presenter) = self.presenter.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => presenter.resize(Size {
                width: size.width,
                height: size.height,
            }),
            WindowEvent::RedrawRequested => {
                let redraw_start = Instant::now();
                let latest_start = Instant::now();
                let rgb = match self.rgb.next_frame() {
                    Ok(frame) => frame,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };
                let ir = match self.ir.next_frame() {
                    Ok(frame) => frame,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };
                let latest = latest_start.elapsed();

                let now = Instant::now();
                let rgb_detect_start = Instant::now();
                if let Err(err) = self.rgb_checkerboard.update(rgb.as_deref(), now) {
                    self.set_error(event_loop, err);
                    return;
                }
                let rgb_detect = rgb_detect_start.elapsed();

                let ir_detect_start = Instant::now();
                if let Err(err) = self.ir_checkerboard.update(ir.as_deref(), now) {
                    self.set_error(event_loop, err);
                    return;
                }
                let ir_detect = ir_detect_start.elapsed();

                let rgb_checkerboard = self.rgb_checkerboard.detection();
                let ir_checkerboard = self.ir_checkerboard.detection();

                let present_start = Instant::now();
                if let Err(err) = presenter.present(CalibrationView {
                    rgb: rgb.as_deref().map(|frame| frame.as_frame()),
                    ir: ir.as_deref().map(|frame| frame.as_frame()),
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
                    rgb: rgb.as_deref(),
                    ir: ir.as_deref(),
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
    interval: Duration,
    last_frame_id: Option<u64>,
    last_run: Option<Instant>,
    detection: Option<CheckerboardDetection>,
}

impl CheckerboardDetectionState {
    fn new(config: OpenCvCheckerboardConfig, interval: Duration) -> Self {
        Self {
            detector: OpenCvCheckerboardDetector::new(config),
            interval,
            last_frame_id: None,
            last_run: None,
            detection: None,
        }
    }

    fn update(&mut self, frame: Option<&OwnedFrame>, now: Instant) -> Result<()> {
        let Some(frame) = frame else {
            self.detection = None;
            return Ok(());
        };
        let frame_id = frame.meta.id;
        if self.last_frame_id == Some(frame_id) {
            return Ok(());
        }
        if self
            .last_run
            .map(|last_run| now.saturating_duration_since(last_run) < self.interval)
            .unwrap_or(false)
        {
            return Ok(());
        }

        self.last_frame_id = Some(frame_id);
        self.last_run = Some(now);
        self.detection = self
            .detector
            .process(frame.as_frame().view(), tron_api::NoContext)?;
        Ok(())
    }

    fn detection(&self) -> Option<&CheckerboardDetection> {
        self.detection.as_ref()
    }
}
