use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tron_api::{DepthSource, FrameSource, NoContext, Processor, Renderer, RoiResult, Size};
use tron_core::StereoFrameSource;
use tron_core::projection::{
    CheckerboardDepthProjection, HandProjectionInput, HandProjectionProcessor,
};
use tron_core::roi::mediapipe::{
    MediaPipeHandLandmarkConfig, MediaPipeHandLandmarkInput, MediaPipeHandLandmarkProcessor,
    MediaPipeRoiConfig, MediaPipeRoiProcessor,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use crate::renderer::{CollectorRenderer, CollectorView};

pub fn run<R, I>(
    rgb: R,
    ir: I,
    max_sync_delta_us: u64,
    mediapipe_model: PathBuf,
    mediapipe_config: MediaPipeRoiConfig,
    landmark_model: PathBuf,
    landmark_config: MediaPipeHandLandmarkConfig,
    hand_projection: Option<HandProjectionProcessor<CheckerboardDepthProjection>>,
    roi_depth_source: Option<Box<dyn DepthSource + Send>>,
) -> Result<()>
where
    R: FrameSource + Send,
    I: FrameSource + Send,
{
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let frames = StereoFrameSource::new(rgb, ir, max_sync_delta_us);
    let mut app = WindowApp::new(
        frames,
        mediapipe_model,
        mediapipe_config,
        landmark_model,
        landmark_config,
        hand_projection,
        roi_depth_source,
    )?;
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp<R, I> {
    frames: StereoFrameSource<R, I>,
    mediapipe: MediaPipeRoiProcessor,
    landmarks: MediaPipeHandLandmarkProcessor,
    hand_projection: Option<HandProjectionProcessor<CheckerboardDepthProjection>>,
    roi_depth_source: Option<Box<dyn DepthSource + Send>>,
    rendered_pair_id: Option<(u64, u64)>,
    window_id: Option<WindowId>,
    window: Option<Arc<winit::window::Window>>,
    renderer: Option<CollectorRenderer>,
    result: Result<()>,
}

impl<R, I> WindowApp<R, I>
where
    R: FrameSource + Send,
    I: FrameSource + Send,
{
    fn new(
        frames: StereoFrameSource<R, I>,
        mediapipe_model: PathBuf,
        mediapipe_config: MediaPipeRoiConfig,
        landmark_model: PathBuf,
        landmark_config: MediaPipeHandLandmarkConfig,
        hand_projection: Option<HandProjectionProcessor<CheckerboardDepthProjection>>,
        roi_depth_source: Option<Box<dyn DepthSource + Send>>,
    ) -> Result<Self> {
        Ok(Self {
            frames,
            mediapipe: MediaPipeRoiProcessor::new(mediapipe_model, mediapipe_config)?,
            landmarks: MediaPipeHandLandmarkProcessor::new(landmark_model, landmark_config)?,
            hand_projection,
            roi_depth_source,
            rendered_pair_id: None,
            window_id: None,
            window: None,
            renderer: None,
            result: Ok(()),
        })
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }
}

impl<R, I> ApplicationHandler for WindowApp<R, I>
where
    R: FrameSource + Send,
    I: FrameSource + Send,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let attrs = WindowAttributes::default().with_title("tron collector");
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.set_error(event_loop, anyhow::Error::new(err).context("create window"));
                return;
            }
        };
        self.window_id = Some(window.id());
        let size = window.inner_size();
        match pollster::block_on(CollectorRenderer::new(
            window.clone(),
            Size {
                width: size.width,
                height: size.height,
            },
        )) {
            Ok(renderer) => {
                self.window = Some(window);
                self.renderer = Some(renderer);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
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
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(Size {
                        width: size.width,
                        height: size.height,
                    });
                    self.rendered_pair_id = None;
                }
            }
            WindowEvent::RedrawRequested => {
                let pair = match pollster::block_on(self.frames.next_pair()) {
                    Ok(pair) => pair,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };
                let Some(pair) = pair else {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                    return;
                };
                let rgb = pair.left;
                let ir = pair.right;
                let pair_id = (rgb.meta.id, ir.meta.id);
                if self.rendered_pair_id == Some(pair_id) {
                    return;
                }

                // Per-frame detection: fresh palm detection on every frame for maximum stability.
                let palm_roi: Option<RoiResult> = match self.mediapipe.process(rgb, NoContext) {
                    Ok(roi) => roi,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };
                let landmarks = match self.landmarks.process(
                    MediaPipeHandLandmarkInput {
                        frame: rgb,
                        roi: palm_roi,
                    },
                    NoContext,
                ) {
                    Ok(landmarks) => landmarks,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };

                if let Some(ref l) = landmarks {
                    let valid_count = l.points.iter().filter(|p| p.x.is_finite()).count();
                    tracing::info!("Detected {} valid landmarks", valid_count);
                }

                let landmark_roi = landmarks.as_ref().and_then(|landmarks| {
                    let roi =
                        landmarks.bounding_roi(rgb.meta.size, self.landmarks.config().roi_scale);
                    if let Some(ref r) = roi {
                        tracing::info!("Landmark ROI: {:?}", r.rect);
                    }
                    roi
                });
                let rgb_roi = landmark_roi.or(palm_roi);
                let projected = if let Some(hand_projection) = self.hand_projection.as_mut() {
                    let depth_sample = match self.roi_depth_source.as_mut() {
                        Some(depth_source) => {
                            match pollster::block_on(
                                depth_source.depth_at(rgb.meta.timestamp.received_at),
                            ) {
                                Ok(sample) => sample,
                                Err(err) => {
                                    self.set_error(event_loop, err);
                                    return;
                                }
                            }
                        }
                        None => None,
                    };
                    match hand_projection.process(
                        HandProjectionInput {
                            roi: rgb_roi,
                            landmarks: landmarks.as_ref(),
                            depth_sample,
                            source_size: rgb.meta.size,
                            target_size: ir.meta.size,
                        },
                        NoContext,
                    ) {
                        Ok(projected) => Some(projected),
                        Err(err) => {
                            self.set_error(event_loop, err);
                            return;
                        }
                    }
                } else {
                    None
                };

                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                if let Err(err) = renderer.render(CollectorView {
                    rgb: Some(rgb),
                    ir: Some(ir),
                    rgb_palm_roi: palm_roi,
                    rgb_roi,
                    ir_roi: projected.as_ref().and_then(|projected| projected.roi),
                    rgb_landmarks: landmarks.as_ref(),
                    ir_landmarks: projected
                        .as_ref()
                        .and_then(|projected| projected.landmarks.as_ref()),
                }) {
                    self.set_error(event_loop, err);
                    return;
                }
                self.rendered_pair_id = Some(pair_id);
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
