use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tron_api::{FrameSource, NoContext, Processor, Renderer, RoiResult, Size};
use tron_core::capture::StereoFrameSource;
use tron_core::roi::mediapipe::{MediaPipeRoiConfig, MediaPipeRoiProcessor};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use crate::renderer::{CollectorRenderer, CollectorView};

pub fn run<L, R>(
    frames: StereoFrameSource<L, R>,
    mediapipe_model: PathBuf,
    mediapipe_config: MediaPipeRoiConfig,
) -> Result<()>
where
    L: FrameSource + Send,
    R: FrameSource + Send,
{
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(frames, mediapipe_model, mediapipe_config)?;
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp<L, R> {
    frames: StereoFrameSource<L, R>,
    mediapipe: MediaPipeRoiProcessor,
    rendered_pair_id: Option<(u64, u64)>,
    window_id: Option<WindowId>,
    window: Option<Arc<winit::window::Window>>,
    renderer: Option<CollectorRenderer>,
    result: Result<()>,
}

impl<L, R> WindowApp<L, R>
where
    L: FrameSource + Send,
    R: FrameSource + Send,
{
    fn new(
        frames: StereoFrameSource<L, R>,
        mediapipe_model: PathBuf,
        mediapipe_config: MediaPipeRoiConfig,
    ) -> Result<Self> {
        Ok(Self {
            frames,
            mediapipe: MediaPipeRoiProcessor::new(mediapipe_model, mediapipe_config)?,
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

impl<L, R> ApplicationHandler for WindowApp<L, R>
where
    L: FrameSource + Send,
    R: FrameSource + Send,
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
                let pair_id = (pair.left.meta.id, pair.right.meta.id);
                if self.rendered_pair_id == Some(pair_id) {
                    return;
                }

                let rgb = pair.left;
                let ir = pair.right;
                let rgb_roi: Option<RoiResult> = match self.mediapipe.process(rgb, NoContext) {
                    Ok(roi) => roi,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };

                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                if let Err(err) = renderer.render(CollectorView {
                    rgb: Some(rgb),
                    ir: Some(ir),
                    rgb_roi,
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
