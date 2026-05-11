use crate::camera_roi::CameraRoiDriver;
use crate::pipeline::{PlaygroundInput, PlaygroundPipeline, PlaygroundPipelineConfig};
use crate::renderer::{PlaygroundRenderer, PlaygroundView};
use anyhow::{Context, Result};
use std::sync::Arc;
use tron::latest::LatestFrameSource;
use tron_api::{Renderer, Size};
use tron_core::render::http::HttpMetadataRenderer;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

pub fn run(
    rgb: LatestFrameSource,
    ir: LatestFrameSource,
    metadata: Option<HttpMetadataRenderer>,
    camera_roi: Option<CameraRoiDriver>,
    pipeline_config: PlaygroundPipelineConfig,
) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(rgb, ir, metadata, camera_roi, pipeline_config)?;
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp {
    rgb: LatestFrameSource,
    ir: LatestFrameSource,
    window_id: Option<WindowId>,
    window: Option<Arc<winit::window::Window>>,
    renderer: Option<PlaygroundRenderer>,
    metadata: Option<HttpMetadataRenderer>,
    camera_roi: Option<CameraRoiDriver>,
    pipeline: PlaygroundPipeline,
    result: Result<()>,
}

impl WindowApp {
    fn new(
        rgb: LatestFrameSource,
        ir: LatestFrameSource,
        metadata: Option<HttpMetadataRenderer>,
        camera_roi: Option<CameraRoiDriver>,
        pipeline_config: PlaygroundPipelineConfig,
    ) -> Result<Self> {
        Ok(Self {
            rgb,
            ir,
            window_id: None,
            window: None,
            renderer: None,
            metadata,
            camera_roi,
            pipeline: PlaygroundPipeline::new(pipeline_config)?,
            result: Ok(()),
        })
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }
}

impl ApplicationHandler for WindowApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let attrs = WindowAttributes::default().with_title("tron playground");
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.set_error(event_loop, anyhow::Error::new(err).context("create window"));
                return;
            }
        };
        self.window_id = Some(window.id());
        let size = window.inner_size();
        match pollster::block_on(PlaygroundRenderer::new(
            window.clone(),
            Size {
                width: size.width,
                height: size.height,
            },
        )) {
            Ok(renderer) => {
                self.window = Some(window);
                self.renderer = Some(renderer);
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
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(Size {
                width: size.width,
                height: size.height,
            }),
            WindowEvent::RedrawRequested => {
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
                let output = match self.pipeline.process(PlaygroundInput { rgb, ir }) {
                    Ok(output) => output,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };
                let mut camera_roi_rect = None;
                if let Some(camera_roi) = self.camera_roi.as_mut()
                    && let Some(ir_diff) = output.ir_diff
                {
                    if let Err(err) = camera_roi.update(
                        output.exposure_roi,
                        output.roi.map(|roi| roi.rect),
                        ir_diff.meta.size,
                    ) {
                        self.set_error(event_loop, err);
                        return;
                    }
                    camera_roi_rect = camera_roi.current_rect();
                }
                match renderer.render(PlaygroundView {
                    rgb: output.rgb.as_ref().map(|frame| frame.as_frame()),
                    depth_cue: output.depth_cue,
                    ir_diff: output.ir_diff,
                    roi: output.roi,
                    rgb_roi: output.rgb_roi,
                    camera_roi: camera_roi_rect,
                    metadata: output.metadata,
                }) {
                    Ok(()) => {}
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                }
                if let Some(metadata) = self.metadata.as_mut()
                    && let Err(err) = metadata.render(output.metadata)
                {
                    self.set_error(event_loop, err);
                    return;
                }
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
