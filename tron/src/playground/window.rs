use crate::latency::{LatencyProbe, LatencySample, age_us, camera_delta_us};
use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tron_api::{Frame, FrameSize, PixelFormat, Presenter};
use tron_core::pipeline::FrameStream;
use tron_core::present::wgpu::{NdcRect, WgpuFramePresenter, WgpuFrameView};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

pub fn run(
    rgb_stream: Box<dyn FrameStream + 'static>,
    ir_stream: Box<dyn FrameStream + 'static>,
) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(rgb_stream, ir_stream);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp {
    rgb_stream: Box<dyn FrameStream>,
    ir_stream: Box<dyn FrameStream>,
    window_id: Option<WindowId>,
    window: Option<Arc<winit::window::Window>>,
    presenter: Option<PlaygroundPresenter>,
    probe: LatencyProbe,
    result: Result<()>,
}

impl WindowApp {
    fn new(rgb_stream: Box<dyn FrameStream>, ir_stream: Box<dyn FrameStream>) -> Self {
        Self {
            rgb_stream,
            ir_stream,
            window_id: None,
            window: None,
            presenter: None,
            probe: LatencyProbe::new(Duration::from_secs(1)),
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
        match pollster::block_on(PlaygroundPresenter::new(
            window.clone(),
            FrameSize {
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
            WindowEvent::Resized(size) => presenter.resize(FrameSize {
                width: size.width,
                height: size.height,
            }),
            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();
                match self.rgb_stream.next_frame().and_then(|rgb| {
                    let after_rgb = Instant::now();
                    if rgb.format != PixelFormat::Bgra8 {
                        anyhow::bail!(
                            "RGB feed expects BGRA8 frames; set --format mjpg --decode-format bgra8"
                        );
                    }
                    let rgb_age_after_acquire = age_us(rgb, after_rgb);
                    let ir = self.ir_stream.next_frame()?;
                    let after_ir = Instant::now();
                    let rgb_age_before_present = age_us(rgb, after_ir);
                    let ir_age_after_acquire = age_us(ir, after_ir);
                    let camera_delta_us = camera_delta_us(rgb, ir);
                    presenter
                        .present(PlaygroundView {
                            rgb: Some(rgb),
                            ir: Some(ir),
                        })
                        .inspect(|()| {
                            self.probe.record(LatencySample {
                                rgb_wait: after_rgb.duration_since(frame_start),
                                ir_wait: after_ir.duration_since(after_rgb),
                                present: Instant::now().duration_since(after_ir),
                                rgb_age_after_acquire,
                                rgb_age_before_present,
                                ir_age_after_acquire,
                                camera_delta_us,
                            });
                        })
                }) {
                    Ok(()) => {}
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
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

struct PlaygroundView<'a> {
    rgb: Option<Frame<'a>>,
    ir: Option<Frame<'a>>,
}

struct PlaygroundPresenter {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: FrameSize,
    rgb_top: WgpuFramePresenter,
    rgb_bottom: WgpuFramePresenter,
    ir: WgpuFramePresenter,
}

impl PlaygroundPresenter {
    async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: FrameSize) -> Result<Self> {
        anyhow::ensure!(
            size.width > 0 && size.height > 0,
            "surface cannot be initialized at zero size"
        );

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(target)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .context("request wgpu adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("tron-playground-wgpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .context("request wgpu device")?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let rgb_top = WgpuFramePresenter::new(&device, format);
        let rgb_bottom = WgpuFramePresenter::new(&device, format);
        let ir = WgpuFramePresenter::new(&device, format);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            rgb_top,
            rgb_bottom,
            ir,
        })
    }

    fn resize(&mut self, size: FrameSize) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }
}

impl<'a> Presenter<PlaygroundView<'a>> for PlaygroundPresenter {
    fn present(&mut self, view: PlaygroundView<'a>) -> Result<()> {
        if let Some(rgb) = view.rgb
            && rgb.format != PixelFormat::Bgra8
        {
            anyhow::bail!("RGB feed expects BGRA8 frames");
        }
        let surface_frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                self.surface
                    .get_current_texture()
                    .context("get surface texture after reconfigure")?
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(()),
            Err(err) => return Err(err).context("get surface texture"),
        };
        let surface_view = surface_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tron-playground-frame-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tron-playground-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.025,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some(rgb) = view.rgb {
                self.rgb_top.present(WgpuFrameView {
                    device: &self.device,
                    queue: &self.queue,
                    pass: &mut pass,
                    frame: rgb,
                    rect: NdcRect {
                        x0: -1.0,
                        y0: 0.0,
                        x1: 0.0,
                        y1: 1.0,
                    },
                    target_size: self.size,
                })?;
                self.rgb_bottom.present(WgpuFrameView {
                    device: &self.device,
                    queue: &self.queue,
                    pass: &mut pass,
                    frame: rgb,
                    rect: NdcRect {
                        x0: -1.0,
                        y0: -1.0,
                        x1: 1.0,
                        y1: 0.0,
                    },
                    target_size: self.size,
                })?;
            }
            if let Some(ir) = view.ir {
                self.ir.present(WgpuFrameView {
                    device: &self.device,
                    queue: &self.queue,
                    pass: &mut pass,
                    frame: ir,
                    rect: NdcRect {
                        x0: 0.0,
                        y0: 0.0,
                        x1: 1.0,
                        y1: 1.0,
                    },
                    target_size: self.size,
                })?;
            }
        }
        self.queue.submit([encoder.finish()]);
        surface_frame.present();
        Ok(())
    }
}
