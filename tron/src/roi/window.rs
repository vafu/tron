use crate::overlay::{RoiOverlayPresenter, RoiOverlayView};
use crate::roi::RoiController;
use crate::sweep::RoiSweep;
use crate::uvc_step::UvcStepper;
use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Instant;
use tron_api::{Frame, Presenter, Size};
use tron_core::pipeline::FrameStream;
use tron_core::present::wgpu::{NdcRect, WgpuFramePresenter, WgpuFrameView};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{WindowAttributes, WindowId};

pub fn run(
    stream: Box<dyn FrameStream + 'static>,
    controller: RoiController,
    sweep_speed: f32,
    uvc_stepper: Option<UvcStepper>,
) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(stream, controller, sweep_speed, uvc_stepper);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct WindowApp {
    stream: Box<dyn FrameStream>,
    controller: RoiController,
    window_id: Option<WindowId>,
    window: Option<Arc<winit::window::Window>>,
    presenter: Option<RoiWindowPresenter>,
    latest_size: Option<Size>,
    window_size: Size,
    cursor_position: Option<PhysicalPosition<f64>>,
    dragging_roi: bool,
    sweep: RoiSweep,
    uvc_stepper: Option<UvcStepper>,
    result: Result<()>,
}

impl WindowApp {
    fn new(
        stream: Box<dyn FrameStream>,
        controller: RoiController,
        sweep_speed: f32,
        uvc_stepper: Option<UvcStepper>,
    ) -> Self {
        Self {
            stream,
            controller,
            window_id: None,
            window: None,
            presenter: None,
            latest_size: None,
            window_size: Size {
                width: 1,
                height: 1,
            },
            cursor_position: None,
            dragging_roi: false,
            sweep: RoiSweep::new(sweep_speed),
            uvc_stepper,
            result: Ok(()),
        }
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, event: KeyEvent) {
        if event.state != ElementState::Pressed || event.repeat {
            return;
        }
        let Some(frame_size) = self.latest_size else {
            return;
        };
        let result = match event.physical_key {
            PhysicalKey::Code(KeyCode::ArrowLeft) => self.controller.move_by(-1, 0, frame_size),
            PhysicalKey::Code(KeyCode::ArrowRight) => self.controller.move_by(1, 0, frame_size),
            PhysicalKey::Code(KeyCode::ArrowUp) => self.controller.move_by(0, -1, frame_size),
            PhysicalKey::Code(KeyCode::ArrowDown) => self.controller.move_by(0, 1, frame_size),
            PhysicalKey::Code(KeyCode::Equal) | PhysicalKey::Code(KeyCode::NumpadAdd) => {
                self.controller.resize_by_pixels(1, frame_size)
            }
            PhysicalKey::Code(KeyCode::Minus) | PhysicalKey::Code(KeyCode::NumpadSubtract) => {
                self.controller.resize_by_pixels(-1, frame_size)
            }
            PhysicalKey::Code(KeyCode::Digit1) => self.controller.set_auto(true),
            PhysicalKey::Code(KeyCode::Digit0) => self.controller.set_auto(false),
            PhysicalKey::Code(KeyCode::KeyR) => self.controller.reset(),
            PhysicalKey::Code(KeyCode::F1) => self.controller.set_square_size(1, frame_size),
            PhysicalKey::Code(KeyCode::F2) => self.controller.set_square_size(8, frame_size),
            PhysicalKey::Code(KeyCode::F3) => self.controller.set_square_size(16, frame_size),
            PhysicalKey::Code(KeyCode::F4) => self.controller.set_square_size(24, frame_size),
            PhysicalKey::Code(KeyCode::F5) => self.controller.set_square_size(32, frame_size),
            PhysicalKey::Code(KeyCode::F6) => self.controller.set_square_size(48, frame_size),
            PhysicalKey::Code(KeyCode::F7) => self.controller.set_square_size(64, frame_size),
            PhysicalKey::Code(KeyCode::KeyA) => {
                self.sweep.toggle();
                eprintln!(
                    "roi-sweep: enabled={} speed={:.1}px/s",
                    self.sweep.enabled(),
                    self.sweep.speed()
                );
                Ok(())
            }
            PhysicalKey::Code(KeyCode::BracketRight) => {
                self.sweep.adjust_speed(20.0);
                eprintln!("roi-sweep: speed={:.1}px/s", self.sweep.speed());
                Ok(())
            }
            PhysicalKey::Code(KeyCode::BracketLeft) => {
                self.sweep.adjust_speed(-20.0);
                eprintln!("roi-sweep: speed={:.1}px/s", self.sweep.speed());
                Ok(())
            }
            PhysicalKey::Code(KeyCode::Period) => {
                self.sweep.adjust_speed(1.0);
                eprintln!("roi-sweep: speed={:.1}px/s", self.sweep.speed());
                Ok(())
            }
            PhysicalKey::Code(KeyCode::Comma) => {
                self.sweep.adjust_speed(-1.0);
                eprintln!("roi-sweep: speed={:.1}px/s", self.sweep.speed());
                Ok(())
            }
            PhysicalKey::Code(KeyCode::Enter) | PhysicalKey::Code(KeyCode::NumpadEnter) => {
                if let Some(stepper) = self.uvc_stepper.as_mut() {
                    stepper.step()
                } else {
                    eprintln!("uvc-step: disabled; restart roi-control with --uvc-step");
                    Ok(())
                }
            }
            _ => Ok(()),
        };
        if let Err(err) = result {
            self.set_error(event_loop, err);
            return;
        }
        eprintln!("roi-control: roi={:?}", self.controller.rect());
    }

    fn handle_mouse_move(&mut self, event_loop: &ActiveEventLoop, position: PhysicalPosition<f64>) {
        self.cursor_position = Some(position);
        if !self.dragging_roi {
            return;
        }
        self.center_roi_at_cursor(event_loop);
    }

    fn handle_mouse_button(
        &mut self,
        event_loop: &ActiveEventLoop,
        state: ElementState,
        button: MouseButton,
    ) {
        if button != MouseButton::Left {
            return;
        }
        self.dragging_roi = state == ElementState::Pressed;
        if self.dragging_roi {
            self.center_roi_at_cursor(event_loop);
        }
    }

    fn handle_mouse_wheel(&mut self, event_loop: &ActiveEventLoop, delta: MouseScrollDelta) {
        let Some(frame_size) = self.latest_size else {
            return;
        };
        let direction = match delta {
            MouseScrollDelta::LineDelta(_, y) if y > 0.0 => 1,
            MouseScrollDelta::LineDelta(_, y) if y < 0.0 => -1,
            MouseScrollDelta::PixelDelta(position) if position.y > 0.0 => 1,
            MouseScrollDelta::PixelDelta(position) if position.y < 0.0 => -1,
            _ => return,
        };
        if let Err(err) = self.controller.resize_by_step(direction, frame_size) {
            self.set_error(event_loop, err);
            return;
        }
        eprintln!("roi-control: roi={:?}", self.controller.rect());
    }

    fn center_roi_at_cursor(&mut self, event_loop: &ActiveEventLoop) {
        let Some(frame_size) = self.latest_size else {
            return;
        };
        let Some(cursor_position) = self.cursor_position else {
            return;
        };
        let Some((x, y)) = window_to_frame(cursor_position, frame_size, self.window_size) else {
            return;
        };
        if let Err(err) = self.controller.center_on(x, y, frame_size) {
            self.set_error(event_loop, err);
            return;
        }
        eprintln!("roi-control: roi={:?}", self.controller.rect());
    }
}

impl ApplicationHandler for WindowApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.presenter.is_some() {
            return;
        }

        let attrs = WindowAttributes::default().with_title("tron roi control");
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.set_error(event_loop, anyhow::Error::new(err).context("create window"));
                return;
            }
        };
        self.window_id = Some(window.id());
        let size = window.inner_size();
        self.window_size = Size {
            width: size.width.max(1),
            height: size.height.max(1),
        };
        match pollster::block_on(RoiWindowPresenter::new(
            window.clone(),
            Size {
                width: size.width,
                height: size.height,
            },
        )) {
            Ok(presenter) => {
                if let Err(err) = self.controller.apply() {
                    self.set_error(event_loop, err);
                    return;
                }
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
            WindowEvent::KeyboardInput { event, .. } => self.handle_key(event_loop, event),
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_mouse_move(event_loop, position)
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.handle_mouse_button(event_loop, state, button)
            }
            WindowEvent::MouseWheel { delta, .. } => self.handle_mouse_wheel(event_loop, delta),
            WindowEvent::Resized(size) => {
                self.window_size = Size {
                    width: size.width.max(1),
                    height: size.height.max(1),
                };
                presenter.resize(Size {
                    width: size.width,
                    height: size.height,
                });
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                match self.stream.next_frame().and_then(|frame| match frame {
                    Some(frame) => {
                        self.latest_size = Some(frame.meta.size);
                        self.sweep
                            .update(&mut self.controller, frame.meta.size, now)?;
                        self.sweep
                            .maybe_log(frame, self.controller.rect(), Instant::now());
                        presenter.present(RoiView {
                            frame,
                            roi: self.controller.rect(),
                        })
                    }
                    None => Ok(()),
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

fn window_to_frame(
    position: PhysicalPosition<f64>,
    frame_size: Size,
    window_size: Size,
) -> Option<(u32, u32)> {
    let frame_aspect = frame_size.width as f64 / frame_size.height.max(1) as f64;
    let window_aspect = window_size.width as f64 / window_size.height.max(1) as f64;

    let (display_x, display_y, display_width, display_height) = if window_aspect > frame_aspect {
        let width = window_size.height as f64 * frame_aspect;
        (
            (window_size.width as f64 - width) * 0.5,
            0.0,
            width,
            window_size.height as f64,
        )
    } else {
        let height = window_size.width as f64 / frame_aspect;
        (
            0.0,
            (window_size.height as f64 - height) * 0.5,
            window_size.width as f64,
            height,
        )
    };

    if position.x < display_x
        || position.y < display_y
        || position.x > display_x + display_width
        || position.y > display_y + display_height
    {
        return None;
    }

    let x = ((position.x - display_x) / display_width * frame_size.width as f64)
        .floor()
        .clamp(0.0, frame_size.width.saturating_sub(1) as f64) as u32;
    let y = ((position.y - display_y) / display_height * frame_size.height as f64)
        .floor()
        .clamp(0.0, frame_size.height.saturating_sub(1) as f64) as u32;
    Some((x, y))
}

struct RoiView<'a> {
    frame: Frame<'a>,
    roi: crate::roi::RoiRect,
}

struct RoiWindowPresenter {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: Size,
    frame: WgpuFramePresenter,
    overlay: RoiOverlayPresenter,
}

impl RoiWindowPresenter {
    async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
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
                    label: Some("tron-roi-control-wgpu-device"),
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

        Ok(Self {
            surface,
            frame: WgpuFramePresenter::new(&device, format),
            overlay: RoiOverlayPresenter::new(&device, format),
            device,
            queue,
            config,
            size,
        })
    }

    fn resize(&mut self, size: Size) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }
}

impl<'a> Presenter<RoiView<'a>> for RoiWindowPresenter {
    fn present(&mut self, view: RoiView<'a>) -> Result<()> {
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
                label: Some("tron-roi-control-frame-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tron-roi-control-render-pass"),
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
            self.frame.present(WgpuFrameView {
                device: &self.device,
                queue: &self.queue,
                pass: &mut pass,
                frame: view.frame,
                rect: NdcRect::FULL,
                target_size: self.size,
            })?;
            self.overlay.present(RoiOverlayView {
                queue: &self.queue,
                pass: &mut pass,
                roi: view.roi,
                frame_size: view.frame.meta.size,
                rect: NdcRect::FULL,
                target_size: self.size,
            })?;
        }
        self.queue.submit([encoder.finish()]);
        surface_frame.present();
        Ok(())
    }
}
