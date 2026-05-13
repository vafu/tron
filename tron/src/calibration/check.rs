use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use tron_api::ViewRow;
use tron_api::{Frame, FrameMeta, FrameSource, PixelFormat, ProjectionMapSource, Renderer, Size};
use tron_core::StereoFrameSource;
use tron_core::projection::FrameProjectionMap;
use tron_core::render::wgpu::{NdcRect, WgpuFrameRenderer, WgpuFrameView, WgpuSurfaceContext};
use tron_core::transform::ProjectedFrameSource;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

pub struct CalibrationCheckConfig {
    pub max_sync_delta_us: u64,
}

pub fn run<R, I, M>(rgb: R, ir: I, map_source: M, config: CalibrationCheckConfig) -> Result<()>
where
    R: FrameSource + Send + 'static,
    I: FrameSource + Send + 'static,
    M: ProjectionMapSource<Map = FrameProjectionMap> + Send + 'static,
{
    let ir = ProjectedFrameSource::new(ir, map_source)?;
    run_projected(rgb, ir, config)
}

fn run_projected<R, I>(rgb: R, ir: I, config: CalibrationCheckConfig) -> Result<()>
where
    R: FrameSource + Send + 'static,
    I: FrameSource + Send + 'static,
{
    let frames = StereoFrameSource::new(rgb, ir, config.max_sync_delta_us);
    let latest = LatestCompositeFrame::default();
    let producer = latest.clone();
    tokio::spawn(async move {
        produce_composite_frames(frames, producer).await;
    });

    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = CheckApp::new(latest);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

#[derive(Clone, Default)]
struct LatestCompositeFrame {
    state: Arc<Mutex<LatestCompositeState>>,
}

#[derive(Default)]
struct LatestCompositeState {
    frame: Option<Arc<CompositeFrame>>,
    error: Option<anyhow::Error>,
}

impl LatestCompositeFrame {
    fn set_frame(&self, frame: CompositeFrame) -> CompositeFrame {
        let Ok(mut state) = self.state.lock() else {
            return frame;
        };

        let previous = state.frame.replace(Arc::new(frame));
        if let Some(previous) = previous {
            Arc::try_unwrap(previous).unwrap_or_default()
        } else {
            CompositeFrame::default()
        }
    }

    fn set_error(&self, err: anyhow::Error) {
        if let Ok(mut state) = self.state.lock() {
            state.error = Some(err);
        }
    }

    fn take_error(&self) -> Option<anyhow::Error> {
        self.state
            .lock()
            .ok()
            .and_then(|mut state| state.error.take())
    }

    fn latest(&self) -> Option<Arc<CompositeFrame>> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.frame.as_ref().cloned())
    }
}

async fn produce_composite_frames<R, I>(
    mut frames: StereoFrameSource<R, I>,
    latest: LatestCompositeFrame,
) where
    R: FrameSource + Send,
    I: FrameSource + Send,
{
    let mut composite = CompositeFrame::default();
    loop {
        match frames.next_pair().await {
            Ok(Some(pair)) => match composite.update(&pair.left, &pair.right) {
                Ok(()) => composite = latest.set_frame(composite),
                Err(err) => {
                    latest.set_error(err);
                    return;
                }
            },
            Ok(None) => {}
            Err(err) => {
                latest.set_error(err);
                return;
            }
        }
        tokio::task::yield_now().await;
    }
}

struct CheckApp {
    latest: LatestCompositeFrame,
    rendered_id: Option<u64>,
    window_id: Option<WindowId>,
    renderer: Option<CheckRenderer>,
    window: Option<Arc<winit::window::Window>>,
    result: Result<()>,
}

impl CheckApp {
    fn new(latest: LatestCompositeFrame) -> Self {
        Self {
            latest,
            rendered_id: None,
            window_id: None,
            renderer: None,
            window: None,
            result: Ok(()),
        }
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }
}

impl ApplicationHandler for CheckApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let attrs = WindowAttributes::default().with_title("tron calibration check");
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.set_error(event_loop, anyhow::Error::new(err).context("create window"));
                return;
            }
        };
        self.window_id = Some(window.id());
        let size = window.inner_size();
        match pollster::block_on(CheckRenderer::new(
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
                    self.rendered_id = None;
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(err) = self.latest.take_error() {
                    self.set_error(event_loop, err);
                    return;
                }

                let Some(frame) = self.latest.latest() else {
                    return;
                };
                let frame_id = frame.id();
                if self.rendered_id == frame_id {
                    return;
                }
                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                if let Err(err) = renderer.render(frame.frame()) {
                    self.set_error(event_loop, err);
                    return;
                }
                self.rendered_id = frame_id;
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

struct CheckRenderer {
    surface: WgpuSurfaceContext,
    frame: WgpuFrameRenderer,
}

impl CheckRenderer {
    async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
        let surface =
            WgpuSurfaceContext::new(target, size, "tron-calibration-check-wgpu-device").await?;
        let format = surface.format();
        Ok(Self {
            frame: WgpuFrameRenderer::new(surface.device(), format),
            surface,
        })
    }

    fn resize(&mut self, size: Size) {
        self.surface.resize(size);
    }
}

impl Renderer<Frame<'_>> for CheckRenderer {
    fn render(&mut self, frame: Frame<'_>) -> Result<()> {
        self.surface.render(
            "tron-calibration-check-render-pass",
            wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.025,
                a: 1.0,
            },
            |surface| {
                let mut pass = surface.pass;
                self.frame.render(WgpuFrameView {
                    device: surface.device,
                    queue: surface.queue,
                    pass: &mut pass,
                    frame,
                    rect: NdcRect::FULL,
                    target_size: surface.size,
                })
            },
        )
    }
}

#[derive(Default)]
struct CompositeFrame {
    meta: Option<FrameMeta>,
    data: Vec<u8>,
}

impl CompositeFrame {
    fn update(&mut self, rgb: &Frame<'_>, ir: &Frame<'_>) -> Result<()> {
        anyhow::ensure!(
            rgb.meta.size == ir.meta.size,
            "RGB frame size {:?} does not match projected IR frame size {:?}",
            rgb.meta.size,
            ir.meta.size,
        );

        let size = rgb.meta.size;
        let pixel_count = size.width as usize * size.height as usize;
        self.data.resize(pixel_count * 4, 255);

        for y in 0..size.height {
            let rgb_row = rgb.row(y)?;
            let ir_row = ir.row(y)?;
            for x in 0..size.width {
                let rgb = bgra_at(rgb_row, rgb.format, x as usize)?;
                let dst = (y as usize * size.width as usize + x as usize) * 4;
                self.data[dst..dst + 4].copy_from_slice(&rgb);

                let ir = gray_at(ir_row, ir.format, x as usize)?;
                blend_ir(&mut self.data[dst..dst + 4], rgb, ir, 0.38);
            }
        }

        self.meta = Some(FrameMeta { size, ..rgb.meta });
        Ok(())
    }

    fn frame(&self) -> Frame<'_> {
        let meta = self.meta.expect("composite frame was not initialized");
        Frame::new(
            meta,
            PixelFormat::Bgra8,
            meta.size.width as usize * 4,
            &self.data,
        )
        .expect("composite frame metadata must match backing buffer")
    }

    fn id(&self) -> Option<u64> {
        self.meta.map(|meta| meta.id)
    }
}

fn bgra_at(row: ViewRow<'_>, format: PixelFormat, x: usize) -> Result<[u8; 4]> {
    match format {
        PixelFormat::Bgra8 => {
            let offset = x * 4;
            Ok([
                row.byte(offset)?,
                row.byte(offset + 1)?,
                row.byte(offset + 2)?,
                row.byte(offset + 3)?,
            ])
        }
        PixelFormat::Gray8 => {
            let value = row.byte(x)?;
            Ok([value, value, value, 255])
        }
        PixelFormat::Yuyv422 => anyhow::bail!("calibration check does not support YUYV422"),
    }
}

fn gray_at(row: ViewRow<'_>, format: PixelFormat, x: usize) -> Result<u8> {
    match format {
        PixelFormat::Gray8 => Ok(row.byte(x)?),
        PixelFormat::Bgra8 => {
            let offset = x * 4;
            Ok(((row.byte(offset)? as u16
                + row.byte(offset + 1)? as u16
                + row.byte(offset + 2)? as u16)
                / 3) as u8)
        }
        PixelFormat::Yuyv422 => anyhow::bail!("calibration check does not support YUYV422"),
    }
}

fn blend_ir(dst: &mut [u8], rgb: [u8; 4], ir: u8, alpha: f32) {
    let base = 1.0 - alpha;
    dst[0] = (rgb[0] as f32 * base + ir as f32 * alpha) as u8;
    dst[1] = (rgb[1] as f32 * base + ir as f32 * alpha) as u8;
    dst[2] = (rgb[2] as f32 * base + ir as f32 * alpha) as u8;
    dst[3] = 255;
}
