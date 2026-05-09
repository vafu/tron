use crate::pipeline::PlaygroundMetadata;
use anyhow::{Context, Result};
use tron_api::{Frame, FrameSize, PixelFormat, Presenter};
use tron_core::present::wgpu::{NdcRect, WgpuFramePresenter, WgpuFrameView};

pub struct PlaygroundView<'a> {
    pub rgb: Option<Frame<'a>>,
    pub ir: Option<Frame<'a>>,
    pub metadata: PlaygroundMetadata,
}

pub struct PlaygroundPresenter {
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
    pub async fn new(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        size: FrameSize,
    ) -> Result<Self> {
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

    pub fn resize(&mut self, size: FrameSize) {
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
        let _ = view.metadata;
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
