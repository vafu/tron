use anyhow::{Context, Result};
use std::sync::Arc;
use tron_api::{FrameStats, FrameViewModel, PixelFormat, Presenter};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
}

const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
};

pub struct WgpuPresenter {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    texture: Option<FrameTexture>,
}

struct FrameTexture {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl WgpuPresenter {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        anyhow::ensure!(
            size.width > 0 && size.height > 0,
            "window surface cannot be initialized at zero size"
        );

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
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
                    label: Some("tron-wgpu-device"),
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

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tron-frame-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tron-frame-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tron-frame-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[VERTEX_LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tron-frame-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tron-frame-vertices"),
            contents: bytemuck::cast_slice(&quad(-1.0, -1.0, 1.0, 1.0)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            bind_group_layout,
            sampler,
            vertex_buffer,
            texture: None,
        })
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        if let Some(texture) = &self.texture {
            self.update_vertices(texture.width, texture.height);
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn ensure_texture(&mut self, width: u32, height: u32) {
        let needs_texture = self
            .texture
            .as_ref()
            .map(|texture| texture.width != width || texture.height != height)
            .unwrap_or(true);
        if !needs_texture {
            return;
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tron-frame-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tron-frame-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.texture = Some(FrameTexture {
            texture,
            bind_group,
            width,
            height,
        });
        self.update_vertices(width, height);
    }

    fn update_vertices(&self, frame_width: u32, frame_height: u32) {
        let (x0, y0, x1, y1) = letterbox(frame_width, frame_height, self.size);
        self.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&quad(x0, y0, x1, y1)),
        );
    }
}

impl<'a> Presenter<FrameViewModel<'a, FrameStats>> for WgpuPresenter {
    fn present(&mut self, view: FrameViewModel<'a, FrameStats>) -> Result<()> {
        let Some(named) = view.frames.first() else {
            return Ok(());
        };
        let frame = named.frame;
        anyhow::ensure!(
            frame.format == PixelFormat::Bgra8,
            "WgpuPresenter currently supports only BGRA8 frames, got {:?}",
            frame.format
        );
        anyhow::ensure!(
            frame.stride == frame.meta.size.width as usize * 4,
            "WgpuPresenter requires tightly packed BGRA8 frames"
        );

        self.ensure_texture(frame.meta.size.width, frame.meta.size.height);
        let texture = self.texture.as_ref().expect("texture was just created");
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            frame.data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(frame.stride as u32),
                rows_per_image: Some(frame.meta.size.height),
            },
            wgpu::Extent3d {
                width: frame.meta.size.width,
                height: frame.meta.size.height,
                depth_or_array_layers: 1,
            },
        );

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
                label: Some("tron-frame-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tron-frame-render-pass"),
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &texture.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        surface_frame.present();
        Ok(())
    }
}

fn letterbox(
    frame_width: u32,
    frame_height: u32,
    window: PhysicalSize<u32>,
) -> (f32, f32, f32, f32) {
    let frame_aspect = frame_width as f32 / frame_height.max(1) as f32;
    let window_aspect = window.width as f32 / window.height.max(1) as f32;
    if window_aspect > frame_aspect {
        let width = frame_aspect / window_aspect;
        (-width, -1.0, width, 1.0)
    } else {
        let height = window_aspect / frame_aspect;
        (-1.0, -height, 1.0, height)
    }
}

fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> [Vertex; 6] {
    [
        Vertex {
            position: [x0, y1],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [x0, y0],
            uv: [0.0, 1.0],
        },
        Vertex {
            position: [x1, y0],
            uv: [1.0, 1.0],
        },
        Vertex {
            position: [x0, y1],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [x1, y0],
            uv: [1.0, 1.0],
        },
        Vertex {
            position: [x1, y1],
            uv: [1.0, 0.0],
        },
    ]
}
