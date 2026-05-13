use anyhow::Result;
use tron_api::{Frame, PixelFormat, Renderer, Size};
use wgpu::util::DeviceExt;

mod surface;

pub use surface::{WgpuSurfaceContext, WgpuSurfaceFrame};

#[derive(Clone, Copy, Debug)]
pub struct NdcRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl NdcRect {
    pub const FULL: Self = Self {
        x0: -1.0,
        y0: -1.0,
        x1: 1.0,
        y1: 1.0,
    };
    pub const LEFT: Self = Self {
        x0: -1.0,
        y0: -1.0,
        x1: 0.0,
        y1: 1.0,
    };
    pub const RIGHT: Self = Self {
        x0: 0.0,
        y0: -1.0,
        x1: 1.0,
        y1: 1.0,
    };
}

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

pub struct WgpuFrameRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    texture: Option<FrameTexture>,
    bgra_scratch: Vec<u8>,
}

pub struct WgpuFrameView<'frame, 'pass> {
    pub device: &'frame wgpu::Device,
    pub queue: &'frame wgpu::Queue,
    pub pass: &'frame mut wgpu::RenderPass<'pass>,
    pub frame: Frame<'frame>,
    pub rect: NdcRect,
    pub target_size: Size,
}

struct FrameTexture {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl WgpuFrameRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
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
                    format: surface_format,
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
            contents: bytemuck::cast_slice(&quad(-1.0, -1.0, 1.0, 1.0, false, false)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            vertex_buffer,
            texture: None,
            bgra_scratch: Vec::new(),
        }
    }
}

impl<'frame, 'pass> Renderer<WgpuFrameView<'frame, 'pass>> for WgpuFrameRenderer {
    fn render(&mut self, view: WgpuFrameView<'frame, 'pass>) -> Result<()> {
        let frame = view.frame;
        self.ensure_texture(view.device, frame.meta.size);
        self.update_vertices(
            view.queue,
            frame.meta.size,
            view.rect,
            view.target_size,
            frame.buffer.is_horizontally_mirrored(),
            frame.buffer.is_vertically_mirrored(),
        );

        let (data, stride) = match frame.format {
            PixelFormat::Bgra8 => {
                anyhow::ensure!(
                    frame.buffer.stride == frame.meta.size.width as usize * 4,
                    "WgpuFrameRenderer requires tightly packed BGRA8 frames"
                );
                (frame.buffer.data, frame.buffer.stride)
            }
            PixelFormat::Gray8 => {
                anyhow::ensure!(
                    frame.buffer.stride == frame.meta.size.width as usize,
                    "WgpuFrameRenderer requires tightly packed Gray8 frames"
                );
                let pixel_count = frame.meta.size.width as usize * frame.meta.size.height as usize;
                self.bgra_scratch.resize(pixel_count * 4, 255);
                for (i, gray) in frame
                    .buffer
                    .data
                    .iter()
                    .take(pixel_count)
                    .copied()
                    .enumerate()
                {
                    let offset = i * 4;
                    self.bgra_scratch[offset] = gray;
                    self.bgra_scratch[offset + 1] = gray;
                    self.bgra_scratch[offset + 2] = gray;
                    self.bgra_scratch[offset + 3] = 255;
                }
                (&self.bgra_scratch[..], frame.meta.size.width as usize * 4)
            }
            PixelFormat::Yuyv422 => {
                anyhow::bail!("WgpuFrameRenderer does not support YUYV422 yet")
            }
        };

        let texture = self.texture.as_ref().expect("texture was just created");
        view.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(stride as u32),
                rows_per_image: Some(frame.meta.size.height),
            },
            wgpu::Extent3d {
                width: frame.meta.size.width,
                height: frame.meta.size.height,
                depth_or_array_layers: 1,
            },
        );

        view.pass.set_pipeline(&self.pipeline);
        view.pass.set_bind_group(0, &texture.bind_group, &[]);
        view.pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        view.pass.draw(0..6, 0..1);
        Ok(())
    }
}

impl WgpuFrameRenderer {
    fn ensure_texture(&mut self, device: &wgpu::Device, size: Size) {
        let needs_texture = self
            .texture
            .as_ref()
            .map(|texture| texture.width != size.width || texture.height != size.height)
            .unwrap_or(true);
        if !needs_texture {
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tron-frame-texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            width: size.width,
            height: size.height,
        });
    }

    fn update_vertices(
        &self,
        queue: &wgpu::Queue,
        frame_size: Size,
        rect: NdcRect,
        target_size: Size,
        flip_x: bool,
        flip_y: bool,
    ) {
        let [x0, y0, x1, y1] = letterbox(frame_size, rect, target_size);
        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&quad(x0, y0, x1, y1, flip_x, flip_y)),
        );
    }
}

pub fn project_frame_point(point: [f32; 2], frame: Size, rect: NdcRect, target: Size) -> [f32; 2] {
    let [x0, y0, x1, y1] = letterbox(frame, rect, target);
    let fx = point[0] / frame.width.max(1) as f32;
    let fy = point[1] / frame.height.max(1) as f32;
    [lerp(x0, x1, fx), lerp(y1, y0, fy)]
}

pub fn letterbox(frame: Size, rect: NdcRect, target: Size) -> [f32; 4] {
    let rect_width_ndc = (rect.x1 - rect.x0).abs();
    let rect_height_ndc = (rect.y1 - rect.y0).abs();
    let rect_pixel_width = target.width as f32 * rect_width_ndc / 2.0;
    let rect_pixel_height = target.height as f32 * rect_height_ndc / 2.0;
    let frame_aspect = frame.width as f32 / frame.height.max(1) as f32;
    let rect_aspect = rect_pixel_width / rect_pixel_height.max(1.0);

    if rect_aspect > frame_aspect {
        let width_ndc = rect_width_ndc * frame_aspect / rect_aspect;
        let cx = (rect.x0 + rect.x1) * 0.5;
        [cx - width_ndc * 0.5, rect.y0, cx + width_ndc * 0.5, rect.y1]
    } else {
        let height_ndc = rect_height_ndc * rect_aspect / frame_aspect;
        let cy = (rect.y0 + rect.y1) * 0.5;
        [
            rect.x0,
            cy - height_ndc * 0.5,
            rect.x1,
            cy + height_ndc * 0.5,
        ]
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn quad(x0: f32, y0: f32, x1: f32, y1: f32, flip_x: bool, flip_y: bool) -> [Vertex; 6] {
    let [u0, u1] = if flip_x { [1.0, 0.0] } else { [0.0, 1.0] };
    let [v0, v1] = if flip_y { [1.0, 0.0] } else { [0.0, 1.0] };
    [
        Vertex {
            position: [x0, y1],
            uv: [u0, v0],
        },
        Vertex {
            position: [x0, y0],
            uv: [u0, v1],
        },
        Vertex {
            position: [x1, y0],
            uv: [u1, v1],
        },
        Vertex {
            position: [x0, y1],
            uv: [u0, v0],
        },
        Vertex {
            position: [x1, y0],
            uv: [u1, v1],
        },
        Vertex {
            position: [x1, y1],
            uv: [u1, v0],
        },
    ]
}
