use crate::types::Image;
use winit::dpi::PhysicalSize;

use super::skeleton::letterbox_rect;

pub(super) fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    fs_entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fs_entry),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs",
            buffers: &[VERTEX_LAYOUT],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: fs_entry,
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
    })
}

/// Expand a single-channel mask into a pre-sized RGBA8 buffer (gray, opaque).
pub(super) fn expand_to_rgba(src: &Image, dst: &mut [u8]) {
    for (i, g) in src.grey_iter().enumerate() {
        let o = i * 4;
        dst[o] = g;
        dst[o + 1] = g;
        dst[o + 2] = g;
        dst[o + 3] = 255;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

const VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
};

fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> [Vertex; 6] {
    [
        Vertex {
            pos: [x0, y1],
            uv: [0.0, 0.0],
        },
        Vertex {
            pos: [x0, y0],
            uv: [0.0, 1.0],
        },
        Vertex {
            pos: [x1, y0],
            uv: [1.0, 1.0],
        },
        Vertex {
            pos: [x0, y1],
            uv: [0.0, 0.0],
        },
        Vertex {
            pos: [x1, y0],
            uv: [1.0, 1.0],
        },
        Vertex {
            pos: [x1, y1],
            uv: [1.0, 0.0],
        },
    ]
}

pub(super) struct TexQuad {
    texture: wgpu::Texture,
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) vbuf: wgpu::Buffer,
    pub(super) w: u32,
    pub(super) h: u32,
    rect: (f32, f32, f32, f32),
    pub(super) last_seq: u64,
}

impl TexQuad {
    pub(super) fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        w: u32,
        h: u32,
        rect: (f32, f32, f32, f32),
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (std::mem::size_of::<Vertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            texture,
            bind_group,
            vbuf,
            w,
            h,
            rect,
            last_seq: 0,
        }
    }

    pub(super) fn fit(&self, queue: &wgpu::Queue, win_size: PhysicalSize<u32>) {
        let (x0, y0, x1, y1) =
            letterbox_rect(self.rect, self.w, self.h, win_size.width, win_size.height);
        let verts = quad(x0, y0, x1, y1);
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
    }

    pub(super) fn upload(&self, queue: &wgpu::Queue, data: &[u8]) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.w * 4),
                rows_per_image: Some(self.h),
            },
            wgpu::Extent3d {
                width: self.w,
                height: self.h,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// A coloured rectangle in NDC. Renders with `fs_solid`, which reads its color
/// from a 16-byte uniform.
pub(super) struct SolidQuad {
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) vbuf: wgpu::Buffer,
    rect: (f32, f32, f32, f32),
}

impl SolidQuad {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        color: [f32; 4],
        rect: (f32, f32, f32, f32),
    ) -> Self {
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solid-color"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ubuf, 0, bytemuck::cast_slice(&color));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("solid-bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solid-vbuf"),
            size: (std::mem::size_of::<Vertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let verts = quad(rect.0, rect.1, rect.2, rect.3);
        queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&verts));
        Self {
            bind_group,
            vbuf,
            rect,
        }
    }

    pub(super) fn set_rect(&mut self, queue: &wgpu::Queue, rect: (f32, f32, f32, f32)) {
        self.rect = rect;
        let verts = quad(rect.0, rect.1, rect.2, rect.3);
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
    }
}
