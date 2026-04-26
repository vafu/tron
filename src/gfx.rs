use crate::camera::SharedImage;
use crate::pipeline::{SharedHand, SharedMask};
use crate::proximity::SharedProx;
use crate::skeleton_render::{letterbox_rect, SkeletonRenderer};
use crate::types::Image;
use anyhow::{Context, Result};
use std::sync::Arc;
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub struct Gfx {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,

    tex_pipeline: wgpu::RenderPipeline,
    solid_pipeline: wgpu::RenderPipeline,

    /// Center pane: raw RGB camera, with skeleton overlay.
    main_view: TexQuad,
    /// Side pane: RGB darkened by IR mask (the "dimmed" debug image).
    masked_view: TexQuad,
    /// Side pane: grayscale IR foreground signal (the mask itself).
    mask_view: TexQuad,

    bar_bg: SolidQuad,
    bar_fill: SolidQuad,

    skeleton: SkeletonRenderer,

    main_pane: (f32, f32, f32, f32),

    rgb_src: SharedImage,
    #[allow(dead_code)]
    ir_src: SharedImage,
    prox_src: SharedProx,
    hand_src: SharedHand,
    mask_src: SharedMask,

    /// Scratch buffer for R8 → RGBA8 expansion when uploading the mask.
    mask_rgba: Vec<u8>,

    prox_max: i64,
}

impl Gfx {
    pub async fn new(
        window: Arc<Window>,
        rgb_src: SharedImage,
        ir_src: SharedImage,
        prox_src: SharedProx,
        hand_src: SharedHand,
        mask_src: SharedMask,
        rgb_size: (u32, u32),
        ir_size: (u32, u32),
    ) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .context("request adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
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
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex_bgl"),
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

        let solid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("solid_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let tex_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tex_layout"),
            bind_group_layouts: &[&tex_bgl],
            push_constant_ranges: &[],
        });
        let solid_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solid_layout"),
            bind_group_layouts: &[&solid_bgl],
            push_constant_ranges: &[],
        });

        let tex_pipeline = make_pipeline(&device, &tex_layout, &shader, "fs_tex", format);
        let solid_pipeline = make_pipeline(&device, &solid_layout, &shader, "fs_solid", format);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // UI Layout (normalized NDC -1..1):
        //   Top row: helpers — IR-diff mask (left), masked RGB / pre-landmark (right).
        //   Below:   main RGB + landmarks, full width.
        let mask_rect   = (-0.95,  0.30, -0.05,  0.95);
        let masked_rect = ( 0.05,  0.30,  0.95,  0.95);
        let main_rect   = (-0.95, -0.90,  0.95,  0.20);

        let main_view   = TexQuad::new(&device, &tex_bgl, &sampler, rgb_size.0, rgb_size.1, main_rect);
        let masked_view = TexQuad::new(&device, &tex_bgl, &sampler, rgb_size.0, rgb_size.1, masked_rect);
        let mask_view   = TexQuad::new(&device, &tex_bgl, &sampler, ir_size.0, ir_size.1, mask_rect);
        main_view.fit(&queue, size);
        masked_view.fit(&queue, size);
        mask_view.fit(&queue, size);

        let bar_bg = SolidQuad::new(&device, &queue, &solid_bgl, [0.1, 0.1, 0.15, 1.0], (-1.0, -1.0, 1.0, -0.97));
        let bar_fill = SolidQuad::new(&device, &queue, &solid_bgl, [0.2, 0.8, 0.4, 1.0], (-1.0, -1.0, -1.0, -0.97));

        let skeleton = SkeletonRenderer::new(&device, format);

        Ok(Self {
            window, surface, device, queue, config, size,
            tex_pipeline, solid_pipeline,
            main_view, masked_view, mask_view, bar_bg, bar_fill,
            main_pane: main_rect,
            rgb_src, ir_src, prox_src, hand_src, mask_src,
            mask_rgba: Vec::new(),
            skeleton,
            prox_max: 1,
        })
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 { return; }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.main_view.fit(&self.queue, size);
        self.masked_view.fit(&self.queue, size);
        self.mask_view.fit(&self.queue, size);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Pull latest data
        let rgb_opt = self.rgb_src.lock().unwrap().clone();
        let hand = self.hand_src.lock().unwrap().clone();
        let mask_opt = self.mask_src.lock().unwrap().clone();

        // Main pane: raw RGB camera (skeleton drawn on top later).
        if let Some(f) = &rgb_opt {
            if f.seq != self.main_view.last_seq {
                self.main_view.upload(&self.queue, &f.data);
                self.main_view.last_seq = f.seq;
            }
        }

        // Masked-RGB pane: pipeline's debug image (RGB darkened by IR mask).
        // Falls back to raw RGB until the pipeline produces its first frame.
        if let Some(state) = &hand {
            if let Some(dbg) = &state.debug_image {
                if dbg.seq != self.masked_view.last_seq {
                    self.masked_view.upload(&self.queue, &dbg.data);
                    self.masked_view.last_seq = dbg.seq;
                }
            }
        } else if let Some(f) = &rgb_opt {
            if f.seq != self.masked_view.last_seq {
                self.masked_view.upload(&self.queue, &f.data);
                self.masked_view.last_seq = f.seq;
            }
        }

        // Mask pane: grayscale IR-diff. Texture is RGBA8 so expand R8 → RGBA8.
        if let Some(m) = &mask_opt {
            if m.seq != self.mask_view.last_seq {
                let needed = (m.width * m.height) as usize * 4;
                if self.mask_rgba.len() != needed {
                    self.mask_rgba.resize(needed, 255);
                }
                expand_to_rgba(m, &mut self.mask_rgba);
                self.mask_view.upload(&self.queue, &self.mask_rgba);
                self.mask_view.last_seq = m.seq;
            }
        }

        // Update prox bar fill
        let prox = *self.prox_src.lock().unwrap();
        if let Some(p) = prox {
            if p > self.prox_max { self.prox_max = p; }
            let norm = (p as f32 / self.prox_max as f32).clamp(0.0, 1.0);
            self.bar_fill.set_rect(&self.queue, (-1.0, -1.0, -1.0 + 2.0 * norm, -0.97));
        }

        // Pull hand state, update skeleton + ROI mesh.
        let mut have_overlay = false;
        let mut gesture_label = "";
        if let Some(state) = &hand {
            have_overlay = true;
            gesture_label = state.gesture.map(|g| g.name()).unwrap_or("");
            let clip = letterbox_rect(self.main_pane, self.main_view.w, self.main_view.h, self.size.width, self.size.height);
            self.skeleton.update(&self.queue, Some(&state.landmarks), Some(&state.roi), clip, (self.size.width, self.size.height));
        }

        // Window title combines proximity + gesture.
        let title = match (prox, gesture_label) {
            (Some(p), g) if !g.is_empty() => format!("tron — prox: {p} — {g}"),
            (Some(p), _) => format!("tron — prox: {p}"),
            (None, g) if !g.is_empty() => format!("tron — {g}"),
            _ => "tron".into(),
        };
        self.window.set_title(&title);

        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("enc"),
        });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.03, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.tex_pipeline);
            for q in [&self.main_view, &self.masked_view, &self.mask_view] {
                rp.set_bind_group(0, &q.bind_group, &[]);
                rp.set_vertex_buffer(0, q.vbuf.slice(..));
                rp.draw(0..6, 0..1);
            }

            rp.set_pipeline(&self.solid_pipeline);
            for q in [&self.bar_bg, &self.bar_fill] {
                rp.set_bind_group(0, &q.bind_group, &[]);
                rp.set_vertex_buffer(0, q.vbuf.slice(..));
                rp.draw(0..6, 0..1);
            }

            if have_overlay {
                self.skeleton.draw(&mut rp);
            }
        }
        self.queue.submit(Some(enc.finish()));
        frame.present();
        Ok(())
    }
}

fn make_pipeline(
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
fn expand_to_rgba(src: &Image, dst: &mut [u8]) {
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
        Vertex { pos: [x0, y1], uv: [0.0, 0.0] },
        Vertex { pos: [x0, y0], uv: [0.0, 1.0] },
        Vertex { pos: [x1, y0], uv: [1.0, 1.0] },
        Vertex { pos: [x0, y1], uv: [0.0, 0.0] },
        Vertex { pos: [x1, y0], uv: [1.0, 1.0] },
        Vertex { pos: [x1, y1], uv: [1.0, 0.0] },
    ]
}

struct TexQuad {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    vbuf: wgpu::Buffer,
    w: u32,
    h: u32,
    rect: (f32, f32, f32, f32),
    last_seq: u64,
}

impl TexQuad {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        w: u32,
        h: u32,
        rect: (f32, f32, f32, f32),
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
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
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (std::mem::size_of::<Vertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { texture, bind_group, vbuf, w, h, rect, last_seq: 0 }
    }

    fn fit(&self, queue: &wgpu::Queue, win_size: PhysicalSize<u32>) {
        let (x0, y0, x1, y1) = letterbox_rect(self.rect, self.w, self.h, win_size.width, win_size.height);
        let verts = quad(x0, y0, x1, y1);
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
    }

    fn upload(&self, queue: &wgpu::Queue, data: &[u8]) {
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
            wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
        );
    }
}

/// A coloured rectangle in NDC. Renders with `fs_solid`, which reads its color
/// from a 16-byte uniform.
struct SolidQuad {
    bind_group: wgpu::BindGroup,
    vbuf: wgpu::Buffer,
    rect: (f32, f32, f32, f32),
}

impl SolidQuad {
    fn new(
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
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solid-vbuf"),
            size: (std::mem::size_of::<Vertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let verts = quad(rect.0, rect.1, rect.2, rect.3);
        queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&verts));
        Self { bind_group, vbuf, rect }
    }

    fn set_rect(&mut self, queue: &wgpu::Queue, rect: (f32, f32, f32, f32)) {
        self.rect = rect;
        let verts = quad(rect.0, rect.1, rect.2, rect.3);
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
    }
}
