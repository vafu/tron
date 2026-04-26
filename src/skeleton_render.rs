use crate::types::{HandLandmarks, RectNorm};
use bytemuck::{Pod, Zeroable};
use std::time::Instant;

/// MediaPipe Hands keypoint topology.
const EDGES: &[(usize, usize)] = &[
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
];

const JOINT_R_PX: f32 = 22.0;
const BONE_HALF_W_PX: f32 = 7.0;
const ROI_HALF_W_PX: f32 = 1.8;

/// Per-landmark radius multiplier — bigger near the palm, smaller toward the
/// tips, so the rendered hand looks like a natural skeleton with vertebrae of
/// decreasing size as you walk out each finger.
/// MediaPipe topology:
///   0 = wrist
///   1,5,9,13,17 = base knuckles (MCP / thumb CMC)
///   2,6,10,14,18 = PIP joints
///   3,7,11,15,19 = DIP joints
///   4,8,12,16,20 = fingertips
const JOINT_RADIUS_SCALE: [f32; 21] = [
    1.60, // 0  wrist
    1.30, 1.05, 0.85, 0.65, // thumb
    1.30, 1.00, 0.80, 0.65, // index
    1.30, 1.00, 0.80, 0.65, // middle
    1.30, 1.00, 0.80, 0.65, // ring
    1.25, 0.95, 0.78, 0.62, // pinky
];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct V {
    pos: [f32; 2],
    uv: [f32; 2],
    /// 0 = joint ring, 1 = bone, 2 = ROI line.
    kind: f32,
    /// Per-vertex brightness multiplier.
    intensity: f32,
}

const LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<V>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![
        0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Float32
    ],
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct U {
    time: f32,
    _pad: [f32; 3],
}

pub struct SkeletonRenderer {
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ubuf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: usize,
    count: u32,
    start: Instant,
}

impl SkeletonRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skeleton-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skeleton-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skeleton-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skeleton-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Additive blending — neon glow stacks over the camera image.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
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

        // 21 joints + 21 bones + 4 ROI bones, 6 verts each.
        let capacity = (21 + 21 + 4) * 6;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skeleton-vbuf"),
            size: (capacity * std::mem::size_of::<V>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skeleton-ubuf"),
            size: std::mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skeleton-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
        });

        Self {
            pipeline,
            vbuf,
            ubuf,
            bind_group,
            capacity,
            count: 0,
            start: Instant::now(),
        }
    }

    /// `clip` is the NDC rect of the camera image (post-letterbox). `win_size`
    /// is needed so we can size joint discs and bone thickness in pixels —
    /// otherwise NDC anisotropy distorts them.
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        lm: Option<&HandLandmarks>,
        roi: Option<&RectNorm>,
        clip: (f32, f32, f32, f32),
        win_size: (u32, u32),
    ) {
        let t = self.start.elapsed().as_secs_f32();
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&U { time: t, _pad: [0.0; 3] }));

        let (x0, y0, x1, y1) = clip;
        let to_ndc = |x: f32, y: f32| -> [f32; 2] {
            let nx = x0 + (x1 - x0) * x;
            let ny = y1 + (y0 - y1) * y;
            [nx, ny]
        };

        // pixel→NDC: 1 px in NDC = 2 / win_dim
        let ndcx = 2.0 / win_size.0 as f32;
        let ndcy = 2.0 / win_size.1 as f32;

        let mut verts: Vec<V> = Vec::with_capacity(self.capacity);

        if let Some(r) = roi {
            let tl = to_ndc(r.x, r.y);
            let tr = to_ndc(r.x + r.w, r.y);
            let br = to_ndc(r.x + r.w, r.y + r.h);
            let bl = to_ndc(r.x, r.y + r.h);
            for (a, b) in [(tl, tr), (tr, br), (br, bl), (bl, tl)] {
                push_bone(&mut verts, a, b, ROI_HALF_W_PX, ndcx, ndcy, 2.0, 0.85);
            }
        }

        if let Some(lm) = lm {
            // Bones first; joints stack on top via additive blending.
            for &(a, b) in EDGES {
                let pa = to_ndc(lm.points[a].x, lm.points[a].y);
                let pb = to_ndc(lm.points[b].x, lm.points[b].y);
                push_bone(&mut verts, pa, pb, BONE_HALF_W_PX, ndcx, ndcy, 1.0, 1.0);
            }
            for (i, p) in lm.points.iter().enumerate() {
                let c = to_ndc(p.x, p.y);
                let r = JOINT_R_PX * JOINT_RADIUS_SCALE[i];
                push_joint(&mut verts, c, r, ndcx, ndcy, 1.0);
            }
        }

        self.count = verts.len() as u32;
        if !verts.is_empty() {
            queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
        }
    }

    pub fn draw<'r>(&'r self, rp: &mut wgpu::RenderPass<'r>) {
        if self.count == 0 {
            return;
        }
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.vbuf.slice(..));
        rp.draw(0..self.count, 0..1);
    }
}

fn push_bone(
    out: &mut Vec<V>,
    a: [f32; 2],
    b: [f32; 2],
    half_w_px: f32,
    ndcx: f32,
    ndcy: f32,
    kind: f32,
    intensity: f32,
) {
    // Compute perpendicular in pixel space so thickness is uniform regardless
    // of bone orientation, then convert back to NDC.
    let dx_px = (b[0] - a[0]) / ndcx;
    let dy_px = (b[1] - a[1]) / ndcy;
    let len = (dx_px * dx_px + dy_px * dy_px).sqrt().max(1e-6);
    let perp_x_ndc = (-dy_px / len * half_w_px) * ndcx;
    let perp_y_ndc = (dx_px / len * half_w_px) * ndcy;

    let am = [a[0] - perp_x_ndc, a[1] - perp_y_ndc];
    let ap = [a[0] + perp_x_ndc, a[1] + perp_y_ndc];
    let bp = [b[0] + perp_x_ndc, b[1] + perp_y_ndc];
    let bm = [b[0] - perp_x_ndc, b[1] - perp_y_ndc];
    out.push(V { pos: am, uv: [0.0, -1.0], kind, intensity });
    out.push(V { pos: bm, uv: [1.0, -1.0], kind, intensity });
    out.push(V { pos: bp, uv: [1.0,  1.0], kind, intensity });
    out.push(V { pos: am, uv: [0.0, -1.0], kind, intensity });
    out.push(V { pos: bp, uv: [1.0,  1.0], kind, intensity });
    out.push(V { pos: ap, uv: [0.0,  1.0], kind, intensity });
}

fn push_joint(out: &mut Vec<V>, c: [f32; 2], r_px: f32, ndcx: f32, ndcy: f32, intensity: f32) {
    let rx = r_px * ndcx;
    let ry = r_px * ndcy;
    let tl = [c[0] - rx, c[1] + ry];
    let tr = [c[0] + rx, c[1] + ry];
    let bl = [c[0] - rx, c[1] - ry];
    let br = [c[0] + rx, c[1] - ry];
    out.push(V { pos: tl, uv: [-1.0,  1.0], kind: 0.0, intensity });
    out.push(V { pos: bl, uv: [-1.0, -1.0], kind: 0.0, intensity });
    out.push(V { pos: br, uv: [ 1.0, -1.0], kind: 0.0, intensity });
    out.push(V { pos: tl, uv: [-1.0,  1.0], kind: 0.0, intensity });
    out.push(V { pos: br, uv: [ 1.0, -1.0], kind: 0.0, intensity });
    out.push(V { pos: tr, uv: [ 1.0,  1.0], kind: 0.0, intensity });
}

/// Compute the NDC rect the image actually occupies inside `pane` after
/// preserving aspect ratio.
pub fn letterbox_rect(
    pane: (f32, f32, f32, f32),
    img_w: u32,
    img_h: u32,
    win_w: u32,
    win_h: u32,
) -> (f32, f32, f32, f32) {
    let (px0, py0, px1, py1) = pane;
    let pane_px_w = (px1 - px0) * 0.5 * win_w as f32;
    let pane_px_h = (py1 - py0) * 0.5 * win_h as f32;
    let pane_ar = pane_px_w / pane_px_h;
    let img_ar = img_w as f32 / img_h as f32;
    let (mut x0, mut y0, mut x1, mut y1) = pane;
    if img_ar > pane_ar {
        let new_px_h = pane_px_w / img_ar;
        let half_ndc = new_px_h / win_h as f32;
        let cy = (py0 + py1) * 0.5;
        y0 = cy - half_ndc;
        y1 = cy + half_ndc;
    } else {
        let new_px_w = pane_px_h * img_ar;
        let half_ndc = new_px_w / win_w as f32;
        let cx = (px0 + px1) * 0.5;
        x0 = cx - half_ndc;
        x1 = cx + half_ndc;
    }
    (x0, y0, x1, y1)
}

const SHADER: &str = r#"
struct U { time: f32, p1: f32, p2: f32, p3: f32 };
@group(0) @binding(0) var<uniform> u: U;

struct VsIn {
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) kind: f32,
  @location(3) intensity: f32,
};
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) kind: f32,
  @location(2) intensity: f32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
  var o: VsOut;
  o.pos = vec4<f32>(in.pos, 0.0, 1.0);
  o.uv = in.uv;
  o.kind = in.kind;
  o.intensity = in.intensity;
  return o;
}

// Tron palette: cyan body, white-hot core.
const CYAN: vec3<f32> = vec3<f32>(0.40, 0.92, 1.10);
const HOT:  vec3<f32> = vec3<f32>(1.00, 1.00, 1.00);

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let kind = i32(in.kind + 0.5);

  if (kind == 0) {
    // Joint — neon ring with a fat hot center and soft outer halo.
    let r = length(in.uv);
    let breathe = 0.85 + 0.15 * sin(u.time * 3.6);
    // Thicker ring: lower exponent → wider band around its peak.
    let ring = exp(-pow((r - 0.58) * 4.5, 2.0));
    // Beefy hot pip — fills more of the inner disc.
    let pip = exp(-pow(r * 4.5, 2.0));
    // Soft outer halo trailing off past the ring.
    let halo = exp(-r * 2.0) * 0.55 * smoothstep(1.05, 0.5, r);
    let core = clamp(ring + pip, 0.0, 1.0);
    let lum = (core + halo) * breathe * in.intensity;
    let col = mix(CYAN, HOT, smoothstep(0.40, 1.0, core));
    return vec4<f32>(col * lum, clamp(lum, 0.0, 1.0));
  }

  if (kind == 1) {
    // Bone — bright core line + perpendicular halo, with a travelling pulse.
    let d = abs(in.uv.y);
    let core = smoothstep(0.18, 0.0, d);
    let halo = exp(-pow(d * 2.1, 2.0)) * 0.55;
    // Pulse that races down the bone.
    let pulse_pos = fract(u.time * 0.55);
    let pulse_d = abs(in.uv.x - pulse_pos);
    let pulse = exp(-pow(pulse_d * 7.0, 2.0)) * smoothstep(1.0, 0.0, d) * 0.85;
    let lum = (core + halo + pulse) * in.intensity;
    let col = mix(CYAN, HOT, smoothstep(0.45, 1.0, core + pulse * 0.6));
    return vec4<f32>(col * lum, clamp(lum, 0.0, 1.0));
  }

  // ROI — thin cyan glow, faint and breathing slowly.
  let d = abs(in.uv.y);
  let core = smoothstep(0.55, 0.0, d);
  let halo = exp(-pow(d * 1.6, 2.0)) * 0.35;
  let breathe = 0.7 + 0.3 * sin(u.time * 1.4);
  let lum = (core * 0.6 + halo) * in.intensity * breathe;
  return vec4<f32>(CYAN * lum, clamp(lum, 0.0, 1.0));
}
"#;
