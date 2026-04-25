use crate::types::{HandLandmarks, RectNorm};
use bytemuck::{Pod, Zeroable};

/// MediaPipe Hands keypoint topology.
const EDGES: &[(usize, usize)] = &[
    // thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    // index
    (0, 5), (5, 6), (6, 7), (7, 8),
    // middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    // ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    // pinky
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LineVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

const LINE_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<LineVertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
};

pub struct SkeletonRenderer {
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    capacity: usize,
    line_count: u32,
    point_count: u32,
    point_offset: u64,
}

impl SkeletonRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skeleton-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skeleton-pl"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let line_pipeline = make_pipeline(device, &layout, &shader, format, wgpu::PrimitiveTopology::LineList);
        let point_pipeline = make_pipeline(device, &layout, &shader, format, wgpu::PrimitiveTopology::TriangleList);

        // Lines: 2 verts per skeleton edge + 8 verts for ROI rect outline.
        // Points: 6 verts per landmark.
        let capacity = (EDGES.len() * 2) + 8 + (21 * 6);
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skeleton-vbuf"),
            size: (capacity * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            line_pipeline,
            point_pipeline,
            vbuf,
            capacity,
            line_count: 0,
            point_count: 0,
            point_offset: 0,
        }
    }

    /// Update vertex buffer. `pane` is the NDC rectangle (x0,y0,x1,y1) the
    /// landmarks should be drawn into; landmark coords are 0..1 in source-image
    /// space (top-left origin). `aspect_clip` is an inner rect (NDC) the camera
    /// quad actually occupies inside the pane after letterboxing.
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        lm: Option<&HandLandmarks>,
        roi: Option<&RectNorm>,
        clip: (f32, f32, f32, f32),
    ) {
        let (x0, y0, x1, y1) = clip;
        let to_ndc = |x: f32, y: f32| -> [f32; 2] {
            // landmark/ROI x,y are 0..1 in source-image coords (top-left origin).
            // Map into the camera quad's NDC clip rect (y up).
            let nx = x0 + (x1 - x0) * x;
            let ny = y1 + (y0 - y1) * y;
            [nx, ny]
        };

        let edge_color = [0.20, 1.00, 0.55, 0.95];
        let point_color = [1.00, 0.85, 0.20, 1.00];
        let roi_color = [1.00, 0.45, 0.20, 0.95];

        let mut verts: Vec<LineVertex> = Vec::with_capacity(self.capacity);

        // ROI rectangle outline (4 line segments = 8 verts).
        if let Some(r) = roi {
            let tl = to_ndc(r.x, r.y);
            let tr = to_ndc(r.x + r.w, r.y);
            let br = to_ndc(r.x + r.w, r.y + r.h);
            let bl = to_ndc(r.x, r.y + r.h);
            verts.push(LineVertex { pos: tl, color: roi_color });
            verts.push(LineVertex { pos: tr, color: roi_color });
            verts.push(LineVertex { pos: tr, color: roi_color });
            verts.push(LineVertex { pos: br, color: roi_color });
            verts.push(LineVertex { pos: br, color: roi_color });
            verts.push(LineVertex { pos: bl, color: roi_color });
            verts.push(LineVertex { pos: bl, color: roi_color });
            verts.push(LineVertex { pos: tl, color: roi_color });
        }

        // Skeleton edges.
        if let Some(lm) = lm {
            for &(a, b) in EDGES {
                let pa = lm.points[a];
                let pb = lm.points[b];
                verts.push(LineVertex { pos: to_ndc(pa.x, pa.y), color: edge_color });
                verts.push(LineVertex { pos: to_ndc(pb.x, pb.y), color: edge_color });
            }
        }

        self.line_count = verts.len() as u32;
        self.point_offset = (verts.len() * std::mem::size_of::<LineVertex>()) as u64;

        // Landmark points as little squares.
        if let Some(lm) = lm {
            let r = 0.008;
            for p in &lm.points {
                let c = to_ndc(p.x, p.y);
                let tl = [c[0] - r, c[1] + r];
                let tr = [c[0] + r, c[1] + r];
                let bl = [c[0] - r, c[1] - r];
                let br = [c[0] + r, c[1] - r];
                verts.push(LineVertex { pos: tl, color: point_color });
                verts.push(LineVertex { pos: bl, color: point_color });
                verts.push(LineVertex { pos: br, color: point_color });
                verts.push(LineVertex { pos: tl, color: point_color });
                verts.push(LineVertex { pos: br, color: point_color });
                verts.push(LineVertex { pos: tr, color: point_color });
            }
            self.point_count = (21 * 6) as u32;
        } else {
            self.point_count = 0;
        }

        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
    }

    pub fn draw<'r>(&'r self, rp: &mut wgpu::RenderPass<'r>) {
        if self.line_count > 0 {
            rp.set_pipeline(&self.line_pipeline);
            rp.set_vertex_buffer(0, self.vbuf.slice(..self.point_offset));
            rp.draw(0..self.line_count, 0..1);
        }
        if self.point_count > 0 {
            rp.set_pipeline(&self.point_pipeline);
            rp.set_vertex_buffer(0, self.vbuf.slice(self.point_offset..));
            rp.draw(0..self.point_count, 0..1);
        }
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    topology: wgpu::PrimitiveTopology,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("skeleton-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs",
            buffers: &[LINE_LAYOUT],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: "fs",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

const SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var o: VsOut;
    o.pos = vec4<f32>(pos, 0.0, 1.0);
    o.color = color;
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

// We want the consumer to know exactly which NDC rect the RGB image occupies
// after letterboxing. Re-export a helper.
pub fn letterbox_rect(pane: (f32, f32, f32, f32), img_w: u32, img_h: u32, win_w: u32, win_h: u32) -> (f32, f32, f32, f32) {
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
