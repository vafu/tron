use super::{CubeEdgeVertex, CubeVertex};

pub(super) const CUBE_VERTICES: &[CubeVertex] = &[
    CubeVertex {
        pos: [-0.7, -0.7, 0.7],
        color: [0.1, 0.9, 1.0],
    },
    CubeVertex {
        pos: [0.7, -0.7, 0.7],
        color: [0.9, 1.0, 1.0],
    },
    CubeVertex {
        pos: [0.7, 0.7, 0.7],
        color: [0.2, 0.6, 1.0],
    },
    CubeVertex {
        pos: [-0.7, 0.7, 0.7],
        color: [0.0, 0.4, 0.9],
    },
    CubeVertex {
        pos: [-0.7, -0.7, -0.7],
        color: [0.0, 0.3, 0.8],
    },
    CubeVertex {
        pos: [0.7, -0.7, -0.7],
        color: [0.0, 0.8, 0.9],
    },
    CubeVertex {
        pos: [0.7, 0.7, -0.7],
        color: [0.7, 0.9, 1.0],
    },
    CubeVertex {
        pos: [-0.7, 0.7, -0.7],
        color: [0.0, 0.6, 1.0],
    },
];

pub(super) const CUBE_INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3, 1, 5, 6, 1, 6, 2, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 3, 2, 6, 3, 6, 7, 4, 5,
    1, 4, 1, 0,
];

const CUBE_EDGES: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

pub(super) fn build_cube_overlay(
    out: &mut Vec<CubeEdgeVertex>,
    mvp: [[f32; 4]; 4],
    win_w: u32,
    win_h: u32,
) {
    let mut projected = [ProjectedPoint::default(); 8];
    for (i, v) in CUBE_VERTICES.iter().enumerate() {
        projected[i] = project_point(mvp, v.pos);
    }

    let ndcx = 2.0 / win_w as f32;
    let ndcy = 2.0 / win_h as f32;
    let edge_color = [0.35, 1.0, 1.2];
    for &(a, b) in CUBE_EDGES {
        push_cube_edge(
            out,
            projected[a].pos,
            projected[b].pos,
            8.5,
            ndcx,
            ndcy,
            edge_color,
            1.15,
        );
    }

    for p in projected {
        let radius = 13.0 + (1.0 - p.depth).clamp(0.0, 1.0) * 25.0;
        push_cube_corner(out, p.pos, radius, ndcx, ndcy, edge_color, 1.2);
    }
}

#[derive(Clone, Copy, Default)]
struct ProjectedPoint {
    pos: [f32; 2],
    depth: f32,
}

fn project_point(m: [[f32; 4]; 4], p: [f32; 3]) -> ProjectedPoint {
    let x = m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0];
    let y = m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1];
    let z = m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2];
    let w = m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3];
    let inv_w = if w.abs() > 1e-5 { 1.0 / w } else { 1.0 };
    ProjectedPoint {
        pos: [x * inv_w, y * inv_w],
        depth: (z * inv_w).clamp(0.0, 1.0),
    }
}

fn push_cube_edge(
    out: &mut Vec<CubeEdgeVertex>,
    a: [f32; 2],
    b: [f32; 2],
    half_w_px: f32,
    ndcx: f32,
    ndcy: f32,
    color: [f32; 3],
    intensity: f32,
) {
    let dx_px = (b[0] - a[0]) / ndcx;
    let dy_px = (b[1] - a[1]) / ndcy;
    let len = (dx_px * dx_px + dy_px * dy_px).sqrt().max(1e-6);
    let perp_x_ndc = (-dy_px / len * half_w_px) * ndcx;
    let perp_y_ndc = (dx_px / len * half_w_px) * ndcy;
    let am = [a[0] - perp_x_ndc, a[1] - perp_y_ndc];
    let ap = [a[0] + perp_x_ndc, a[1] + perp_y_ndc];
    let bp = [b[0] + perp_x_ndc, b[1] + perp_y_ndc];
    let bm = [b[0] - perp_x_ndc, b[1] - perp_y_ndc];
    for (pos, uv) in [
        (am, [0.0, -1.0]),
        (bm, [1.0, -1.0]),
        (bp, [1.0, 1.0]),
        (am, [0.0, -1.0]),
        (bp, [1.0, 1.0]),
        (ap, [0.0, 1.0]),
    ] {
        out.push(CubeEdgeVertex {
            pos,
            uv,
            kind: 1.0,
            color,
            intensity,
        });
    }
}

fn push_cube_corner(
    out: &mut Vec<CubeEdgeVertex>,
    c: [f32; 2],
    r_px: f32,
    ndcx: f32,
    ndcy: f32,
    color: [f32; 3],
    intensity: f32,
) {
    let rx = r_px * ndcx;
    let ry = r_px * ndcy;
    let tl = [c[0] - rx, c[1] + ry];
    let tr = [c[0] + rx, c[1] + ry];
    let bl = [c[0] - rx, c[1] - ry];
    let br = [c[0] + rx, c[1] - ry];
    for (pos, uv) in [
        (tl, [-1.0, 1.0]),
        (bl, [-1.0, -1.0]),
        (br, [1.0, -1.0]),
        (tl, [-1.0, 1.0]),
        (br, [1.0, -1.0]),
        (tr, [1.0, 1.0]),
    ] {
        out.push(CubeEdgeVertex {
            pos,
            uv,
            kind: 0.0,
            color,
            intensity,
        });
    }
}

pub(super) fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fovy * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, (far * near) / (near - far), 0.0],
    ]
}

pub(super) fn translation(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

pub(super) fn rotation_x(a: f32) -> [[f32; 4]; 4] {
    let (s, c) = a.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, s, 0.0],
        [0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub(super) fn rotation_y(a: f32) -> [[f32; 4]; 4] {
    let (s, c) = a.sin_cos();
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub(super) fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] =
                a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    out
}
