use super::V;

pub(super) fn push_bone(
    out: &mut Vec<V>,
    a: [f32; 2],
    b: [f32; 2],
    half_w_px: f32,
    ndcx: f32,
    ndcy: f32,
    kind: f32,
    intensity: f32,
    alert: f32,
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
    out.push(V {
        pos: am,
        uv: [0.0, -1.0],
        kind,
        intensity,
        alert,
    });
    out.push(V {
        pos: bm,
        uv: [1.0, -1.0],
        kind,
        intensity,
        alert,
    });
    out.push(V {
        pos: bp,
        uv: [1.0, 1.0],
        kind,
        intensity,
        alert,
    });
    out.push(V {
        pos: am,
        uv: [0.0, -1.0],
        kind,
        intensity,
        alert,
    });
    out.push(V {
        pos: bp,
        uv: [1.0, 1.0],
        kind,
        intensity,
        alert,
    });
    out.push(V {
        pos: ap,
        uv: [0.0, 1.0],
        kind,
        intensity,
        alert,
    });
}

pub(super) fn push_joint(
    out: &mut Vec<V>,
    c: [f32; 2],
    r_px: f32,
    ndcx: f32,
    ndcy: f32,
    intensity: f32,
    alert: f32,
) {
    let rx = r_px * ndcx;
    let ry = r_px * ndcy;
    let tl = [c[0] - rx, c[1] + ry];
    let tr = [c[0] + rx, c[1] + ry];
    let bl = [c[0] - rx, c[1] - ry];
    let br = [c[0] + rx, c[1] - ry];
    out.push(V {
        pos: tl,
        uv: [-1.0, 1.0],
        kind: 0.0,
        intensity,
        alert,
    });
    out.push(V {
        pos: bl,
        uv: [-1.0, -1.0],
        kind: 0.0,
        intensity,
        alert,
    });
    out.push(V {
        pos: br,
        uv: [1.0, -1.0],
        kind: 0.0,
        intensity,
        alert,
    });
    out.push(V {
        pos: tl,
        uv: [-1.0, 1.0],
        kind: 0.0,
        intensity,
        alert,
    });
    out.push(V {
        pos: br,
        uv: [1.0, -1.0],
        kind: 0.0,
        intensity,
        alert,
    });
    out.push(V {
        pos: tr,
        uv: [1.0, 1.0],
        kind: 0.0,
        intensity,
        alert,
    });
}
