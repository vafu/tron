struct U { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: U;

struct VsIn {
  @location(0) pos: vec3<f32>,
  @location(1) color: vec3<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) local: vec3<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
  var out: VsOut;
  out.pos = u.mvp * vec4<f32>(in.pos, 1.0);
  out.color = in.color;
  out.local = in.pos;
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let a = abs(in.local);
  let near_x = smoothstep(0.46, 0.66, a.x);
  let near_y = smoothstep(0.46, 0.66, a.y);
  let near_z = smoothstep(0.46, 0.66, a.z);
  let edge = clamp(near_x * near_y + near_x * near_z + near_y * near_z, 0.0, 1.0);
  let pulse = 0.82 + 0.18 * sin((in.local.x + in.local.y + in.local.z) * 9.0);
  let face = vec3<f32>(0.02, 0.10, 0.14);
  let tron = vec3<f32>(0.25, 1.05, 1.25) * (1.25 + edge * 1.8 * pulse);
  let hot = vec3<f32>(1.0, 1.0, 1.0) * edge * 0.9;
  let color = mix(face, tron + hot, edge);
  return vec4<f32>(color, 0.94);
}

struct EdgeIn {
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) kind: f32,
  @location(3) color: vec3<f32>,
  @location(4) intensity: f32,
};

struct EdgeOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) kind: f32,
  @location(2) color: vec3<f32>,
  @location(3) intensity: f32,
};

@vertex
fn vs_edge(in: EdgeIn) -> EdgeOut {
  var out: EdgeOut;
  out.pos = vec4<f32>(in.pos, 0.0, 1.0);
  out.uv = in.uv;
  out.kind = in.kind;
  out.color = in.color;
  out.intensity = in.intensity;
  return out;
}

@fragment
fn fs_edge(in: EdgeOut) -> @location(0) vec4<f32> {
  let kind = i32(in.kind + 0.5);
  if (kind == 0) {
    let r = length(in.uv);
    let ring = exp(-pow((r - 0.58) * 4.2, 2.0));
    let pip = exp(-pow(r * 4.2, 2.0));
    let halo = exp(-r * 2.0) * 0.65 * smoothstep(1.08, 0.45, r);
    let core = clamp(ring + pip, 0.0, 1.0);
    let lum = (core + halo) * in.intensity;
    let col = mix(in.color, vec3<f32>(1.0, 1.0, 1.0), smoothstep(0.45, 1.0, core));
    return vec4<f32>(col * lum, clamp(lum, 0.0, 1.0));
  }

  let d = abs(in.uv.y);
  let core = smoothstep(0.22, 0.0, d);
  let halo = exp(-pow(d * 2.0, 2.0)) * 0.55;
  let pulse = exp(-pow(abs(in.uv.x - 0.5) * 4.2, 2.0)) * 0.35;
  let lum = (core + halo + pulse) * in.intensity;
  let col = mix(in.color, vec3<f32>(1.0, 1.0, 1.0), smoothstep(0.35, 1.0, core + pulse));
  return vec4<f32>(col * lum, clamp(lum, 0.0, 1.0));
}
