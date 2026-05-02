struct U { time: f32, p1: f32, p2: f32, p3: f32 };
@group(0) @binding(0) var<uniform> u: U;

struct VsIn {
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) kind: f32,
  @location(3) intensity: f32,
  @location(4) alert: f32,
};
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) kind: f32,
  @location(2) intensity: f32,
  @location(3) alert: f32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
  var o: VsOut;
  o.pos = vec4<f32>(in.pos, 0.0, 1.0);
  o.uv = in.uv;
  o.kind = in.kind;
  o.intensity = in.intensity;
  o.alert = in.alert;
  return o;
}

// Tron palette: cyan body, white-hot core.
const CYAN: vec3<f32> = vec3<f32>(0.40, 0.92, 1.10);
const RED:  vec3<f32> = vec3<f32>(1.80, 0.04, 0.02);
const GREEN: vec3<f32> = vec3<f32>(0.15, 1.55, 0.28);
const HOT:  vec3<f32> = vec3<f32>(1.00, 1.00, 1.00);
const HOT_RED: vec3<f32> = vec3<f32>(1.00, 0.25, 0.10);
const HOT_GREEN: vec3<f32> = vec3<f32>(0.78, 1.00, 0.68);

fn body_color(alert: f32) -> vec3<f32> {
  if (alert > 1.5) {
    return GREEN;
  }
  return mix(CYAN, RED, clamp(alert, 0.0, 1.0));
}

fn hot_color(alert: f32) -> vec3<f32> {
  if (alert > 1.5) {
    return HOT_GREEN;
  }
  return mix(HOT, HOT_RED, clamp(alert, 0.0, 1.0));
}

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
    let col = mix(body_color(in.alert), hot_color(in.alert), smoothstep(0.40, 1.0, core));
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
    let col = mix(body_color(in.alert), hot_color(in.alert), smoothstep(0.45, 1.0, core + pulse * 0.6));
    return vec4<f32>(col * lum, clamp(lum, 0.0, 1.0));
  }

  // ROI — thin cyan glow, faint and breathing slowly.
  if (kind == 3) {
    let lum = 0.95;
    return vec4<f32>(hot_color(in.alert) * lum, 0.88);
  }

  let d = abs(in.uv.y);
  let core = smoothstep(0.55, 0.0, d);
  let halo = exp(-pow(d * 1.6, 2.0)) * 0.35;
  let breathe = 0.7 + 0.3 * sin(u.time * 1.4);
  let lum = (core * 0.6 + halo) * in.intensity * breathe;
  return vec4<f32>(body_color(in.alert) * lum, clamp(lum, 0.0, 1.0));
}
