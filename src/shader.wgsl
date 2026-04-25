struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    var o: VsOut;
    o.pos = vec4<f32>(pos, 0.0, 1.0);
    o.uv = uv;
    return o;
}

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs_tex(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv);
}

struct Solid { color: vec4<f32> };
@group(0) @binding(0) var<uniform> solid: Solid;

@fragment
fn fs_solid(in: VsOut) -> @location(0) vec4<f32> {
    return solid.color;
}
