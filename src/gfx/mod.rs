mod cube;
mod depth;
mod render;
mod setup;
mod skeleton;
mod texture;

use crate::camera::SharedImage;
use crate::pipeline::{SharedHand, SharedMask, SharedPointer};
use crate::proximity::SharedProx;
use crate::types::SharedPipelineControls;
use anyhow::{Context, Result};
use cube::CubeRenderer;
use depth::DepthTexture;
use skeleton::{SkeletonRenderer, letterbox_rect};
use std::sync::Arc;
use std::time::{Duration, Instant};
use texture::{SolidQuad, TexQuad, expand_to_rgba, make_pipeline};
use winit::dpi::PhysicalSize;
use winit::window::Window;

#[derive(Clone, Copy)]
pub struct RenderOptions {
    pub cube: bool,
    pub skeleton: bool,
    pub classifier_debug: bool,
}

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
    cube: CubeRenderer,
    depth: DepthTexture,

    main_pane: (f32, f32, f32, f32),

    rgb_src: SharedImage,
    #[allow(dead_code)]
    ir_src: SharedImage,
    prox_src: SharedProx,
    controls: SharedPipelineControls,
    hand_src: SharedHand,
    mask_src: SharedMask,
    pointer_src: SharedPointer,
    options: RenderOptions,

    /// Scratch buffer for R8 → RGBA8 expansion when uploading the mask.
    mask_rgba: Vec<u8>,

    prox_max: i64,
    last_grab_pos: Option<[f32; 2]>,
    render_timing: RenderTiming,
}

#[derive(Default)]
struct RenderTiming {
    last_log: Option<Instant>,
    frames: u32,
    lock_us: u64,
    upload_us: u64,
    overlay_us: u64,
    encode_us: u64,
    submit_us: u64,
}
