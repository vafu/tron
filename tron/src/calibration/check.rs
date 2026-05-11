use std::sync::Arc;

use anyhow::{Context, Result};
use opencv::calib3d;
use opencv::core::{Mat, Point2f, Point3f, Vector};
use opencv::prelude::*;
use tron_api::{
    CheckerboardStereoCalibration, Frame, FrameMeta, FrameSource, OwnedFrame, PixelFormat,
    Renderer, Size,
};
use tron_core::pipeline::{FramePairSource, FrameSynchronizer};
use tron_core::render::wgpu::{NdcRect, WgpuFrameRenderer, WgpuFrameView, WgpuSurfaceContext};
use tron_core::view::{IntoView, ViewExt};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

pub struct CalibrationCheckConfig {
    pub calibration: CheckerboardStereoCalibration,
    pub max_sync_delta_us: i64,
    pub depth_mm: f64,
}

pub fn run<R, I>(rgb: R, ir: I, config: CalibrationCheckConfig) -> Result<()>
where
    R: FrameSource,
    I: FrameSource,
{
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = CheckApp::new(rgb, ir, config);
    event_loop.run_app(&mut app).context("run winit app")?;
    app.result
}

struct CheckApp<R, I> {
    synchronizer: FrameSynchronizer<R, I>,
    calibration: CheckerboardStereoCalibration,
    depth_mm: f64,
    window_id: Option<WindowId>,
    renderer: Option<CheckRenderer>,
    window: Option<Arc<winit::window::Window>>,
    composite: CompositeFrame,
    result: Result<()>,
}

impl<R, I> CheckApp<R, I>
where
    R: FrameSource,
    I: FrameSource,
{
    fn new(rgb: R, ir: I, config: CalibrationCheckConfig) -> Self {
        Self {
            synchronizer: FrameSynchronizer::new(rgb, ir, config.max_sync_delta_us),
            calibration: config.calibration,
            depth_mm: config.depth_mm,
            window_id: None,
            renderer: None,
            window: None,
            composite: CompositeFrame::default(),
            result: Ok(()),
        }
    }

    fn set_error(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }
}

impl<R, I> ApplicationHandler for CheckApp<R, I>
where
    R: FrameSource,
    I: FrameSource,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let attrs = WindowAttributes::default().with_title("tron calibration check");
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.set_error(event_loop, anyhow::Error::new(err).context("create window"));
                return;
            }
        };
        self.window_id = Some(window.id());
        let size = window.inner_size();
        match pollster::block_on(CheckRenderer::new(
            window.clone(),
            Size {
                width: size.width,
                height: size.height,
            },
        )) {
            Ok(renderer) => {
                self.window = Some(window);
                self.renderer = Some(renderer);
            }
            Err(err) => self.set_error(event_loop, err),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(Size {
                        width: size.width,
                        height: size.height,
                    });
                }
            }
            WindowEvent::RedrawRequested => {
                let pair = match FramePairSource::next_pair(&mut self.synchronizer) {
                    Ok(pair) => pair,
                    Err(err) => {
                        self.set_error(event_loop, err);
                        return;
                    }
                };
                let Some(pair) = pair else {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                    return;
                };

                if let Err(err) =
                    self.composite
                        .update(&pair.left, &pair.right, &self.calibration, self.depth_mm)
                {
                    self.set_error(event_loop, err);
                    return;
                }

                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                if let Err(err) = renderer.render(self.composite.frame()) {
                    self.set_error(event_loop, err);
                    return;
                }
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

struct CheckRenderer {
    surface: WgpuSurfaceContext,
    frame: WgpuFrameRenderer,
}

impl CheckRenderer {
    async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
        let surface =
            WgpuSurfaceContext::new(target, size, "tron-calibration-check-wgpu-device").await?;
        let format = surface.format();
        Ok(Self {
            frame: WgpuFrameRenderer::new(surface.device(), format),
            surface,
        })
    }

    fn resize(&mut self, size: Size) {
        self.surface.resize(size);
    }
}

impl Renderer<Frame<'_>> for CheckRenderer {
    fn render(&mut self, frame: Frame<'_>) -> Result<()> {
        self.surface.render(
            "tron-calibration-check-render-pass",
            wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.025,
                a: 1.0,
            },
            |surface| {
                let mut pass = surface.pass;
                self.frame.render(WgpuFrameView {
                    device: surface.device,
                    queue: surface.queue,
                    pass: &mut pass,
                    frame,
                    rect: NdcRect::FULL,
                    target_size: surface.size,
                })
            },
        )
    }
}

#[derive(Default)]
struct CompositeFrame {
    meta: Option<FrameMeta>,
    data: Vec<u8>,
    projection_key: Option<ProjectionMapKey>,
    projection_map: Vec<Option<(u32, u32)>>,
}

impl CompositeFrame {
    fn update(
        &mut self,
        rgb: &OwnedFrame,
        ir: &OwnedFrame,
        calibration: &CheckerboardStereoCalibration,
        depth_mm: f64,
    ) -> Result<()> {
        anyhow::ensure!(depth_mm > 0.0, "check depth must be positive");
        anyhow::ensure!(
            rgb.meta.size == calibration.left.image_size,
            "RGB frame size {:?} does not match calibration left image size {:?}",
            rgb.meta.size,
            calibration.left.image_size
        );
        anyhow::ensure!(
            ir.meta.size == calibration.right.image_size,
            "IR frame size {:?} does not match calibration right image size {:?}",
            ir.meta.size,
            calibration.right.image_size
        );

        let size = rgb.meta.size;
        let pixel_count = size.width as usize * size.height as usize;
        self.data.resize(pixel_count * 4, 255);
        self.ensure_projection_map(calibration, rgb.meta.size, ir.meta.size, depth_mm)?;

        let rgb_view = rgb.as_frame().view();
        let ir_view = ir.as_frame().view();
        for y in 0..size.height {
            let rgb_row = rgb_view.row(y)?;
            for x in 0..size.width {
                let rgb = bgra_at(rgb_row, rgb.format, x as usize)?;
                let dst = (y as usize * size.width as usize + x as usize) * 4;
                self.data[dst..dst + 4].copy_from_slice(&rgb);

                let Some((ir_x, ir_y)) =
                    self.projection_map[y as usize * size.width as usize + x as usize]
                else {
                    continue;
                };
                let ir_row = ir_view.row(ir_y)?;
                let ir = gray_at(ir_row, ir.format, ir_x as usize)?;
                blend_ir(&mut self.data[dst..dst + 4], rgb, ir, 0.38);
            }
        }

        self.meta = Some(FrameMeta { size, ..rgb.meta });
        Ok(())
    }

    fn ensure_projection_map(
        &mut self,
        calibration: &CheckerboardStereoCalibration,
        rgb_size: Size,
        ir_size: Size,
        depth_mm: f64,
    ) -> Result<()> {
        let key = ProjectionMapKey {
            rgb_size,
            ir_size,
            depth_bits: depth_mm.to_bits(),
        };
        if self.projection_key == Some(key) {
            return Ok(());
        }

        self.projection_map = build_projection_map(calibration, rgb_size, ir_size, depth_mm)?;
        self.projection_key = Some(key);
        Ok(())
    }

    fn frame(&self) -> Frame<'_> {
        let meta = self.meta.expect("composite frame was not initialized");
        Frame {
            meta,
            format: PixelFormat::Bgra8,
            stride: meta.size.width as usize * 4,
            data: &self.data,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProjectionMapKey {
    rgb_size: Size,
    ir_size: Size,
    depth_bits: u64,
}

fn build_projection_map(
    calibration: &CheckerboardStereoCalibration,
    rgb_size: Size,
    ir_size: Size,
    depth_mm: f64,
) -> Result<Vec<Option<(u32, u32)>>> {
    let pixel_count = rgb_size.width as usize * rgb_size.height as usize;

    let mut rgb_pixels = Vector::<Point2f>::with_capacity(pixel_count);
    for y in 0..rgb_size.height {
        for x in 0..rgb_size.width {
            rgb_pixels.push(Point2f::new(x as f32, y as f32));
        }
    }

    let left_camera = mat3(calibration.left.camera_matrix)?;
    let left_dist = mat_vec(&calibration.left.distortion)?;
    let mut normalized = Vector::<Point2f>::new();
    calib3d::undistort_points_def(&rgb_pixels, &mut normalized, &left_camera, &left_dist)
        .context("undistort RGB pixels")?;

    let mut object_points = Vector::<Point3f>::with_capacity(pixel_count);
    for point in normalized {
        object_points.push(Point3f::new(
            point.x * depth_mm as f32,
            point.y * depth_mm as f32,
            depth_mm as f32,
        ));
    }

    let rotation = mat3(calibration.rotation)?;
    let mut rvec = Mat::default();
    calib3d::rodrigues_def(&rotation, &mut rvec).context("convert stereo rotation to rvec")?;
    let tvec = mat_vec(&calibration.translation)?;
    let right_camera = mat3(calibration.right.camera_matrix)?;
    let right_dist = mat_vec(&calibration.right.distortion)?;
    let mut projected = Vector::<Point2f>::new();
    calib3d::project_points_def(
        &object_points,
        &rvec,
        &tvec,
        &right_camera,
        &right_dist,
        &mut projected,
    )
    .context("project RGB depth plane into IR")?;

    let mut map = Vec::with_capacity(pixel_count);
    for point in projected {
        let x = point.x as f64;
        let y = point.y as f64;
        if (0.0..ir_size.width as f64).contains(&x) && (0.0..ir_size.height as f64).contains(&y) {
            map.push(Some((x as u32, y as u32)));
        } else {
            map.push(None);
        }
    }
    Ok(map)
}

fn mat3(values: [[f64; 3]; 3]) -> Result<Mat> {
    let mat = Mat::from_slice_2d(&values).context("create OpenCV 3x3 matrix")?;
    mat.try_clone().context("clone OpenCV 3x3 matrix")
}

fn mat_vec(values: &[f64]) -> Result<Mat> {
    let mat = Mat::from_slice(values).context("create OpenCV vector")?;
    mat.try_clone().context("clone OpenCV vector")
}

fn bgra_at(row: &[u8], format: PixelFormat, x: usize) -> Result<[u8; 4]> {
    match format {
        PixelFormat::Bgra8 => {
            let offset = x * 4;
            Ok([
                row[offset],
                row[offset + 1],
                row[offset + 2],
                row[offset + 3],
            ])
        }
        PixelFormat::Gray8 => {
            let value = row[x];
            Ok([value, value, value, 255])
        }
        PixelFormat::Yuyv422 => anyhow::bail!("calibration check does not support YUYV422"),
    }
}

fn gray_at(row: &[u8], format: PixelFormat, x: usize) -> Result<u8> {
    match format {
        PixelFormat::Gray8 => Ok(row[x]),
        PixelFormat::Bgra8 => {
            let offset = x * 4;
            Ok(((row[offset] as u16 + row[offset + 1] as u16 + row[offset + 2] as u16) / 3) as u8)
        }
        PixelFormat::Yuyv422 => anyhow::bail!("calibration check does not support YUYV422"),
    }
}

fn blend_ir(dst: &mut [u8], rgb: [u8; 4], ir: u8, alpha: f32) {
    let base = 1.0 - alpha;
    dst[0] = (rgb[0] as f32 * base + ir as f32 * alpha) as u8;
    dst[1] = (rgb[1] as f32 * base + ir as f32 * alpha) as u8;
    dst[2] = (rgb[2] as f32 * base + ir as f32 * alpha) as u8;
    dst[3] = 255;
}
