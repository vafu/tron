use anyhow::Result;
use tron_api::{CheckerboardDetection, Frame, Sink, Size};
use tron_core::render::checkerboard_overlay::{
    CheckerboardOverlayRenderer, CheckerboardOverlayView,
};
use tron_core::render::wgpu::{NdcRect, WgpuFrameRenderer, WgpuFrameView, WgpuSurfaceContext};

pub struct CalibrationView<'a> {
    pub rgb: Option<Frame<'a>>,
    pub ir: Option<Frame<'a>>,
    pub rgb_checkerboard: Option<&'a CheckerboardDetection>,
    pub ir_checkerboard: Option<&'a CheckerboardDetection>,
}

pub struct CalibrationRenderer {
    surface: WgpuSurfaceContext,
    rgb: WgpuFrameRenderer,
    ir: WgpuFrameRenderer,
    rgb_checkerboard: CheckerboardOverlayRenderer,
    ir_checkerboard: CheckerboardOverlayRenderer,
}

impl CalibrationRenderer {
    pub async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
        let surface = WgpuSurfaceContext::new(target, size, "tron-calibration-wgpu-device").await?;
        let format = surface.format();
        let rgb = WgpuFrameRenderer::new(surface.device(), format);
        let ir = WgpuFrameRenderer::new(surface.device(), format);
        let rgb_checkerboard = CheckerboardOverlayRenderer::new(surface.device(), format);
        let ir_checkerboard = CheckerboardOverlayRenderer::new(surface.device(), format);
        Ok(Self {
            surface,
            rgb,
            ir,
            rgb_checkerboard,
            ir_checkerboard,
        })
    }

    pub fn resize(&mut self, size: Size) {
        self.surface.resize(size);
    }
}

#[async_trait::async_trait(?Send)]
impl<'a> Sink<CalibrationView<'a>> for CalibrationRenderer {
    async fn consume(&mut self, view: CalibrationView<'a>) -> Result<()> {
        self.surface.render(
            "tron-calibration-render-pass",
            wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.025,
                a: 1.0,
            },
            |surface| {
                let mut pass = surface.pass;
                if let Some(rgb) = view.rgb {
                    pollster::block_on(self.rgb.consume(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: rgb,
                        rect: NdcRect::LEFT,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(detection) = view.rgb_checkerboard {
                    pollster::block_on(self.rgb_checkerboard.consume(CheckerboardOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        detection,
                        color: [1.0, 0.05, 0.05, 1.0],
                        rect: NdcRect::LEFT,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(ir) = view.ir {
                    pollster::block_on(self.ir.consume(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: ir,
                        rect: NdcRect::RIGHT,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(detection) = view.ir_checkerboard {
                    pollster::block_on(self.ir_checkerboard.consume(CheckerboardOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        detection,
                        color: [1.0, 0.35, 0.1, 1.0],
                        rect: NdcRect::RIGHT,
                        target_size: surface.size,
                    }))?;
                }
                Ok(())
            },
        )
    }
}
