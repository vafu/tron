use anyhow::Result;
use tron_api::{CheckerboardDetection, Frame, Presenter, Size};
use tron_core::present::checkerboard_overlay::{
    CheckerboardOverlayPresenter, CheckerboardOverlayView,
};
use tron_core::present::wgpu::{NdcRect, WgpuFramePresenter, WgpuFrameView, WgpuSurfaceContext};

pub struct CalibrationView<'a> {
    pub rgb: Option<Frame<'a>>,
    pub ir: Option<Frame<'a>>,
    pub rgb_checkerboard: Option<&'a CheckerboardDetection>,
    pub ir_checkerboard: Option<&'a CheckerboardDetection>,
}

pub struct CalibrationPresenter {
    surface: WgpuSurfaceContext,
    rgb: WgpuFramePresenter,
    ir: WgpuFramePresenter,
    rgb_checkerboard: CheckerboardOverlayPresenter,
    ir_checkerboard: CheckerboardOverlayPresenter,
}

impl CalibrationPresenter {
    pub async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
        let surface = WgpuSurfaceContext::new(target, size, "tron-calibration-wgpu-device").await?;
        let format = surface.format();
        let rgb = WgpuFramePresenter::new(surface.device(), format);
        let ir = WgpuFramePresenter::new(surface.device(), format);
        let rgb_checkerboard = CheckerboardOverlayPresenter::new(surface.device(), format);
        let ir_checkerboard = CheckerboardOverlayPresenter::new(surface.device(), format);
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

impl<'a> Presenter<CalibrationView<'a>> for CalibrationPresenter {
    fn present(&mut self, view: CalibrationView<'a>) -> Result<()> {
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
                    self.rgb.present(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: rgb,
                        rect: NdcRect::LEFT,
                        target_size: surface.size,
                    })?;
                }
                if let Some(detection) = view.rgb_checkerboard {
                    self.rgb_checkerboard.present(CheckerboardOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        detection,
                        color: [0.1, 0.9, 1.0, 1.0],
                        rect: NdcRect::LEFT,
                        target_size: surface.size,
                    })?;
                }
                if let Some(ir) = view.ir {
                    self.ir.present(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: ir,
                        rect: NdcRect::RIGHT,
                        target_size: surface.size,
                    })?;
                }
                if let Some(detection) = view.ir_checkerboard {
                    self.ir_checkerboard.present(CheckerboardOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        detection,
                        color: [1.0, 0.35, 0.1, 1.0],
                        rect: NdcRect::RIGHT,
                        target_size: surface.size,
                    })?;
                }
                Ok(())
            },
        )
    }
}
