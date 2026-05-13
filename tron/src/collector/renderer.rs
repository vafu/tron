use anyhow::Result;
use tron_api::{Frame, PixelFormat, Renderer, RoiResult, Size};
use tron_core::render::roi_overlay::{RoiOverlayRenderer, RoiOverlayView};
use tron_core::render::wgpu::{NdcRect, WgpuFrameRenderer, WgpuFrameView, WgpuSurfaceContext};

pub struct CollectorView<'a> {
    pub rgb: Option<Frame<'a>>,
    pub ir: Option<Frame<'a>>,
    pub rgb_palm_roi: Option<RoiResult>,
    pub rgb_roi: Option<RoiResult>,
}

pub struct CollectorRenderer {
    surface: WgpuSurfaceContext,
    rgb: WgpuFrameRenderer,
    ir: WgpuFrameRenderer,
    rgb_palm_roi_overlay: RoiOverlayRenderer,
    rgb_roi_overlay: RoiOverlayRenderer,
}

impl CollectorRenderer {
    pub async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
        let surface = WgpuSurfaceContext::new(target, size, "tron-collector-wgpu-device").await?;
        let format = surface.format();
        Ok(Self {
            rgb: WgpuFrameRenderer::new(surface.device(), format),
            ir: WgpuFrameRenderer::new(surface.device(), format),
            rgb_palm_roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            rgb_roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            surface,
        })
    }

    pub fn resize(&mut self, size: Size) {
        self.surface.resize(size);
    }
}

impl<'a> Renderer<CollectorView<'a>> for CollectorRenderer {
    fn render(&mut self, view: CollectorView<'a>) -> Result<()> {
        if let Some(rgb) = view.rgb
            && rgb.format != PixelFormat::Bgra8
        {
            anyhow::bail!("collector RGB feed expects BGRA8 frames");
        }

        self.surface.render(
            "tron-collector-render-pass",
            wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.025,
                a: 1.0,
            },
            |surface| {
                let mut pass = surface.pass;
                if let Some(rgb) = view.rgb {
                    self.rgb.render(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: rgb,
                        rect: NdcRect::LEFT,
                        target_size: surface.size,
                    })?;
                    if let Some(rgb_palm_roi) = view.rgb_palm_roi {
                        self.rgb_palm_roi_overlay.render(RoiOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            roi: rgb_palm_roi.rect,
                            oriented_roi: rgb_palm_roi.oriented_box,
                            color: [1.0, 0.62, 0.08, 1.0],
                            frame_size: rgb.meta.size,
                            rect: NdcRect::LEFT,
                            target_size: surface.size,
                        })?;
                    }
                    if let Some(rgb_roi) = view.rgb_roi {
                        self.rgb_roi_overlay.render(RoiOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            roi: rgb_roi.rect,
                            oriented_roi: None,
                            color: [0.2, 1.0, 0.2, 1.0],
                            frame_size: rgb.meta.size,
                            rect: NdcRect::LEFT,
                            target_size: surface.size,
                        })?;
                    }
                }
                if let Some(ir) = view.ir {
                    self.ir.render(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: ir,
                        rect: NdcRect::RIGHT,
                        target_size: surface.size,
                    })?;
                }
                Ok(())
            },
        )
    }
}
