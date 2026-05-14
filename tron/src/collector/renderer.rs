use anyhow::Result;
use tron_api::{PixelFormat, Sink, Size};
use tron_core::render::hand_landmarks_overlay::{
    HandLandmarksOverlayRenderer, HandLandmarksOverlayView,
};
use tron_core::render::roi_overlay::{RoiOverlayRenderer, RoiOverlayView};
use tron_core::render::wgpu::{NdcRect, WgpuFrameRenderer, WgpuFrameView, WgpuSurfaceContext};

use crate::aggregate::Aggregate;

pub struct Renderer {
    surface: WgpuSurfaceContext,
    rgb: WgpuFrameRenderer,
    ir: WgpuFrameRenderer,
    rgb_palm_roi_overlay: RoiOverlayRenderer,
    rgb_roi_overlay: RoiOverlayRenderer,
    ir_roi_overlay: RoiOverlayRenderer,
    rgb_landmarks_overlay: HandLandmarksOverlayRenderer,
    ir_landmarks_overlay: HandLandmarksOverlayRenderer,
}

impl Renderer {
    pub async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
        let surface = WgpuSurfaceContext::new(target, size, "tron-collector-wgpu-device").await?;
        let format = surface.format();
        Ok(Self {
            rgb: WgpuFrameRenderer::new(surface.device(), format),
            ir: WgpuFrameRenderer::new(surface.device(), format),
            rgb_palm_roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            rgb_roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            ir_roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            rgb_landmarks_overlay: HandLandmarksOverlayRenderer::new(surface.device(), format),
            ir_landmarks_overlay: HandLandmarksOverlayRenderer::new(surface.device(), format),
            surface,
        })
    }

    pub fn resize(&mut self, size: Size) {
        self.surface.resize(size);
    }
}

#[async_trait::async_trait(?Send)]
impl<'a> Sink<&'a Aggregate<'a>> for Renderer {
    async fn consume(&mut self, view: &'a Aggregate<'a>) -> Result<()> {
        if view.rgb.format != PixelFormat::Bgra8 {
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
                let rgbrect = NdcRect::LEFT;
                let mut pass = surface.pass;
                let rgb = view.rgb;
                pollster::block_on(self.rgb.consume(WgpuFrameView {
                    device: surface.device,
                    queue: surface.queue,
                    pass: &mut pass,
                    frame: rgb,
                    rect: rgbrect,
                    target_size: surface.size,
                }))?;
                if let Some(rgb_palm_roi) = view.palm_roi {
                    pollster::block_on(self.rgb_palm_roi_overlay.consume(RoiOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        roi: rgb_palm_roi.rect,
                        oriented_roi: rgb_palm_roi.oriented_box,
                        color: [1.0, 0.62, 0.08, 1.0],
                        frame_size: rgb.meta.size,
                        rect: rgbrect,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(rgb_roi) = view.rgb_roi {
                    pollster::block_on(self.rgb_roi_overlay.consume(RoiOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        roi: rgb_roi.rect,
                        oriented_roi: rgb_roi.oriented_box,
                        color: [0.2, 1.0, 0.2, 1.0],
                        frame_size: rgb.meta.size,
                        rect: rgbrect,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(landmarks) = view.landmarks.as_ref() {
                    pollster::block_on(self.rgb_landmarks_overlay.consume(
                        HandLandmarksOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            landmarks,
                            frame_size: rgb.meta.size,
                            rect: rgbrect,
                            target_size: surface.size,
                        },
                    ))?;
                }

                let ir = view.ir;
                pollster::block_on(self.ir.consume(WgpuFrameView {
                    device: surface.device,
                    queue: surface.queue,
                    pass: &mut pass,
                    frame: ir,
                    rect: NdcRect::RIGHT,
                    target_size: surface.size,
                }))?;
                if let Some(ir_roi) = view
                    .projection
                    .as_ref()
                    .and_then(|projection| projection.roi)
                {
                    pollster::block_on(self.ir_roi_overlay.consume(RoiOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        roi: ir_roi.rect,
                        oriented_roi: ir_roi.oriented_box,
                        color: [0.1, 0.85, 1.0, 1.0],
                        frame_size: ir.meta.size,
                        rect: NdcRect::RIGHT,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(landmarks) = view
                    .projection
                    .as_ref()
                    .and_then(|projection| projection.landmarks.as_ref())
                {
                    pollster::block_on(self.ir_landmarks_overlay.consume(
                        HandLandmarksOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            landmarks,
                            frame_size: ir.meta.size,
                            rect: NdcRect::RIGHT,
                            target_size: surface.size,
                        },
                    ))?;
                }
                Ok(())
            },
        )
    }
}
