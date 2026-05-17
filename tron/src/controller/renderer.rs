use anyhow::Result;
use tron_api::{PixelFormat, PointerOutput, Sink, Size};
use tron_core::render::hand_landmarks_overlay::{
    HandLandmarksOverlayRenderer, HandLandmarksOverlayView,
};
use tron_core::render::hand_velocity_overlay::{
    HandVelocityOverlayRenderer, HandVelocityOverlayView,
};
use tron_core::render::roi_overlay::{RoiOverlayRenderer, RoiOverlayView};
use tron_core::render::wgpu::{
    NdcRect, WgpuCachedFrameView, WgpuFrameRenderer, WgpuFrameView, WgpuSurfaceContext,
};

use crate::pipeline::ControllerFrame;
use crate::pointer_sink::PointerOverlaySink;

pub struct Renderer {
    surface: WgpuSurfaceContext,
    rgb: WgpuFrameRenderer,
    palm_roi_overlay: RoiOverlayRenderer,
    landmark_input_roi_overlay: RoiOverlayRenderer,
    roi_overlay: RoiOverlayRenderer,
    landmarks_overlay: HandLandmarksOverlayRenderer,
    velocity_overlay: HandVelocityOverlayRenderer,
    pointer: PointerOverlaySink,
}

impl Renderer {
    pub async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
        let surface = WgpuSurfaceContext::new(target, size, "tron-controller-wgpu-device").await?;
        let format = surface.format();
        Ok(Self {
            rgb: WgpuFrameRenderer::new(surface.device(), format),
            palm_roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            landmark_input_roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            roi_overlay: RoiOverlayRenderer::new(surface.device(), format),
            landmarks_overlay: HandLandmarksOverlayRenderer::new(surface.device(), format),
            velocity_overlay: HandVelocityOverlayRenderer::new(surface.device(), format),
            pointer: PointerOverlaySink::new(surface.device(), format),
            surface,
        })
    }

    pub fn resize(&mut self, size: Size) {
        self.surface.resize(size);
    }

    pub async fn render_cached(&mut self) -> Result<()> {
        self.surface.render(
            "tron-controller-cached-render-pass",
            wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.025,
                a: 1.0,
            },
            |surface| {
                let mut pass = surface.pass;
                pollster::block_on(self.rgb.consume(WgpuCachedFrameView {
                    queue: surface.queue,
                    pass: &mut pass,
                    rect: NdcRect::FULL,
                    target_size: surface.size,
                }))?;
                self.pointer
                    .render(surface.device, surface.queue, &mut pass, surface.size)?;
                Ok(())
            },
        )
    }
}

#[async_trait::async_trait(?Send)]
impl<'a> Sink<&'a ControllerFrame<'a>> for Renderer {
    async fn consume(&mut self, view: &'a ControllerFrame<'a>) -> Result<()> {
        if view.rgb.format != PixelFormat::Bgra8 {
            anyhow::bail!("controller RGB feed expects BGRA8 frames");
        }

        self.surface.render(
            "tron-controller-render-pass",
            wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.025,
                a: 1.0,
            },
            |surface| {
                let mut pass = surface.pass;
                let rgb = view.rgb;
                pollster::block_on(self.rgb.consume(WgpuFrameView {
                    device: surface.device,
                    queue: surface.queue,
                    pass: &mut pass,
                    frame: rgb,
                    rect: NdcRect::FULL,
                    target_size: surface.size,
                }))?;
                if let Some(palm_roi) = view.palm_roi {
                    pollster::block_on(self.palm_roi_overlay.consume(RoiOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        roi: palm_roi.rect,
                        oriented_roi: palm_roi.oriented_box,
                        color: [1.0, 0.9, 0.0, 1.0],
                        frame_size: rgb.meta.size,
                        rect: NdcRect::FULL,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(landmark_input_roi) = view.landmark_input_roi {
                    pollster::block_on(self.landmark_input_roi_overlay.consume(RoiOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        roi: landmark_input_roi.rect,
                        oriented_roi: landmark_input_roi.oriented_box,
                        color: [0.2, 1.0, 0.2, 1.0],
                        frame_size: rgb.meta.size,
                        rect: NdcRect::FULL,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(rgb_roi) = view.output_roi {
                    pollster::block_on(self.roi_overlay.consume(RoiOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        roi: rgb_roi.rect,
                        oriented_roi: rgb_roi.oriented_box,
                        color: [1.0, 0.12, 0.12, 1.0],
                        frame_size: rgb.meta.size,
                        rect: NdcRect::FULL,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(landmarks) = view.landmarks.as_ref() {
                    pollster::block_on(self.landmarks_overlay.consume(HandLandmarksOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        landmarks,
                        frame_size: rgb.meta.size,
                        rect: NdcRect::FULL,
                        target_size: surface.size,
                    }))?;
                }
                if let Some(motion) = view.landmark_motion.as_ref() {
                    pollster::block_on(self.velocity_overlay.consume(HandVelocityOverlayView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        motion,
                        frame_size: rgb.meta.size,
                        rect: NdcRect::FULL,
                        target_size: surface.size,
                    }))?;
                }
                self.pointer
                    .render(surface.device, surface.queue, &mut pass, surface.size)?;
                Ok(())
            },
        )
    }
}

#[async_trait::async_trait(?Send)]
impl Sink<PointerOutput> for Renderer {
    async fn consume(&mut self, output: PointerOutput) -> Result<()> {
        self.pointer.consume(output).await
    }
}
