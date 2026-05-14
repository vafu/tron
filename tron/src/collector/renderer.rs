use anyhow::Result;
use tron_api::{Frame, PixelFormat, Renderer, RoiResult, Size};
use tron_core::render::hand_landmarks_overlay::{
    HandLandmarksOverlayRenderer, HandLandmarksOverlayView,
};
use tron_core::render::roi_overlay::{RoiOverlayRenderer, RoiOverlayView};
use tron_core::render::wgpu::{NdcRect, WgpuFrameRenderer, WgpuFrameView, WgpuSurfaceContext};
use tron_core::roi::mediapipe::HandLandmarks;

pub struct CollectorView<'a> {
    pub rgb: Option<Frame<'a>>,
    pub ir: Option<Frame<'a>>,
    pub rgb_palm_roi: Option<RoiResult>,
    pub rgb_roi: Option<RoiResult>,
    pub ir_roi: Option<RoiResult>,
    pub rgb_landmarks: Option<&'a HandLandmarks>,
    pub ir_landmarks: Option<&'a HandLandmarks>,
}

pub struct CollectorRenderer {
    surface: WgpuSurfaceContext,
    rgb: WgpuFrameRenderer,
    ir: WgpuFrameRenderer,
    rgb_palm_roi_overlay: RoiOverlayRenderer,
    rgb_roi_overlay: RoiOverlayRenderer,
    ir_roi_overlay: RoiOverlayRenderer,
    rgb_landmarks_overlay: HandLandmarksOverlayRenderer,
    ir_landmarks_overlay: HandLandmarksOverlayRenderer,
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
                let rgbrect = if view.ir.is_some() {
                    NdcRect::LEFT
                } else {
                    NdcRect::FULL
                };
                let mut pass = surface.pass;
                if let Some(rgb) = view.rgb {
                    self.rgb.render(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: rgb,
                        rect: rgbrect,
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
                            rect: rgbrect,
                            target_size: surface.size,
                        })?;
                    }
                    if let Some(rgb_roi) = view.rgb_roi {
                        self.rgb_roi_overlay.render(RoiOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            roi: rgb_roi.rect,
                            oriented_roi: rgb_roi.oriented_box,
                            color: [0.2, 1.0, 0.2, 1.0],
                            frame_size: rgb.meta.size,
                            rect: rgbrect,
                            target_size: surface.size,
                        })?;
                    }
                    if let Some(landmarks) = view.rgb_landmarks {
                        self.rgb_landmarks_overlay
                            .render(HandLandmarksOverlayView {
                                device: surface.device,
                                queue: surface.queue,
                                pass: &mut pass,
                                landmarks,
                                frame_size: rgb.meta.size,
                                rect: rgbrect,
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
                    if let Some(ir_roi) = view.ir_roi {
                        self.ir_roi_overlay.render(RoiOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            roi: ir_roi.rect,
                            oriented_roi: ir_roi.oriented_box,
                            color: [0.1, 0.85, 1.0, 1.0],
                            frame_size: ir.meta.size,
                            rect: NdcRect::RIGHT,
                            target_size: surface.size,
                        })?;
                    }
                    if let Some(landmarks) = view.ir_landmarks {
                        self.ir_landmarks_overlay.render(HandLandmarksOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            landmarks,
                            frame_size: ir.meta.size,
                            rect: NdcRect::RIGHT,
                            target_size: surface.size,
                        })?;
                    }
                }
                Ok(())
            },
        )
    }
}
