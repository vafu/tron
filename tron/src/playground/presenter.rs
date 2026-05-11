use crate::metadata::PlaygroundMetadata;
use anyhow::Result;
use tron_api::{Frame, PixelFormat, Presenter, Rect, RoiResult, Size};
use tron_core::present::roi_overlay::{RoiOverlayPresenter, RoiOverlayView};
use tron_core::present::wgpu::{NdcRect, WgpuFramePresenter, WgpuFrameView, WgpuSurfaceContext};

pub struct PlaygroundView<'a> {
    pub rgb: Option<Frame<'a>>,
    pub depth_cue: Option<Frame<'a>>,
    pub ir_diff: Option<Frame<'a>>,
    pub roi: Option<RoiResult>,
    pub rgb_roi: Option<RoiResult>,
    pub camera_roi: Option<Rect>,
    pub metadata: PlaygroundMetadata,
}

pub struct PlaygroundPresenter {
    surface: WgpuSurfaceContext,
    depth_cue: WgpuFramePresenter,
    ir_diff: WgpuFramePresenter,
    roi_overlay: RoiOverlayPresenter,
    camera_roi_overlay: RoiOverlayPresenter,
    rgb_roi_overlay: RoiOverlayPresenter,
    rgb: WgpuFramePresenter,
}

impl PlaygroundPresenter {
    pub async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, size: Size) -> Result<Self> {
        let surface = WgpuSurfaceContext::new(target, size, "tron-playground-wgpu-device").await?;
        let format = surface.format();
        let depth_cue = WgpuFramePresenter::new(surface.device(), format);
        let ir_diff = WgpuFramePresenter::new(surface.device(), format);
        let roi_overlay = RoiOverlayPresenter::new(surface.device(), format);
        let camera_roi_overlay = RoiOverlayPresenter::new(surface.device(), format);
        let rgb_roi_overlay = RoiOverlayPresenter::new(surface.device(), format);
        let rgb = WgpuFramePresenter::new(surface.device(), format);

        Ok(Self {
            surface,
            depth_cue,
            ir_diff,
            roi_overlay,
            camera_roi_overlay,
            rgb_roi_overlay,
            rgb,
        })
    }

    pub fn resize(&mut self, size: Size) {
        self.surface.resize(size);
    }
}

impl<'a> Presenter<PlaygroundView<'a>> for PlaygroundPresenter {
    fn present(&mut self, view: PlaygroundView<'a>) -> Result<()> {
        let _ = view.metadata;
        if let Some(rgb) = view.rgb
            && rgb.format != PixelFormat::Bgra8
        {
            anyhow::bail!("RGB feed expects BGRA8 frames");
        }
        self.surface.render(
            "tron-playground-render-pass",
            wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.025,
                a: 1.0,
            },
            |surface| {
                let mut pass = surface.pass;
                if let Some(depth_cue) = view.depth_cue {
                    self.depth_cue.present(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: depth_cue,
                        rect: NdcRect {
                            x0: -1.0,
                            y0: 0.0,
                            x1: 0.0,
                            y1: 1.0,
                        },
                        target_size: surface.size,
                    })?;
                }
                if let Some(ir_diff) = view.ir_diff {
                    let rect = NdcRect {
                        x0: 0.0,
                        y0: 0.0,
                        x1: 1.0,
                        y1: 1.0,
                    };
                    self.ir_diff.present(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: ir_diff,
                        rect,
                        target_size: surface.size,
                    })?;
                    if let Some(roi) = view.roi {
                        self.roi_overlay.present(RoiOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            roi: roi.rect,
                            color: [0.1, 0.9, 1.0, 1.0],
                            frame_size: ir_diff.meta.size,
                            rect,
                            target_size: surface.size,
                        })?;
                    }
                    if let Some(camera_roi) = view.camera_roi {
                        self.camera_roi_overlay.present(RoiOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            roi: camera_roi,
                            color: [1.0, 0.1, 0.08, 1.0],
                            frame_size: ir_diff.meta.size,
                            rect,
                            target_size: surface.size,
                        })?;
                    }
                }
                if let Some(rgb) = view.rgb {
                    let rect = NdcRect {
                        x0: -1.0,
                        y0: -1.0,
                        x1: 1.0,
                        y1: 0.0,
                    };
                    self.rgb.present(WgpuFrameView {
                        device: surface.device,
                        queue: surface.queue,
                        pass: &mut pass,
                        frame: rgb,
                        rect,
                        target_size: surface.size,
                    })?;
                    if let Some(rgb_roi) = view.rgb_roi {
                        self.rgb_roi_overlay.present(RoiOverlayView {
                            device: surface.device,
                            queue: surface.queue,
                            pass: &mut pass,
                            roi: rgb_roi.rect,
                            color: [0.2, 1.0, 0.2, 1.0],
                            frame_size: rgb.meta.size,
                            rect,
                            target_size: surface.size,
                        })?;
                    }
                }
                Ok(())
            },
        )
    }
}
