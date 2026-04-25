use super::RoiHinter;
use crate::calib;
use crate::pipeline::FrameContext;
use crate::types::{Image, PixelFormat, RectNorm};
use anyhow::Result;
use opencv::core::{Mat, Point, Scalar, CV_8UC1};
use opencv::imgproc;
use opencv::prelude::*;

pub struct IrRoiHinter {
    /// Minimum blob area as a fraction of the image — rejects speckle.
    pub min_area_frac: f32,
    pad: f32,
}

impl IrRoiHinter {
    pub fn new() -> Self {
        Self { min_area_frac: 0.005, pad: 0.20 }
    }
}

impl Default for IrRoiHinter {
    fn default() -> Self {
        Self::new()
    }
}

impl RoiHinter for IrRoiHinter {
    fn hint(&mut self, ctx: &FrameContext) -> Option<RectNorm> {
        let ir = ctx.ir?;
        match find_blob(ir, self.min_area_frac) {
            Ok(Some(rect)) => {
                // Map IR-frame coords → RGB-frame coords, then pad.
                let mapped = calib::current().map_rect(rect);
                Some(mapped.padded(self.pad))
            }
            Ok(None) => None,
            Err(e) => {
                eprintln!("ir roi: {e}");
                None
            }
        }
    }
}

fn find_blob(ir: &Image, min_area_frac: f32) -> Result<Option<RectNorm>> {
    // Camera publishes IR as triplicated RGBA8. Grab the R channel into a
    // contiguous greyscale buffer.
    let w = ir.width as i32;
    let h = ir.height as i32;
    let mut grey = vec![0u8; (w * h) as usize];
    match ir.format {
        PixelFormat::Rgba8 => {
            for (i, g) in grey.iter_mut().enumerate() {
                *g = ir.data[i * 4];
            }
        }
        PixelFormat::R8 => {
            grey.copy_from_slice(&ir.data);
        }
    }

    // Wrap the buffer as a Mat (no copy).
    let mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(h, w, CV_8UC1, grey.as_mut_ptr() as *mut _)?
    };

    // Otsu threshold — IR emitter + 1/r² fall-off makes near-field a clear bimodal histogram.
    let mut binary = Mat::default();
    imgproc::threshold(
        &mat,
        &mut binary,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;

    // open(3x3) → close(5x5)
    let k3 = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        opencv::core::Size::new(3, 3),
        Point::new(-1, -1),
    )?;
    let k5 = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        opencv::core::Size::new(5, 5),
        Point::new(-1, -1),
    )?;
    let mut opened = Mat::default();
    imgproc::morphology_ex(
        &binary,
        &mut opened,
        imgproc::MORPH_OPEN,
        &k3,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &opened,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &k5,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

    // Connected components, keep largest by area (excluding label 0 = background).
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let n = imgproc::connected_components_with_stats(
        &closed,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        opencv::core::CV_32S,
    )?;

    let total_area = (w * h) as f32;
    let min_area = (total_area * min_area_frac) as i32;
    let mut best: Option<(i32, i32, i32, i32, i32)> = None; // x, y, w, h, area
    for i in 1..n {
        let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
        if area < min_area { continue; }
        let bx = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
        let by = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
        let bw = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
        let bh = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
        if best.map_or(true, |(_, _, _, _, a)| area > a) {
            best = Some((bx, by, bw, bh, area));
        }
    }

    Ok(best.map(|(bx, by, bw, bh, _)| RectNorm {
        x: bx as f32 / w as f32,
        y: by as f32 / h as f32,
        w: bw as f32 / w as f32,
        h: bh as f32 / h as f32,
    }))
}
