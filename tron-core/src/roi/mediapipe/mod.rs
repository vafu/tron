use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use glam::{Affine2, Vec2};
use opencv::core::{self, Mat, Vec4b};
use opencv::imgproc;
use opencv::prelude::*;
use tron_api::{Frame, PixelFormat, Size};

mod landmark;
mod palm;

pub use landmark::{
    HandLandmark, HandLandmarks, Handedness, MediaPipeHandLandmarkConfig,
    MediaPipeHandLandmarkInput, MediaPipeHandLandmarkProcessor,
};
pub use palm::{MediaPipeRoiConfig, MediaPipeRoiProcessor};

const MODEL_INPUT_CHANNELS: usize = 3;
const BGRA_CHANNELS: usize = 4;
const DEBUG_DUMP_DIR_ENV: &str = "TRON_MEDIAPIPE_DUMP_DIR";
const HAND_CONNECTIONS: [(usize, usize); 20] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

pub(super) fn preprocess_bgra(
    frame: Frame<'_>,
    input_size: usize,
    inverse_affine: &Mat,
    debug_label: &str,
    output: &mut [f32],
) -> Result<()> {
    anyhow::ensure!(
        frame.format == PixelFormat::Bgra8,
        "MediaPipe preprocessing expects BGRA8 frames, got {:?}",
        frame.format
    );
    let source_w = frame.meta.size.width as usize;
    let source_h = frame.meta.size.height as usize;
    anyhow::ensure!(source_w > 0 && source_h > 0, "empty RGB frame");
    anyhow::ensure!(
        frame.buffer.stride() == source_w * BGRA_CHANNELS,
        "MediaPipe OpenCV preprocessing requires tightly packed BGRA8 frames"
    );
    anyhow::ensure!(
        output.len() >= MODEL_INPUT_CHANNELS * input_size * input_size,
        "MediaPipe tensor output buffer too small"
    );

    let raw = unsafe { frame.buffer.raw() };
    anyhow::ensure!(
        raw.len() >= source_w * source_h * BGRA_CHANNELS,
        "BGRA frame buffer too small"
    );
    let src_data: &[Vec4b] =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const Vec4b, source_w * source_h) };
    let src = Mat::new_rows_cols_with_data(source_h as i32, source_w as i32, src_data)
        .context("wrap BGRA frame in OpenCV Mat")?;

    let mut warped = Mat::default();
    imgproc::warp_affine(
        &src,
        &mut warped,
        inverse_affine,
        core::Size::new(input_size as i32, input_size as i32),
        imgproc::INTER_LINEAR | imgproc::WARP_INVERSE_MAP,
        core::BORDER_CONSTANT,
        core::Scalar::all(0.0),
    )
    .context("warp MediaPipe model input")?;

    let _ = debug_label;
    bgra_mat_to_nchw(&warped, input_size, output)
}

pub(super) fn letterbox_inverse_affine(
    source_size: Size,
    resized_w: usize,
    resized_h: usize,
    pad_x: usize,
    pad_y: usize,
    mirror_x: bool,
    mirror_y: bool,
) -> Result<Mat> {
    let source_w = source_size.width as usize;
    let source_h = source_size.height as usize;
    let scale_x = source_w as f64 / resized_w as f64;
    let scale_y = source_h as f64 / resized_h as f64;
    let (m00, m02) = if mirror_x {
        (
            -scale_x,
            (resized_w as f64 - 0.5 + pad_x as f64) * scale_x - 0.5,
        )
    } else {
        (scale_x, (0.5 - pad_x as f64) * scale_x - 0.5)
    };
    let (m11, m12) = if mirror_y {
        (
            -scale_y,
            (resized_h as f64 - 0.5 + pad_y as f64) * scale_y - 0.5,
        )
    } else {
        (scale_y, (0.5 - pad_y as f64) * scale_y - 0.5)
    };
    Mat::from_slice_2d(&[[m00, 0.0, m02], [0.0, m11, m12]])
        .context("create letterbox affine transform")
}

pub(super) fn crop_inverse_affine(
    source_size: Size,
    crop: Affine2,
    input_size: usize,
    mirror_x: bool,
    mirror_y: bool,
) -> Result<Mat> {
    let scale = 1.0 / input_size as f32;
    let x_axis = crop.matrix2.x_axis * scale;
    let y_axis = crop.matrix2.y_axis * scale;
    let origin = crop.translation + (crop.matrix2.x_axis + crop.matrix2.y_axis) * (0.5 * scale)
        - Vec2::splat(0.5);

    let (x_axis, y_axis, origin) =
        apply_mirror_to_affine(source_size, x_axis, y_axis, origin, mirror_x, mirror_y);

    Mat::from_slice_2d(&[
        [x_axis.x as f64, y_axis.x as f64, origin.x as f64],
        [x_axis.y as f64, y_axis.y as f64, origin.y as f64],
    ])
    .context("create crop affine transform")
}

fn apply_mirror_to_affine(
    source_size: Size,
    mut x_axis: Vec2,
    mut y_axis: Vec2,
    mut origin: Vec2,
    mirror_x: bool,
    mirror_y: bool,
) -> (Vec2, Vec2, Vec2) {
    if mirror_x {
        x_axis.x = -x_axis.x;
        y_axis.x = -y_axis.x;
        origin.x = source_size.width.saturating_sub(1) as f32 - origin.x;
    }
    if mirror_y {
        x_axis.y = -x_axis.y;
        y_axis.y = -y_axis.y;
        origin.y = source_size.height.saturating_sub(1) as f32 - origin.y;
    }
    (x_axis, y_axis, origin)
}

fn bgra_mat_to_nchw(input: &Mat, input_size: usize, output: &mut [f32]) -> Result<()> {
    let input_size_squared = input_size * input_size;
    let bytes = input.data_bytes()?;
    anyhow::ensure!(
        bytes.len() >= input_size_squared * BGRA_CHANNELS,
        "MediaPipe warped input buffer too small"
    );
    for y in 0..input_size {
        for x in 0..input_size {
            let offset = (y * input_size + x) * BGRA_CHANNELS;
            let b = bytes[offset];
            let g = bytes[offset + 1];
            let r = bytes[offset + 2];

            let dst = y * input_size + x;
            output[dst] = r as f32 / 255.0;
            output[input_size_squared + dst] = g as f32 / 255.0;
            output[2 * input_size_squared + dst] = b as f32 / 255.0;
        }
    }
    Ok(())
}

pub(super) fn dump_landmark_overlay(
    frame_id: u64,
    input_size: usize,
    tensor: &[f32],
    points: &[Option<Vec2>; 21],
) -> Result<()> {
    let Some(root) = std::env::var_os(DEBUG_DUMP_DIR_ENV).map(PathBuf::from) else {
        return Ok(());
    };

    let mut rgb = tensor_to_rgb(input_size, tensor)?;
    for (a, b) in HAND_CONNECTIONS {
        let Some(a) = points[a] else {
            continue;
        };
        let Some(b) = points[b] else {
            continue;
        };
        draw_line(&mut rgb, input_size, a, b, [255, 230, 32]);
    }
    for point in points.iter().flatten().copied() {
        draw_cross(&mut rgb, input_size, point, [32, 225, 255]);
    }

    fs::create_dir_all(&root).with_context(|| {
        format!(
            "create MediaPipe debug dump directory {:?}",
            root.as_os_str()
        )
    })?;
    let path = root.join(format!("landmark-overlay-{frame_id:08}.ppm"));
    write_rgb_ppm(&path, input_size, &rgb)
}

fn tensor_to_rgb(input_size: usize, tensor: &[f32]) -> Result<Vec<u8>> {
    let input_size_squared = input_size * input_size;
    anyhow::ensure!(
        tensor.len() >= MODEL_INPUT_CHANNELS * input_size_squared,
        "MediaPipe tensor debug buffer too small"
    );
    let mut rgb = vec![0_u8; input_size_squared * MODEL_INPUT_CHANNELS];
    for i in 0..input_size_squared {
        rgb[i * MODEL_INPUT_CHANNELS] = tensor[i].mul_add(255.0, 0.0).clamp(0.0, 255.0) as u8;
        rgb[i * MODEL_INPUT_CHANNELS + 1] = tensor[input_size_squared + i]
            .mul_add(255.0, 0.0)
            .clamp(0.0, 255.0) as u8;
        rgb[i * MODEL_INPUT_CHANNELS + 2] = tensor[2 * input_size_squared + i]
            .mul_add(255.0, 0.0)
            .clamp(0.0, 255.0) as u8;
    }
    Ok(rgb)
}

fn write_rgb_ppm(path: &PathBuf, input_size: usize, rgb: &[u8]) -> Result<()> {
    let mut writer = BufWriter::new(
        File::create(path).with_context(|| format!("create MediaPipe debug dump {path:?}"))?,
    );
    writeln!(writer, "P6\n{input_size} {input_size}\n255")
        .with_context(|| format!("write MediaPipe debug dump header {path:?}"))?;
    writer
        .write_all(rgb)
        .with_context(|| format!("write MediaPipe debug dump pixels {path:?}"))?;
    Ok(())
}

fn draw_cross(rgb: &mut [u8], input_size: usize, point: Vec2, color: [u8; 3]) {
    let Some((x, y)) = image_point(input_size, point) else {
        return;
    };
    for d in -3..=3 {
        put_pixel(rgb, input_size, x + d, y, color);
        put_pixel(rgb, input_size, x, y + d, color);
    }
}

fn draw_line(rgb: &mut [u8], input_size: usize, a: Vec2, b: Vec2, color: [u8; 3]) {
    let Some((mut x0, mut y0)) = image_point(input_size, a) else {
        return;
    };
    let Some((x1, y1)) = image_point(input_size, b) else {
        return;
    };

    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put_pixel(rgb, input_size, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn image_point(input_size: usize, point: Vec2) -> Option<(i32, i32)> {
    if !point.x.is_finite() || !point.y.is_finite() {
        return None;
    }
    let max = input_size.saturating_sub(1) as f32;
    let x = (point.x * input_size as f32).round().clamp(0.0, max) as i32;
    let y = (point.y * input_size as f32).round().clamp(0.0, max) as i32;
    Some((x, y))
}

fn put_pixel(rgb: &mut [u8], input_size: usize, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 || x >= input_size as i32 || y >= input_size as i32 {
        return;
    }
    let offset = (y as usize * input_size + x as usize) * MODEL_INPUT_CHANNELS;
    rgb[offset..offset + MODEL_INPUT_CHANNELS].copy_from_slice(&color);
}
