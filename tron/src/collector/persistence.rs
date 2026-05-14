use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tron_api::{
    DepthSample, Frame, NoContext, PixelFormat, Processor, Rect, RoiResult, SensorKind, Sink, Size,
};
use tron_core::projection::{HandProjectionOutput, LandmarkDepthEstimate};
use tron_core::roi::mediapipe::{HandLandmark, HandLandmarks, Handedness};

use crate::aggregate::Aggregate;

pub struct Persistence {
    root: PathBuf,
    metadata: MetadataProcessor,
}

impl Persistence {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        create_dir_all(&root)
            .with_context(|| format!("create collector persistence dir {}", root.display()))?;
        Ok(Self {
            root,
            metadata: MetadataProcessor,
        })
    }
}

#[async_trait::async_trait(?Send)]
impl<'a> Sink<&'a Aggregate<'a>> for Persistence {
    async fn consume(&mut self, aggregate: &'a Aggregate<'a>) -> Result<()> {
        let frame_dir = self.root.join(format!(
            "pair-{:010}-{:010}",
            aggregate.rgb.meta.id, aggregate.ir.meta.id
        ));
        create_dir_all(&frame_dir)
            .with_context(|| format!("create collector frame dir {}", frame_dir.display()))?;

        write_bmp(&aggregate.rgb, &frame_dir.join("rgb.bmp")).context("write RGB bitmap")?;
        write_bmp(&aggregate.ir, &frame_dir.join("ir.bmp")).context("write IR bitmap")?;

        let metadata = self.metadata.process(aggregate, NoContext)?;
        let file = File::create(frame_dir.join("metadata.json")).context("create metadata JSON")?;
        serde_json::to_writer_pretty(BufWriter::new(file), &metadata)
            .context("write metadata JSON")?;
        Ok(())
    }
}

struct MetadataProcessor;

impl Processor<&Aggregate<'_>, NoContext> for MetadataProcessor {
    type Output = Value;

    fn process(&mut self, aggregate: &Aggregate<'_>, _context: NoContext) -> Result<Self::Output> {
        Ok(json!({
            "pair": {
                "rgb_id": aggregate.rgb.meta.id,
                "ir_id": aggregate.ir.meta.id,
                "sync_delta_us": aggregate.sync_delta_us,
            },
            "rgb": frame_json(&aggregate.rgb, "rgb.bmp"),
            "ir": frame_json(&aggregate.ir, "ir.bmp"),
            "palm_roi": roi_json(aggregate.palm_roi),
            "rgb_roi": roi_json(aggregate.rgb_roi),
            "camera_roi": aggregate.camera_roi.map(rect_json),
            "landmarks": aggregate.landmarks.as_ref().map(landmarks_json),
            "depth_sample": aggregate.depth_sample.map(depth_sample_json),
            "projection": aggregate.projection.as_ref().map(projection_json),
        }))
    }
}

fn write_bmp(frame: &Frame<'_>, path: &Path) -> Result<()> {
    let file = File::create(path).with_context(|| format!("create bitmap {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    match frame.format {
        PixelFormat::Bgra8 => write_bgra_bmp(frame, &mut writer),
        PixelFormat::Gray8 => write_gray_bmp(frame, &mut writer),
    }
}

fn write_bgra_bmp(frame: &Frame<'_>, writer: &mut impl Write) -> Result<()> {
    let width = checked_i32(frame.meta.size.width)?;
    let height = checked_i32(frame.meta.size.height)?;
    let row_bytes = checked_row_bytes(frame.meta.size.width, 4)?;
    let image_size = checked_image_size(row_bytes, frame.meta.size.height)?;
    write_bmp_header(writer, width, -height, 32, 0, image_size)?;

    let view = frame.view()?;
    for row in view.outer_iter() {
        for pixel in row.outer_iter() {
            writer.write_all(&[pixel[0], pixel[1], pixel[2], pixel[3]])?;
        }
    }
    Ok(())
}

fn write_gray_bmp(frame: &Frame<'_>, writer: &mut impl Write) -> Result<()> {
    let width = checked_i32(frame.meta.size.width)?;
    let height = checked_i32(frame.meta.size.height)?;
    let row_bytes = padded_row_bytes(frame.meta.size.width)?;
    let image_size = checked_image_size(row_bytes, frame.meta.size.height)?;
    write_bmp_header(writer, width, -height, 8, 256, image_size)?;

    for value in 0_u8..=255 {
        writer.write_all(&[value, value, value, 0])?;
    }

    let padding = row_bytes - frame.meta.size.width as usize;
    let pad = [0_u8; 3];
    let view = frame.view()?;
    for row in view.outer_iter() {
        for pixel in row.outer_iter() {
            writer.write_all(&[pixel[0]])?;
        }
        writer.write_all(&pad[..padding])?;
    }
    Ok(())
}

fn write_bmp_header(
    writer: &mut impl Write,
    width: i32,
    height: i32,
    bits_per_pixel: u16,
    palette_entries: u32,
    image_size: usize,
) -> Result<()> {
    let palette_bytes = palette_entries
        .checked_mul(4)
        .context("BMP palette byte count overflow")?;
    let pixel_offset = 14_u32
        .checked_add(40)
        .and_then(|offset| offset.checked_add(palette_bytes))
        .context("BMP pixel offset overflow")?;
    let file_size = pixel_offset
        .checked_add(u32::try_from(image_size)?)
        .context("BMP file size overflow")?;

    writer.write_all(b"BM")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(&0_u16.to_le_bytes())?;
    writer.write_all(&0_u16.to_le_bytes())?;
    writer.write_all(&pixel_offset.to_le_bytes())?;

    writer.write_all(&40_u32.to_le_bytes())?;
    writer.write_all(&width.to_le_bytes())?;
    writer.write_all(&height.to_le_bytes())?;
    writer.write_all(&1_u16.to_le_bytes())?;
    writer.write_all(&bits_per_pixel.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.write_all(&u32::try_from(image_size)?.to_le_bytes())?;
    writer.write_all(&0_i32.to_le_bytes())?;
    writer.write_all(&0_i32.to_le_bytes())?;
    writer.write_all(&palette_entries.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    Ok(())
}

fn checked_i32(value: u32) -> Result<i32> {
    i32::try_from(value).context("bitmap dimension exceeds i32")
}

fn checked_row_bytes(width: u32, bytes_per_pixel: usize) -> Result<usize> {
    (width as usize)
        .checked_mul(bytes_per_pixel)
        .context("bitmap row byte count overflow")
}

fn padded_row_bytes(width: u32) -> Result<usize> {
    let row = width as usize;
    row.checked_add(3)
        .map(|value| value & !3)
        .context("bitmap padded row byte count overflow")
}

fn checked_image_size(row_bytes: usize, height: u32) -> Result<usize> {
    row_bytes
        .checked_mul(height as usize)
        .context("bitmap image byte count overflow")
}

fn frame_json(frame: &Frame<'_>, file: &str) -> Value {
    json!({
        "file": file,
        "id": frame.meta.id,
        "sensor": sensor_kind_json(frame.meta.sensor),
        "size": size_json(frame.meta.size),
        "format": pixel_format_json(frame.format),
        "sequence": frame.meta.sequence,
        "camera_monotonic_us": frame.meta.timestamp.camera_monotonic_us,
    })
}

fn projection_json(projection: &HandProjectionOutput) -> Value {
    json!({
        "roi": roi_json(projection.roi),
        "landmarks": projection.landmarks.as_ref().map(landmarks_json),
        "depth": depth_estimate_json(&projection.depth),
    })
}

fn depth_estimate_json(depth: &LandmarkDepthEstimate) -> Value {
    json!({
        "anchor_depth_mm": depth.anchor_depth_mm,
        "anchor_landmark": depth.anchor_landmark,
        "closest_relative_z": depth.closest_relative_z,
        "landmark_depths_mm": depth.landmark_depths_mm,
        "used_depth_sample": depth.used_depth_sample,
    })
}

fn landmarks_json(landmarks: &HandLandmarks) -> Value {
    json!({
        "presence": landmarks.presence,
        "handedness": landmarks.handedness.map(handedness_json),
        "points": landmarks.points.iter().map(landmark_json).collect::<Vec<_>>(),
    })
}

fn landmark_json(point: &HandLandmark) -> Value {
    json!({
        "x": finite_f32_json(point.x),
        "y": finite_f32_json(point.y),
        "z": finite_f32_json(point.z),
    })
}

fn depth_sample_json(sample: DepthSample) -> Value {
    json!({
        "sequence": sample.sequence,
        "sensor_timestamp_us": sample.sensor_timestamp_us,
        "printed_at_ms": sample.printed_at_ms,
        "resolution": sample.resolution,
        "center_mm": sample.center_mm,
        "min_mm": sample.min_mm,
        "max_mm": sample.max_mm,
        "valid_zones": sample.valid_zones,
        "zones": sample.zones[..sample.zone_count].to_vec(),
    })
}

fn roi_json(roi: Option<RoiResult>) -> Option<Value> {
    roi.map(|roi| {
        json!({
            "rect": rect_json(roi.rect),
            "oriented_box": roi.oriented_box.map(|oriented_box| {
                json!({ "corners": oriented_box.corners })
            }),
        })
    })
}

fn rect_json(rect: Rect) -> Value {
    json!({
        "x": rect.x,
        "y": rect.y,
        "size": size_json(rect.size),
    })
}

fn size_json(size: Size) -> Value {
    json!({
        "width": size.width,
        "height": size.height,
    })
}

fn sensor_kind_json(sensor: SensorKind) -> &'static str {
    match sensor {
        SensorKind::Rgb => "rgb",
        SensorKind::Ir => "ir",
    }
}

fn pixel_format_json(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::Gray8 => "gray8",
        PixelFormat::Bgra8 => "bgra8",
    }
}

fn handedness_json(handedness: Handedness) -> &'static str {
    match handedness {
        Handedness::Left => "left",
        Handedness::Right => "right",
    }
}

fn finite_f32_json(value: f32) -> Option<f32> {
    value.is_finite().then_some(value)
}
