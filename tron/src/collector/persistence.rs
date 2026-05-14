use std::fs::{File, create_dir};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use tron_api::{Frame, PixelFormat, Sink};

use crate::aggregate::Aggregate;

pub struct Persistence {
    root: PathBuf,
}

impl Persistence {
    pub fn new_tmp() -> Result<Self> {
        let root = create_tmp_dir()?;
        eprintln!("collector: IR persistence directory {}", root.display());
        Ok(Self { root })
    }
}

#[async_trait::async_trait(?Send)]
impl<'a> Sink<&'a Aggregate<'a>> for Persistence {
    async fn consume(&mut self, aggregate: &'a Aggregate<'a>) -> Result<()> {
        let path = self
            .root
            .join(format!("ir-{:010}.bmp", aggregate.ir.meta.id));
        write_bmp(&aggregate.ir, &path).context("write IR bitmap")
    }
}

fn create_tmp_dir() -> Result<PathBuf> {
    for attempt in 0..32_u32 {
        let root = std::env::temp_dir().join(format!("tron-{}", random_id(attempt)));
        match create_dir(&root) {
            Ok(()) => return Ok(root),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err).context("create collector temp persistence dir"),
        }
    }
    anyhow::bail!("create unique collector temp persistence dir")
}

fn random_id(attempt: u32) -> String {
    let mut bytes = [0_u8; 8];
    if let Ok(mut file) = File::open("/dev/urandom")
        && file.read_exact(&mut bytes).is_ok()
    {
        let value = u64::from_le_bytes(bytes);
        return format!("{value:016x}");
    }

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{:x}-{:x}-{:x}", process::id(), nanos, attempt)
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
