use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use memmap2::Mmap;
use opencv::core::{AlgorithmHint, Mat};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use tron_api::{
    CaptureFormat, Frame, FrameMeta, FrameSource, FrameTimestamp, IterableFrameSource,
    OpenedCameraInfo, OwnedFrame, PixelFormat, SensorKind, Size, TimestampSource,
};

const DEFAULT_FRAME_INTERVAL: Duration = Duration::from_millis(33);
const SUPPORTED_SUFFIXES: &[&str] = &["bmp", "jpg", "jpeg", "png", "ppm", "webp"];

#[derive(Clone, Debug)]
pub struct FromFileFrameSourceConfig {
    pub sensor: SensorKind,
    pub repeat: bool,
    pub frame_interval: Duration,
}

impl Default for FromFileFrameSourceConfig {
    fn default() -> Self {
        Self {
            sensor: SensorKind::Rgb,
            repeat: true,
            frame_interval: DEFAULT_FRAME_INTERVAL,
        }
    }
}

pub struct FromFileFrameSource {
    info: OpenedCameraInfo,
    paths: Vec<PathBuf>,
    repeat: bool,
    frame_interval: Duration,
    current_index: usize,
    next_id: u64,
    current: Option<OwnedFrame>,
}

impl FromFileFrameSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_config(path, FromFileFrameSourceConfig::default())
    }

    pub fn open_with_config(
        path: impl AsRef<Path>,
        config: FromFileFrameSourceConfig,
    ) -> Result<Self> {
        let root = path.as_ref();
        let paths = image_paths(root)?;
        let first_path = paths
            .first()
            .with_context(|| format!("no image files found in {}", root.display()))?;
        let first = decode_bgra_frame(
            first_path,
            frame_meta(
                0,
                config.sensor,
                Size {
                    width: 1,
                    height: 1,
                },
                config.frame_interval,
            ),
        )?;
        let size = first.meta.size;

        Ok(Self {
            info: OpenedCameraInfo {
                id: root.display().to_string(),
                sensor: config.sensor,
                // FrameSource still exposes camera-oriented info. File images are
                // decoded below and returned as Bgra8 frames.
                format: CaptureFormat::Mjpeg,
                size,
            },
            paths,
            repeat: config.repeat,
            frame_interval: config.frame_interval,
            current_index: 0,
            next_id: 0,
            current: None,
        })
    }
}

#[async_trait::async_trait]
impl FrameSource for FromFileFrameSource {
    fn info(&self) -> &OpenedCameraInfo {
        &self.info
    }

    async fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        if self.current_index >= self.paths.len() {
            if !self.repeat {
                return Ok(None);
            }
            self.current_index = 0;
        }

        let index = self.current_index;
        let path = &self.paths[index];

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.current_index += 1;
        let frame = decode_bgra_frame(
            path,
            frame_meta(id, self.info.sensor, self.info.size, self.frame_interval),
        )?;
        self.info.size = frame.meta.size;
        self.current = Some(frame);
        Ok(Some(
            self.current
                .as_ref()
                .expect("decoded frame was just stored")
                .as_frame(),
        ))
    }
}

impl IterableFrameSource for FromFileFrameSource {
    fn prev_frame(&mut self) -> Result<bool> {
        if self.paths.is_empty() {
            return Ok(false);
        }
        let len = self.paths.len();
        self.current_index = (self.current_index as i64 - 2).rem_euclid(len as i64) as usize;
        Ok(true)
    }
}

fn image_paths(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    anyhow::ensure!(
        path.is_dir(),
        "{} is not a file or directory",
        path.display()
    );
    let mut paths = fs::read_dir(path)
        .with_context(|| format!("read image source directory {}", path.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("read entries from {}", path.display()))?;
    paths.retain(|path| {
        path.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    SUPPORTED_SUFFIXES
                        .iter()
                        .any(|supported| extension.eq_ignore_ascii_case(supported))
                })
    });
    paths.sort();
    Ok(paths)
}

fn decode_bgra_frame(path: &Path, mut meta: FrameMeta) -> Result<OwnedFrame> {
    let file = File::open(path).with_context(|| format!("open image {}", path.display()))?;
    let mapped =
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmap image {}", path.display()))?;
    anyhow::ensure!(
        !mapped.is_empty(),
        "image {} is an empty file",
        path.display()
    );
    let encoded = Mat::from_slice::<u8>(&mapped)
        .with_context(|| format!("wrap mmap image bytes for {}", path.display()))?;
    let bgr = imgcodecs::imdecode(&encoded, imgcodecs::IMREAD_COLOR)
        .with_context(|| format!("decode image {}", path.display()))?;
    anyhow::ensure!(!bgr.empty(), "image {} decoded empty", path.display());

    let mut bgra = Mat::default();
    imgproc::cvt_color(
        &bgr,
        &mut bgra,
        imgproc::COLOR_BGR2BGRA,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .with_context(|| format!("convert image {} to BGRA", path.display()))?;
    anyhow::ensure!(
        bgra.is_continuous(),
        "decoded image {} is not contiguous",
        path.display()
    );

    let width = u32::try_from(bgra.cols()).context("decoded image width is negative")?;
    let height = u32::try_from(bgra.rows()).context("decoded image height is negative")?;
    meta.size = Size { width, height };
    let stride = width as usize * PixelFormat::Bgra8.channels();
    let data = bgra
        .data_bytes()
        .with_context(|| format!("read decoded image bytes for {}", path.display()))?
        .to_vec();
    anyhow::ensure!(
        data.len() >= stride * height as usize,
        "decoded image {} buffer is too small",
        path.display()
    );

    Ok(OwnedFrame {
        meta,
        format: PixelFormat::Bgra8,
        stride,
        data,
    })
}

fn frame_meta(id: u64, sensor: SensorKind, size: Size, interval: Duration) -> FrameMeta {
    let interval_us = interval.as_micros().min(i64::MAX as u128) as i64;
    FrameMeta {
        id,
        sensor,
        size,
        timestamp: FrameTimestamp {
            camera_monotonic_us: Some((id as i64).saturating_mul(interval_us)),
            source: TimestampSource::Driver,
            received_at: Instant::now(),
        },
        sequence: Some(id),
    }
}
