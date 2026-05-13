use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::time::Instant;
use tron_api::{
    CameraOpenRequest, CameraOpener, CameraSelector, CaptureFormat, Frame, FrameMeta, FrameSource,
    FrameTimestamp, OpenedCameraInfo, PixelFormat, SensorKind, Size, TimestampSource,
};
use v4l::FourCC;
use v4l::buffer::{Flags, Metadata, Type};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::video::capture::Parameters;

use crate::decode::mjpeg::TurboMjpegDecoder;
use crate::decode::{EncodedFormat, EncodedFrame, FrameDecoder};

const DEFAULT_DEVICE: &str = "/dev/video53";
const DEFAULT_BUFFERS: u32 = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct V4lCameraOpener {
    decoded_mjpeg_format: Option<PixelFormat>,
}

impl V4lCameraOpener {
    pub fn with_decoded_mjpeg_format(decoded_mjpeg_format: PixelFormat) -> Self {
        Self {
            decoded_mjpeg_format: Some(decoded_mjpeg_format),
        }
    }
}

impl CameraOpener for V4lCameraOpener {
    type Source = V4lFrameSource;

    fn open(&self, request: CameraOpenRequest) -> Result<Self::Source> {
        let path = resolve_device(&request.selector)?;
        let dev = Device::with_path(&path).with_context(|| format!("open {path}"))?;
        let mut fmt = dev.format()?;
        if let Some(Size { width, height }) = request.size {
            fmt.width = width;
            fmt.height = height;
        }
        if let Some(format) = request.format {
            fmt.fourcc = fourcc(format);
        }

        let requested_fourcc = request.format.map(fourcc);
        let requested_size = request.size;
        let fmt = dev
            .set_format(&fmt)
            .with_context(|| format!("set V4L format on {path}"))?;
        if let Some(requested_fourcc) = requested_fourcc {
            anyhow::ensure!(
                fmt.fourcc == requested_fourcc,
                "V4L negotiated {} but requested {} on {}",
                fmt.fourcc,
                requested_fourcc,
                path
            );
        }
        if let Some(requested_size) = requested_size {
            anyhow::ensure!(
                fmt.width == requested_size.width && fmt.height == requested_size.height,
                "V4L negotiated {}x{} but requested {}x{} on {}",
                fmt.width,
                fmt.height,
                requested_size.width,
                requested_size.height,
                path
            );
        }
        let format = capture_format(fmt.fourcc)?;
        let size = Size {
            width: fmt.width,
            height: fmt.height,
        };

        if let Some(fps) = request.fps {
            dev.set_params(&Parameters::with_fps(fps))
                .with_context(|| format!("set V4L frame interval on {path}"))?;
        }

        let buffers = request.buffers.unwrap_or(DEFAULT_BUFFERS);
        anyhow::ensure!(buffers > 0, "V4L buffer count must be non-zero");
        let stream = MmapStream::with_buffers(&dev, Type::VideoCapture, buffers)
            .with_context(|| format!("create V4L mmap stream on {path} with {buffers} buffers"))?;

        let decoder = match format {
            CaptureFormat::Mjpeg => Some(TurboMjpegDecoder::new(
                self.decoded_mjpeg_format.unwrap_or(PixelFormat::Bgra8),
            )?),
            CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => None,
        };

        Ok(V4lFrameSource {
            info: OpenedCameraInfo {
                id: path,
                sensor: request.selector.sensor,
                format,
                size,
            },
            stream,
            decoder,
            dropped_mjpeg_buffers: 0,
            next_id: 0,
        })
    }
}

pub struct V4lFrameSource {
    info: OpenedCameraInfo,
    stream: MmapStream<'static>,
    decoder: Option<TurboMjpegDecoder>,
    dropped_mjpeg_buffers: u32,
    next_id: u64,
}

#[async_trait::async_trait]
impl FrameSource for V4lFrameSource {
    fn info(&self) -> &OpenedCameraInfo {
        &self.info
    }

    async fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        let (buf, v4l_meta) = self.stream.next().context("dequeue V4L frame")?;
        let used_len = (v4l_meta.bytesused as usize).min(buf.len());
        if used_len == 0 {
            return Ok(None);
        }

        let data = &buf[..used_len];
        let meta = frame_meta(id, self.info.sensor, self.info.size, v4l_meta);

        match self.info.format {
            CaptureFormat::Mjpeg => {
                let decoder = self
                    .decoder
                    .as_mut()
                    .context("V4L MJPEG source has no decoder")?;
                let header = match decoder.read_header(data) {
                    Ok(header) => header,
                    Err(err) => {
                        note_dropped_mjpeg_buffer(
                            &self.info,
                            &mut self.dropped_mjpeg_buffers,
                            v4l_meta,
                            &format!("{err:#}"),
                        );
                        return Ok(None);
                    }
                };
                if header.width as u32 != self.info.size.width
                    || header.height as u32 != self.info.size.height
                {
                    note_dropped_mjpeg_buffer(
                        &self.info,
                        &mut self.dropped_mjpeg_buffers,
                        v4l_meta,
                        &format!(
                            "payload dimensions {}x{} do not match negotiated {}x{}",
                            header.width,
                            header.height,
                            self.info.size.width,
                            self.info.size.height
                        ),
                    );
                    return Ok(None);
                }
                decoder
                    .decode(EncodedFrame {
                        meta,
                        format: EncodedFormat::Mjpeg,
                        data,
                    })
                    .map(Some)
            }
            CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => {
                let format = PixelFormat::try_from(self.info.format)?;
                Ok(Some(Frame::new(
                    meta,
                    format,
                    stride(format, self.info.size.width),
                    data,
                )?))
            }
        }
    }
}

fn note_dropped_mjpeg_buffer(
    info: &OpenedCameraInfo,
    dropped_count: &mut u32,
    meta: &Metadata,
    reason: &str,
) {
    *dropped_count = dropped_count.saturating_add(1);
    if *dropped_count == 1 || *dropped_count % 120 == 0 {
        eprintln!(
            "v4l: dropped MJPEG buffer from {} seq={} bytesused={} flags={:?}: {}",
            info.id, meta.sequence, meta.bytesused, meta.flags, reason
        );
    }
}

#[derive(Clone, Debug)]
struct VideoDevice {
    path: String,
    name: String,
    index: Option<u32>,
    number: u32,
}

pub fn resolve_device(selector: &CameraSelector) -> Result<String> {
    if let Some(id) = &selector.id {
        return Ok(id.clone());
    }
    let Some(name) = &selector.name else {
        return Ok(DEFAULT_DEVICE.to_string());
    };
    resolve_named_device(name, selector.sensor)
        .with_context(|| format!("resolve camera {name:?} for {:?}", selector.sensor))
}

fn resolve_named_device(selector: &str, sensor: SensorKind) -> Result<String> {
    let selector = selector.trim().to_ascii_lowercase();
    anyhow::ensure!(!selector.is_empty(), "camera name cannot be empty");

    let mut matches = video_devices()?
        .into_iter()
        .filter(|device| device.name.to_ascii_lowercase().contains(&selector))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !matches.is_empty(),
        "no V4L devices matched camera name {selector:?}"
    );
    matches.sort_by_key(|device| device.number);
    let capture_matches = matches
        .iter()
        .filter(|device| device.index == Some(0))
        .collect::<Vec<_>>();
    let usable = if capture_matches.is_empty() {
        matches.iter().collect::<Vec<_>>()
    } else {
        capture_matches
    };

    let hinted = usable
        .iter()
        .find(|device| sensor_name_score(&device.name, sensor) > 0);
    let fallback = match sensor {
        SensorKind::Rgb => usable.first().copied(),
        SensorKind::Ir => usable.get(1).copied().or_else(|| usable.first().copied()),
    };

    hinted
        .copied()
        .or(fallback)
        .map(|device| device.path.clone())
        .context("camera match disappeared")
}

fn video_devices() -> Result<Vec<VideoDevice>> {
    let root = Path::new("/sys/class/video4linux");
    let entries = fs::read_dir(root).with_context(|| format!("read {}", root.display()))?;
    let mut devices = Vec::new();
    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name();
        let node = file_name.to_string_lossy();
        if !node.starts_with("video") {
            continue;
        }
        let name_path = entry.path().join("name");
        let name = fs::read_to_string(&name_path)
            .with_context(|| format!("read {}", name_path.display()))?
            .trim()
            .to_string();
        let index = fs::read_to_string(entry.path().join("index"))
            .ok()
            .and_then(|index| index.trim().parse::<u32>().ok());
        let number = node
            .strip_prefix("video")
            .and_then(|number| number.parse::<u32>().ok())
            .unwrap_or(u32::MAX);
        devices.push(VideoDevice {
            path: format!("/dev/{node}"),
            name,
            index,
            number,
        });
    }
    Ok(devices)
}

fn sensor_name_score(name: &str, sensor: SensorKind) -> u8 {
    let name = name.to_ascii_lowercase();
    let is_ir = name.contains(" ir")
        || name.contains("ir ")
        || name.contains("infrared")
        || name.contains("depth");
    match sensor {
        SensorKind::Ir if is_ir => 2,
        SensorKind::Rgb if !is_ir => 1,
        _ => 0,
    }
}

fn stride(format: PixelFormat, width: u32) -> usize {
    match format {
        PixelFormat::Gray8 => width as usize,
        PixelFormat::Yuyv422 => width as usize * 2,
        PixelFormat::Bgra8 => width as usize * 4,
    }
}

fn fourcc(format: CaptureFormat) -> FourCC {
    match format {
        CaptureFormat::Mjpeg => FourCC::new(b"MJPG"),
        CaptureFormat::Gray8 => FourCC::new(b"GREY"),
        CaptureFormat::Yuyv422 => FourCC::new(b"YUYV"),
    }
}

fn capture_format(fourcc: FourCC) -> Result<CaptureFormat> {
    match &fourcc.repr {
        b"MJPG" => Ok(CaptureFormat::Mjpeg),
        b"GREY" => Ok(CaptureFormat::Gray8),
        b"YUYV" => Ok(CaptureFormat::Yuyv422),
        _ => anyhow::bail!("unsupported negotiated V4L format {fourcc}"),
    }
}

fn frame_meta(id: u64, sensor: SensorKind, size: Size, meta: &Metadata) -> FrameMeta {
    FrameMeta {
        id,
        sensor,
        size,
        timestamp: FrameTimestamp {
            camera_monotonic_us: camera_monotonic_us(meta),
            source: timestamp_source(meta.flags),
            received_at: Instant::now(),
        },
        sequence: Some(meta.sequence as u64),
    }
}

fn camera_monotonic_us(meta: &Metadata) -> Option<i64> {
    let timestamp_type = meta.flags.bits() & Flags::TIMESTAMP_MASK.bits();
    if timestamp_type != Flags::TIMESTAMP_MONOTONIC.bits() {
        return None;
    }
    Some(meta.timestamp.sec as i64 * 1_000_000 + meta.timestamp.usec as i64)
}

fn timestamp_source(flags: Flags) -> TimestampSource {
    let source = flags.bits() & Flags::TSTAMP_SRC_MASK.bits();
    if source == Flags::TSTAMP_SRC_SOE.bits() {
        TimestampSource::StartOfExposure
    } else if source == Flags::TSTAMP_SRC_EOF.bits() {
        TimestampSource::EndOfFrame
    } else {
        TimestampSource::Unknown
    }
}
