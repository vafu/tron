use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::time::Instant;
use tron_api::{
    CameraOpenRequest, CameraOpener, CameraSelector, CaptureFormat, CapturedFrame, EncodedFormat,
    EncodedFrame, Frame, FrameMeta, FrameSize, FrameSource, FrameTimestamp, OpenedCameraInfo,
    PixelFormat, SensorKind, TimestampSource,
};
use v4l::FourCC;
use v4l::buffer::{Flags, Metadata, Type};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::video::capture::Parameters;

const DEFAULT_DEVICE: &str = "/dev/video53";
const DEFAULT_BUFFERS: u32 = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct V4lCameraOpener;

impl CameraOpener for V4lCameraOpener {
    type Source = V4lFrameSource;

    fn open(&self, request: CameraOpenRequest) -> Result<Self::Source> {
        let path = resolve_device(&request.selector)?;
        let dev = Device::with_path(&path).with_context(|| format!("open {path}"))?;
        let mut fmt = dev.format()?;
        if let Some(FrameSize { width, height }) = request.size {
            fmt.width = width;
            fmt.height = height;
        }
        if let Some(format) = request.format {
            fmt.fourcc = fourcc(format);
        }

        let requested_fourcc = request.format.map(fourcc);
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
        let format = capture_format(fmt.fourcc)?;
        let size = FrameSize {
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

        Ok(V4lFrameSource {
            info: OpenedCameraInfo {
                id: path,
                sensor: request.selector.sensor,
                format,
                size,
            },
            stream,
            next_id: 0,
        })
    }
}

pub struct V4lFrameSource {
    info: OpenedCameraInfo,
    stream: MmapStream<'static>,
    next_id: u64,
}

impl FrameSource for V4lFrameSource {
    fn info(&self) -> &OpenedCameraInfo {
        &self.info
    }

    fn next_frame(&mut self) -> Result<Option<CapturedFrame<'_>>> {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        let (buf, v4l_meta) = self.stream.next().context("dequeue V4L frame")?;
        let used_len = (v4l_meta.bytesused as usize).min(buf.len());
        let data = if used_len > 0 { &buf[..used_len] } else { buf };
        let meta = frame_meta(id, self.info.sensor, self.info.size, v4l_meta);

        match self.info.format {
            CaptureFormat::Mjpeg => Ok(Some(
                EncodedFrame {
                    meta,
                    format: EncodedFormat::Mjpeg,
                    data,
                }
                .into(),
            )),
            CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => {
                let format = PixelFormat::try_from(self.info.format)?;
                Ok(Some(
                    Frame {
                        meta,
                        format,
                        stride: stride(format, self.info.size.width),
                        data,
                    }
                    .into(),
                ))
            }
        }
    }
}

#[derive(Clone, Debug)]
struct VideoDevice {
    path: String,
    name: String,
    index: Option<u32>,
    number: u32,
}

fn resolve_device(selector: &CameraSelector) -> Result<String> {
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

fn frame_meta(id: u64, sensor: SensorKind, size: FrameSize, meta: &Metadata) -> FrameMeta {
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
