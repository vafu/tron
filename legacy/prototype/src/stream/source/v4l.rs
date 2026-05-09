use anyhow::{Context, Result};
use std::time::Instant;
use v4l::FourCC;
use v4l::buffer::{Flags, Metadata, Type};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::video::capture::Parameters;

use crate::stream::frame::{
    CaptureFormat, CapturedFrame, EncodedFormat, EncodedFrame, Frame, FrameMeta, FrameTimestamp,
    PixelFormat, TimestampSource,
};
use crate::stream::source::{FrameSource, SourceConfig};

pub struct V4lFrameSource {
    config: SourceConfig,
    stream: MmapStream<'static>,
    next_id: u64,
}

impl V4lFrameSource {
    pub fn open(mut config: SourceConfig) -> Result<Self> {
        anyhow::ensure!(config.buffers > 0, "V4L buffer count must be non-zero");

        let dev =
            Device::with_path(&config.path).with_context(|| format!("open {}", config.path))?;
        let mut fmt = dev.format()?;
        fmt.width = config.width;
        fmt.height = config.height;
        let requested_fourcc = FourCC::from(config.format);
        fmt.fourcc = requested_fourcc;
        let fmt = dev
            .set_format(&fmt)
            .with_context(|| format!("set V4L format on {}", config.path))?;
        anyhow::ensure!(
            fmt.fourcc == requested_fourcc,
            "V4L negotiated {} but requested {} on {}",
            fmt.fourcc,
            requested_fourcc,
            config.path
        );
        config.width = fmt.width;
        config.height = fmt.height;

        if let Some(fps) = config.fps {
            dev.set_params(&Parameters::with_fps(fps))
                .with_context(|| format!("set V4L frame interval on {}", config.path))?;
        }

        let stream = MmapStream::with_buffers(&dev, Type::VideoCapture, config.buffers)
            .with_context(|| {
                format!(
                    "create V4L mmap stream on {} with {} buffers",
                    config.path, config.buffers
                )
            })?;

        Ok(Self {
            config,
            stream,
            next_id: 0,
        })
    }
}

impl FrameSource for V4lFrameSource {
    fn next_frame(&mut self) -> Result<CapturedFrame<'_>> {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let SourceConfig {
            sensor,
            format,
            width,
            height,
            ..
        } = self.config;

        let (buf, v4l_meta) = self.stream.next().context("dequeue V4L frame")?;
        let used_len = (v4l_meta.bytesused as usize).min(buf.len());
        let data = if used_len > 0 { &buf[..used_len] } else { buf };
        let meta = frame_meta(id, sensor, width, height, v4l_meta);

        match format {
            CaptureFormat::Mjpeg => Ok(EncodedFrame {
                meta,
                format: EncodedFormat::Mjpeg,
                data,
            }
            .into()),
            CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => {
                let format = PixelFormat::try_from(format)?;
                Ok(Frame {
                    meta,
                    format,
                    stride: stride(format, width),
                    data,
                }
                .into())
            }
        }
    }
}

fn stride(format: PixelFormat, width: u32) -> usize {
    match format {
        PixelFormat::Gray8 => width as usize,
        PixelFormat::Yuyv422 => width as usize * 2,
        PixelFormat::Bgra8 => width as usize * 4,
    }
}

impl From<CaptureFormat> for FourCC {
    fn from(format: CaptureFormat) -> Self {
        match format {
            CaptureFormat::Mjpeg => FourCC::new(b"MJPG"),
            CaptureFormat::Gray8 => FourCC::new(b"GREY"),
            CaptureFormat::Yuyv422 => FourCC::new(b"YUYV"),
        }
    }
}

impl TryFrom<CaptureFormat> for PixelFormat {
    type Error = anyhow::Error;

    fn try_from(format: CaptureFormat) -> Result<Self> {
        match format {
            CaptureFormat::Gray8 => Ok(PixelFormat::Gray8),
            CaptureFormat::Yuyv422 => Ok(PixelFormat::Yuyv422),
            CaptureFormat::Mjpeg => {
                anyhow::bail!("MJPEG is encoded and cannot be converted to a pixel format")
            }
        }
    }
}

fn frame_meta(
    id: u64,
    sensor: crate::stream::frame::SensorKind,
    width: u32,
    height: u32,
    meta: &Metadata,
) -> FrameMeta {
    FrameMeta {
        id,
        sensor,
        width,
        height,
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
