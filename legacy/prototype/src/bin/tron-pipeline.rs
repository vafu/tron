#[allow(dead_code)]
#[path = "../stream/mod.rs"]
mod stream;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use std::time::{Duration, Instant};
use stream::decode::mjpeg::TurboMjpegDecoder;
use stream::frame::{CaptureFormat, PixelFormat, SensorKind};
use stream::render::{FrameStats, RenderSink, TextStatsSink};
use stream::source::SourceConfig;
use stream::source::v4l::V4lFrameSource;
use stream::{DecodeStream, FrameStream, PassthroughStream};

#[derive(Debug, Parser)]
#[command(name = "tron-pipeline")]
#[command(about = "Minimal capture/decode pipeline probe")]
struct Cli {
    #[arg(long, default_value = "/dev/video53")]
    device: String,

    #[arg(long, value_enum, default_value = "rgb")]
    sensor: SensorArg,

    #[arg(long, value_enum, default_value = "mjpg")]
    format: CaptureFormatArg,

    #[arg(long, value_parser = parse_size, default_value = "1280x720")]
    size: Size,

    #[arg(long, value_parser = parse_positive_u32, default_value = "30")]
    fps: u32,

    #[arg(long, value_parser = parse_positive_u32, default_value = "4")]
    buffers: u32,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SensorArg {
    Rgb,
    Ir,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CaptureFormatArg {
    #[value(alias = "mjpeg")]
    Mjpg,
    #[value(alias = "gray", alias = "grey")]
    Gray8,
    #[value(alias = "yuyv")]
    Yuyv422,
}

#[derive(Clone, Copy, Debug)]
struct Size {
    width: u32,
    height: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let capture_format = CaptureFormat::from(cli.format);
    let source_cfg = source_config(&cli);
    eprintln!(
        "tron-pipeline: opening {} {:?} {:?} {}x{}{} buffers={}",
        source_cfg.path,
        source_cfg.sensor,
        source_cfg.format,
        source_cfg.width,
        source_cfg.height,
        source_cfg
            .fps
            .map(|fps| format!("@{fps}fps"))
            .unwrap_or_default(),
        source_cfg.buffers
    );

    let source = V4lFrameSource::open(source_cfg)?;
    match capture_format {
        CaptureFormat::Mjpeg => {
            let decoder = TurboMjpegDecoder::new(PixelFormat::Bgra8)?;
            run_stream(DecodeStream::new(source, decoder))
        }
        CaptureFormat::Gray8 | CaptureFormat::Yuyv422 => run_stream(PassthroughStream::new(source)),
    }
}

fn source_config(cli: &Cli) -> SourceConfig {
    SourceConfig::new(
        cli.device.clone(),
        cli.sensor.into(),
        cli.format.into(),
        cli.size.width,
        cli.size.height,
    )
    .with_fps(cli.fps)
    .with_buffers(cli.buffers)
}

fn run_stream(mut stream: impl FrameStream) -> Result<()> {
    let mut renderer = TextStatsSink::new(Duration::from_secs(2));

    loop {
        let start = Instant::now();
        let frame = stream.next_frame()?;
        renderer.submit(
            frame,
            FrameStats {
                acquire_us: start.elapsed().as_micros() as u64,
            },
        )?;
    }
}

fn parse_size(value: &str) -> std::result::Result<Size, String> {
    let Some((width, height)) = value.split_once('x') else {
        return Err(format!("invalid size {value:?}; expected WIDTHxHEIGHT"));
    };
    Ok(Size {
        width: parse_positive_u32(width)?,
        height: parse_positive_u32(height)?,
    })
}

fn parse_positive_u32(value: &str) -> std::result::Result<u32, String> {
    match value.parse::<u32>() {
        Ok(v) if v > 0 => Ok(v),
        _ => Err(format!(
            "invalid value {value:?}; expected a positive integer"
        )),
    }
}

impl From<SensorArg> for SensorKind {
    fn from(value: SensorArg) -> Self {
        match value {
            SensorArg::Rgb => Self::Rgb,
            SensorArg::Ir => Self::Ir,
        }
    }
}

impl From<CaptureFormatArg> for CaptureFormat {
    fn from(value: CaptureFormatArg) -> Self {
        match value {
            CaptureFormatArg::Mjpg => Self::Mjpeg,
            CaptureFormatArg::Gray8 => Self::Gray8,
            CaptureFormatArg::Yuyv422 => Self::Yuyv422,
        }
    }
}
