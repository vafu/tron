use clap::{Args, ValueEnum};
use tron_api::{CaptureFormat, PixelFormat, SensorKind};
use tron_core::capture::v4l::V4lSourceConfig;

#[derive(Clone, Debug, Args)]
pub struct CameraArgs {
    /// Human-oriented camera selector. Name resolution is intentionally not wired yet.
    #[arg(long)]
    pub camera: Option<String>,

    /// V4L capture node.
    #[arg(long, default_value = "/dev/video53")]
    pub device: String,

    /// Sensor label attached to captured frame metadata.
    #[arg(long, value_enum, default_value = "rgb")]
    pub sensor: SensorArg,

    /// Requested capture format.
    #[arg(long, value_enum, default_value = "mjpg")]
    pub format: CaptureFormatArg,

    /// Requested capture size.
    #[arg(long, value_parser = parse_size, default_value = "1280x720")]
    pub size: Size,

    /// Requested frame rate.
    #[arg(long, value_parser = parse_positive_u32, default_value = "30")]
    pub fps: u32,

    /// Requested V4L mmap buffer count.
    #[arg(long, value_parser = parse_positive_u32, default_value = "4")]
    pub buffers: u32,
}

impl CameraArgs {
    pub fn capture_format(&self) -> CaptureFormat {
        self.format.into()
    }

    pub fn to_v4l_config(&self) -> V4lSourceConfig {
        V4lSourceConfig::new(
            self.device.clone(),
            self.sensor.into(),
            self.format.into(),
            self.size.width,
            self.size.height,
        )
        .with_fps(self.fps)
        .with_buffers(self.buffers)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum SensorArg {
    Rgb,
    Ir,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum CaptureFormatArg {
    #[value(alias = "mjpeg")]
    Mjpg,
    #[value(alias = "gray", alias = "grey")]
    Gray8,
    #[value(alias = "yuyv")]
    Yuyv422,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum PixelFormatArg {
    Gray8,
    Bgra8,
    Yuyv422,
}

#[derive(Clone, Copy, Debug)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

pub fn parse_size(value: &str) -> std::result::Result<Size, String> {
    let Some((width, height)) = value.split_once('x') else {
        return Err(format!("invalid size {value:?}; expected WIDTHxHEIGHT"));
    };
    Ok(Size {
        width: parse_positive_u32(width)?,
        height: parse_positive_u32(height)?,
    })
}

pub fn parse_positive_u32(value: &str) -> std::result::Result<u32, String> {
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

impl From<PixelFormatArg> for PixelFormat {
    fn from(value: PixelFormatArg) -> Self {
        match value {
            PixelFormatArg::Gray8 => Self::Gray8,
            PixelFormatArg::Bgra8 => Self::Bgra8,
            PixelFormatArg::Yuyv422 => Self::Yuyv422,
        }
    }
}
