use clap::{Args, ValueEnum};
use tron_api::{CameraOpenRequest, CameraSelector, CaptureFormat, PixelFormat, SensorKind, Size};

#[derive(Clone, Debug, Args)]
pub struct CameraArgs {
    /// Human-oriented camera selector matched by the selected camera backend.
    #[arg(long)]
    pub camera: Option<String>,

    /// Backend-native camera identifier. On V4L this is a path such as /dev/video51.
    #[arg(long, alias = "device")]
    pub camera_id: Option<String>,

    /// Sensor label attached to captured frame metadata.
    #[arg(long, value_enum, default_value = "rgb")]
    pub sensor: SensorArg,

    /// Requested capture format. If omitted, the backend keeps its default.
    #[arg(long, value_enum)]
    pub format: Option<CaptureFormatArg>,

    /// Requested capture size. If omitted, the backend keeps its default.
    #[arg(long, value_parser = parse_size)]
    pub size: Option<SizeArg>,

    /// Requested frame rate. If omitted, the backend keeps its default.
    #[arg(long, value_parser = parse_positive_u32)]
    pub fps: Option<u32>,

    /// Requested capture buffer count. If omitted, the backend chooses.
    #[arg(long, value_parser = parse_positive_u32)]
    pub buffers: Option<u32>,
}

impl CameraArgs {
    pub fn open_request(&self) -> CameraOpenRequest {
        CameraOpenRequest {
            selector: CameraSelector {
                id: self.camera_id.clone(),
                name: self.camera.clone(),
                sensor: self.sensor.into(),
            },
            format: self.format.map(Into::into),
            size: self.size.map(Into::into),
            fps: self.fps,
            buffers: self.buffers,
        }
    }

    pub fn requested_format(&self) -> Option<CaptureFormat> {
        self.format.map(Into::into)
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
pub struct SizeArg {
    pub width: u32,
    pub height: u32,
}

pub fn parse_size(value: &str) -> std::result::Result<SizeArg, String> {
    let Some((width, height)) = value.split_once('x') else {
        return Err(format!("invalid size {value:?}; expected WIDTHxHEIGHT"));
    };
    Ok(SizeArg {
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

impl From<SizeArg> for Size {
    fn from(value: SizeArg) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}
