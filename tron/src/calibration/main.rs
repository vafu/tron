use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;
use tron::capture::open_v4l_stream;
use tron::config::{CameraArgs, PixelFormatArg};
use tron::latest::LatestFrameSource;
use tron_api::{CameraOpenRequest, CaptureFormat, CheckerboardSpec, PixelFormat, SensorKind, Size};

mod latency;
mod presenter;
mod window;

#[derive(Debug, Parser)]
#[command(name = "tron-calibration")]
#[command(about = "RGB/IR calibration capture view")]
struct Cli {
    #[command(flatten)]
    camera: CameraArgs,

    /// Pixel format produced when decoding MJPEG.
    #[arg(long, value_enum, default_value = "bgra8")]
    decode_format: PixelFormatArg,

    /// Backend-native IR camera identifier. On V4L this is a path such as /dev/video51.
    #[arg(long)]
    ir_camera_id: Option<String>,

    /// Number of inner checkerboard corners horizontally.
    #[arg(long, default_value_t = 7)]
    checkerboard_cols: u32,

    /// Number of inner checkerboard corners vertically.
    #[arg(long, default_value_t = 7)]
    checkerboard_rows: u32,

    /// Physical square size in millimeters.
    #[arg(long, default_value_t = 25.0)]
    checkerboard_square_mm: f64,

    /// Minimum interval between checkerboard searches per feed. Set to 0 to search every new frame.
    #[arg(long, default_value_t = 250)]
    checkerboard_detect_ms: u64,
}

fn main() -> Result<()> {
    init_tracing();
    run(Cli::parse())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn,calibration::latency=info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

fn run(cli: Cli) -> Result<()> {
    if let Some(camera) = &cli.camera.camera {
        eprintln!("tron-calibration: resolving --camera {camera:?}");
    }

    let (ir_info, ir_stream) = open_v4l_stream(ir_request(&cli), PixelFormat::Bgra8)?;
    eprintln!(
        "tron-calibration: opened ir {} {:?} {}x{}",
        ir_info.id, ir_info.format, ir_info.size.width, ir_info.size.height
    );

    let (rgb_info, rgb_stream) = open_v4l_stream(
        rgb_request(&cli, ir_info.size),
        PixelFormat::from(cli.decode_format),
    )?;
    anyhow::ensure!(
        rgb_info.format == CaptureFormat::Mjpeg,
        "calibration RGB feed currently requires MJPEG"
    );
    eprintln!(
        "tron-calibration: opened rgb {} {:?} {}x{}",
        rgb_info.id, rgb_info.format, rgb_info.size.width, rgb_info.size.height
    );

    let rgb = LatestFrameSource::spawn("calibration-rgb", rgb_stream);
    let ir = LatestFrameSource::spawn("calibration-ir", ir_stream);
    window::run(
        rgb,
        ir,
        checkerboard_spec(&cli),
        Duration::from_millis(cli.checkerboard_detect_ms),
    )
}

fn checkerboard_spec(cli: &Cli) -> CheckerboardSpec {
    CheckerboardSpec {
        inner_corners: Size {
            width: cli.checkerboard_cols,
            height: cli.checkerboard_rows,
        },
        square_size_mm: cli.checkerboard_square_mm,
    }
}

fn rgb_request(cli: &Cli, default_size: Size) -> CameraOpenRequest {
    let mut request = cli.camera.open_request();
    request.selector.sensor = SensorKind::Rgb;
    if request.format.is_none() {
        request.format = Some(CaptureFormat::Mjpeg);
    }
    if request.size.is_none() {
        request.size = Some(default_size);
    }
    request
}

fn ir_request(cli: &Cli) -> CameraOpenRequest {
    let mut request = cli.camera.open_request();
    request.selector.sensor = SensorKind::Ir;
    request.selector.id = cli.ir_camera_id.clone();
    request.format = None;
    request
}
