use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;
use tron::capture::{WindowsHelloV4lConfig, open_windows_hello_v4l_streams};
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{
    CameraOpenRequest, CaptureFormat, CheckerboardSpec, CheckerboardStereoCalibration,
    DepthProjectionMap, PixelFormat, SensorKind, Size,
};
use tron_core::projection::{
    CheckerboardDepthProjection, DepthProjectionMapSource, StaticProjectionMapSource,
};
use tron_core::sensor::vl53l5cx_serial::Vl53l5cxSerialDepthSource;

mod check;
mod latency;
mod renderer;
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

    /// Backend-native IR metadata node. Defaults to the next /dev/videoN after the IR node.
    #[arg(long)]
    ir_metadata_id: Option<String>,

    /// Number of inner checkerboard corners horizontally.
    #[arg(long, default_value_t = 7)]
    checkerboard_cols: u32,

    /// Number of inner checkerboard corners vertically.
    #[arg(long, default_value_t = 7)]
    checkerboard_rows: u32,

    /// Physical square size in millimeters.
    #[arg(long, default_value_t = 25.0)]
    checkerboard_square_mm: f64,

    /// Minimum paired checkerboard samples before allowing calibration.
    #[arg(long, default_value_t = 8)]
    min_samples: usize,

    /// Maximum allowed RGB/IR timestamp delta for a synchronized pair, in microseconds.
    #[arg(long, default_value_t = 20_000)]
    max_sync_delta_us: u64,

    /// Path for stereo calibration JSON output.
    #[arg(long, default_value = "calibration.json")]
    output: PathBuf,

    /// Load a calibration JSON and display RGB with a translucent IR overlay.
    #[arg(long)]
    check: Option<PathBuf>,

    /// Assumed RGB-camera depth, in millimeters, for --check RGB-to-IR overlay projection.
    #[arg(long, default_value_t = 700.0)]
    check_depth_mm: f64,

    /// VL53L5CX serial port to use as live projection depth for --check.
    #[arg(long)]
    check_tof_serial: Option<PathBuf>,

    /// Baud rate for --check-tof-serial.
    #[arg(long, default_value_t = 115200)]
    check_tof_baud: u32,

    /// Serial read timeout for --check-tof-serial, in milliseconds.
    #[arg(long, default_value_t = 1)]
    check_tof_timeout_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
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

    let streams = open_windows_hello_v4l_streams(WindowsHelloV4lConfig {
        rgb_request: rgb_request(&cli),
        ir_request: ir_request(&cli),
        ir_metadata_id: cli.ir_metadata_id.clone(),
        decoded_rgb_format: PixelFormat::from(cli.decode_format),
        decoded_ir_format: PixelFormat::Bgra8,
    })?;
    anyhow::ensure!(
        streams.rgb_info.format == CaptureFormat::Mjpeg,
        "calibration RGB feed currently requires MJPEG"
    );
    eprintln!(
        "tron-calibration: opened rgb {} {:?} {}x{}",
        streams.rgb_info.id,
        streams.rgb_info.format,
        streams.rgb_info.size.width,
        streams.rgb_info.size.height
    );

    let ir_stream = streams.ir_stream;
    let rgb_stream = streams.rgb_stream;
    eprintln!(
        "tron-calibration: opened ir {} {:?} {}x{}",
        streams.ir_info.id,
        streams.ir_info.format,
        streams.ir_info.size.width,
        streams.ir_info.size.height
    );

    if let Some(path) = cli.check.as_ref() {
        let file = std::fs::File::open(path)
            .map(std::io::BufReader::new)
            .map_err(anyhow::Error::from)?;
        let calibration: CheckerboardStereoCalibration = serde_json::from_reader(file)?;
        eprintln!(
            "tron-calibration: checking {} with RGB+IR overlay",
            path.display()
        );
        let projection = CheckerboardDepthProjection::new(calibration);
        let config = check::CalibrationCheckConfig {
            max_sync_delta_us: cli.max_sync_delta_us,
        };
        if let Some(port) = cli.check_tof_serial.as_ref() {
            eprintln!(
                "tron-calibration: using TOF depth from {} at {} baud",
                port.display(),
                cli.check_tof_baud
            );
            let depth_source = Vl53l5cxSerialDepthSource::open(
                port,
                cli.check_tof_baud,
                Duration::from_millis(cli.check_tof_timeout_ms),
            )?;
            let map_source = DepthProjectionMapSource::new(projection, depth_source)?;
            return check::run(rgb_stream, ir_stream, map_source, config);
        }

        let map = projection.map(cli.check_depth_mm)?;
        return check::run(
            rgb_stream,
            ir_stream,
            StaticProjectionMapSource::new(map),
            config,
        );
    }

    eprintln!(
        "tron-calibration: press Space to capture a paired checkerboard sample; press C to calibrate and write {}",
        cli.output.display()
    );
    window::run(
        rgb_stream,
        ir_stream,
        window::CalibrationRunConfig {
            checkerboard: checkerboard_spec(&cli),
            min_samples: cli.min_samples,
            max_sync_delta_us: cli.max_sync_delta_us,
            output: cli.output,
        },
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

fn rgb_request(cli: &Cli) -> CameraOpenRequest {
    let mut request = cli.camera.open_request();
    request.selector.sensor = SensorKind::Rgb;
    if request.format.is_none() {
        request.format = Some(CaptureFormat::Mjpeg);
    }
    if request.size.is_none() {
        request.size = Some(Size {
            width: 640,
            height: 480,
        });
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
