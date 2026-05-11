use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;
use tron::capture::{LitIrFrameStream, open_v4l_stream};
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{CameraOpenRequest, CaptureFormat, CheckerboardSpec, PixelFormat, SensorKind, Size};
use tron_core::capture::v4l::V4lUvcmMetadataSource;
use tron_core::pipeline::BufferedFrameSource;

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
    max_sync_delta_us: i64,

    /// Path for stereo calibration JSON output.
    #[arg(long, default_value = "calibration.json")]
    output: PathBuf,
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
    let ir_metadata_id = cli
        .ir_metadata_id
        .clone()
        .or_else(|| infer_metadata_node(&ir_info.id));
    let ir_metadata_id = ir_metadata_id.ok_or_else(|| {
        anyhow::anyhow!("--ir-metadata-id is required when IR node is not /dev/videoN")
    })?;
    let ir_metadata = V4lUvcmMetadataSource::open(&ir_metadata_id)?;
    let ir_stream = LitIrFrameStream::new(ir_stream, ir_metadata);
    eprintln!(
        "tron-calibration: opened ir {} {:?} {}x{} using lit-frame metadata {}",
        ir_info.id, ir_info.format, ir_info.size.width, ir_info.size.height, ir_metadata_id
    );

    let (rgb_info, rgb_stream) =
        open_v4l_stream(rgb_request(&cli), PixelFormat::from(cli.decode_format))?;
    anyhow::ensure!(
        rgb_info.format == CaptureFormat::Mjpeg,
        "calibration RGB feed currently requires MJPEG"
    );
    eprintln!(
        "tron-calibration: opened rgb {} {:?} {}x{}",
        rgb_info.id, rgb_info.format, rgb_info.size.width, rgb_info.size.height
    );

    eprintln!(
        "tron-calibration: press Space to capture a paired checkerboard sample; press C to calibrate and write {}",
        cli.output.display()
    );
    let rgb_stream = BufferedFrameSource::spawn("calibration-rgb", rgb_stream, 4);
    let ir_stream = BufferedFrameSource::spawn("calibration-ir", ir_stream, 4);
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

fn infer_metadata_node(video_node: &str) -> Option<String> {
    let number = video_node.strip_prefix("/dev/video")?.parse::<u32>().ok()?;
    Some(format!("/dev/video{}", number + 1))
}
