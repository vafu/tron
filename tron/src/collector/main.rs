use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tron::capture::{WindowsHelloV4lConfig, open_windows_hello_v4l_streams};
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{CameraOpenRequest, CaptureFormat, PixelFormat, SensorKind};
use tron_core::capture::{LitIrFrameStream, StereoFrameSource, V4lUvcmMetadataSource};
use tron_core::roi::mediapipe::MediaPipeRoiConfig;

mod renderer;
mod window;

#[derive(Debug, Parser)]
#[command(name = "collector")]
#[command(about = "RGB/IR data collection view with RGB MediaPipe ROI")]
struct Cli {
    #[command(flatten)]
    camera: CameraArgs,

    /// Pixel format produced when decoding MJPEG.
    #[arg(long, value_enum, default_value = "bgra8")]
    decode_format: PixelFormatArg,

    /// Backend-native IR camera identifier. On V4L this is a path such as /dev/video51.
    #[arg(long)]
    ir_camera_id: Option<String>,

    /// Maximum allowed RGB/IR timestamp delta for a synchronized pair, in microseconds.
    #[arg(long, default_value_t = 20_000)]
    max_sync_delta_us: u64,

    /// ONNX model path for RGB MediaPipe ROI detection.
    #[arg(long, default_value = "models/hand_detector/model.onnx")]
    rgb_mediapipe_model: PathBuf,

    /// Minimum MediaPipe palm detector confidence for RGB ROI.
    #[arg(long, default_value_t = 0.75)]
    rgb_mediapipe_min_score: f32,

    /// Scale applied around the raw MediaPipe palm detector box before rendering ROI.
    #[arg(long, default_value_t = 1.0)]
    rgb_mediapipe_box_scale: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    run(Cli::parse()).await
}

async fn run(cli: Cli) -> Result<()> {
    if let Some(camera) = &cli.camera.camera {
        eprintln!("collector: resolving --camera {camera:?}");
    }

    let streams = open_windows_hello_v4l_streams(WindowsHelloV4lConfig {
        rgb_request: rgb_request(&cli),
        ir_request: ir_request(&cli),
        ir_metadata_id: None,
        decoded_rgb_format: PixelFormat::from(cli.decode_format),
        decoded_ir_format: PixelFormat::Bgra8,
    })?;
    anyhow::ensure!(
        streams.rgb_info.format == CaptureFormat::Mjpeg,
        "collector RGB feed currently requires MJPEG"
    );
    eprintln!(
        "collector: opened rgb {} {:?} {}x{}",
        streams.rgb_info.id,
        streams.rgb_info.format,
        streams.rgb_info.size.width,
        streams.rgb_info.size.height
    );
    eprintln!(
        "collector: opened ir {} {:?} {}x{}",
        streams.ir_info.id,
        streams.ir_info.format,
        streams.ir_info.size.width,
        streams.ir_info.size.height
    );

    let ir_metadata = V4lUvcmMetadataSource::open(&streams.ir_metadata_id)?;
    let ir = LitIrFrameStream::new(streams.ir_stream, ir_metadata);
    let frames = StereoFrameSource::new(streams.rgb_stream, ir, cli.max_sync_delta_us);
    window::run(
        frames,
        cli.rgb_mediapipe_model,
        MediaPipeRoiConfig {
            min_score: cli.rgb_mediapipe_min_score,
            box_scale: cli.rgb_mediapipe_box_scale,
        },
    )
}

fn rgb_request(cli: &Cli) -> CameraOpenRequest {
    let mut request = cli.camera.open_request();
    request.selector.sensor = SensorKind::Rgb;
    if request.format.is_none() {
        request.format = Some(CaptureFormat::Mjpeg);
    }
    request
}

fn ir_request(cli: &Cli) -> CameraOpenRequest {
    let mut request = cli.camera.open_request();
    request.selector.sensor = SensorKind::Ir;
    request.selector.id = cli.ir_camera_id.clone();
    request.format = None;
    request.size = None;
    request
}
