use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tron::capture::{WindowsHelloV4lConfig, open_windows_hello_v4l_streams};
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{CameraOpenRequest, CaptureFormat, PixelFormat, SensorKind, Size};
use tron_core::roi::mediapipe::{MediaPipeHandLandmarkConfig, MediaPipeRoiConfig};
use tron_core::transform::MirroredFrameSource;

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

    /// Backend-native IR metadata node. Defaults to the next /dev/videoN after the IR node.
    #[arg(long)]
    ir_metadata_id: Option<String>,

    /// Maximum allowed RGB/IR timestamp delta for a synchronized pair, in microseconds.
    #[arg(long, default_value_t = 20_000)]
    max_sync_delta_us: u64,

    /// ONNX model path for RGB MediaPipe ROI detection.
    #[arg(long, default_value = "models/hand_detector/model.onnx")]
    rgb_mediapipe_model: PathBuf,

    /// ONNX model path for RGB MediaPipe hand landmark extraction.
    #[arg(long, default_value = "models/hand_landmark/hand_landmark.onnx")]
    rgb_mediapipe_landmark_model: PathBuf,

    /// Minimum MediaPipe palm detector confidence for RGB ROI.
    #[arg(long, default_value_t = 0.75)]
    rgb_mediapipe_min_score: f32,

    /// Minimum MediaPipe landmark presence confidence.
    #[arg(long, default_value_t = 0.9)]
    rgb_mediapipe_landmark_min_presence: f32,

    /// MediaPipe-style scale applied to the palm detector rect before landmarks.
    #[arg(long, default_value_t = 2.6)]
    rgb_mediapipe_box_scale: f32,

    /// MediaPipe-style scale applied to the landmark tracking rect.
    #[arg(long, default_value_t = 1.2)]
    rgb_mediapipe_landmark_roi_scale: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    run(Cli::parse()).await
}

async fn run(cli: Cli) -> Result<()> {
    if let Some(camera) = &cli.camera.camera {
        eprintln!("collector: resolving --camera {camera:?}");
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

    let rgb = MirroredFrameSource::horizontal(streams.rgb_stream);
    let ir = MirroredFrameSource::horizontal(streams.ir_stream);
    window::run(
        rgb,
        ir,
        cli.max_sync_delta_us,
        cli.rgb_mediapipe_model,
        MediaPipeRoiConfig {
            min_score: cli.rgb_mediapipe_min_score,
            box_scale: cli.rgb_mediapipe_box_scale,
        },
        cli.rgb_mediapipe_landmark_model,
        MediaPipeHandLandmarkConfig {
            min_presence: cli.rgb_mediapipe_landmark_min_presence,
            roi_scale: cli.rgb_mediapipe_landmark_roi_scale,
        },
    )
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
    request.size = None;
    request
}
