use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tron::capture::open_v4l_stream;
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{CameraOpenRequest, CaptureFormat, PixelFormat, SensorKind};
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

    let (rgb_info, rgb_stream) =
        open_v4l_stream(rgb_request(&cli), PixelFormat::from(cli.decode_format))?;
    anyhow::ensure!(
        rgb_info.format == CaptureFormat::Mjpeg,
        "collector RGB feed currently requires MJPEG"
    );
    eprintln!(
        "collector: opened rgb {} {:?} {}x{}",
        rgb_info.id, rgb_info.format, rgb_info.size.width, rgb_info.size.height
    );

    let rgb = MirroredFrameSource::horizontal(rgb_stream);
    window::run(
        rgb,
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
    request
}
