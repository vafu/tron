use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Duration;
use tron::capture::{WindowsHelloV4lConfig, open_windows_hello_v4l_streams};
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{CameraOpenRequest, CaptureFormat, PixelFormat, SensorKind};
use tron_core::capture::v4l_control::V4lCameraRoiControl;
use tron_core::render::http::HttpJsonSink;

use crate::camera_roi::{CameraRoiConfig, CameraRoiDriver};
use crate::pipeline::PlaygroundPipelineConfig;

mod camera_roi;
mod exposure_roi;
mod metadata;
mod pipeline;
mod renderer;
mod window;

#[derive(Debug, Parser)]
#[command(name = "tron-playground")]
#[command(about = "Composable playground for capture/decode/process/render experiments")]
struct Cli {
    #[command(flatten)]
    camera: CameraArgs,

    /// Pixel format produced when decoding MJPEG.
    #[arg(long, value_enum, default_value = "bgra8")]
    decode_format: PixelFormatArg,

    /// Backend-native IR camera identifier. On V4L this is a path such as /dev/video51.
    #[arg(long)]
    ir_camera_id: Option<String>,

    /// Local HTTP port for live metadata.
    #[arg(long, default_value_t = 8787)]
    metadata_port: u16,

    /// Disable the local HTTP metadata endpoint.
    #[arg(long)]
    no_metadata_http: bool,

    /// Binary threshold for ROI detection on the ambient-rejected IR frame.
    #[arg(long, default_value_t = 32)]
    roi_threshold: u8,

    /// Raw IR pixel threshold used to find clipped regions for camera exposure ROI.
    #[arg(long, default_value_t = 250)]
    exposure_roi_threshold: u8,

    /// Drive the camera exposure ROI from the detected IR ROI.
    #[arg(long)]
    camera_roi_from_detection: bool,

    /// Minimum edge size for the camera exposure ROI rectangle.
    #[arg(long, default_value_t = 40)]
    camera_roi_min_edge: u32,

    /// Minimum interval between camera exposure ROI updates. Set 0 to disable throttling.
    #[arg(long, default_value_t = 100)]
    camera_roi_update_ms: u64,

    /// Disable camera exposure ROI update throttling.
    #[arg(long)]
    no_camera_roi_throttle: bool,

    /// Run the MediaPipe/Qualcomm palm detector on the RGB frame and render its ROI.
    #[arg(long)]
    rgb_mediapipe_roi: bool,

    /// ONNX model path for RGB MediaPipe ROI detection.
    #[arg(long, default_value = "models/google_hand_detector/model.onnx")]
    rgb_mediapipe_model: PathBuf,

    /// Minimum MediaPipe palm detector confidence for RGB ROI.
    #[arg(long, default_value_t = 0.75)]
    rgb_mediapipe_min_score: f32,

    /// Fingertip-direction scale applied to the oriented MediaPipe palm ROI.
    #[arg(long, default_value_t = 1.0)]
    rgb_mediapipe_box_scale: f32,
}

fn main() -> Result<()> {
    run(Cli::parse())
}

fn run(cli: Cli) -> Result<()> {
    if let Some(camera) = &cli.camera.camera {
        eprintln!(
            "tron-playground: resolving --camera {camera:?} for {:?}",
            cli.camera.sensor
        );
    }

    let streams = open_windows_hello_v4l_streams(WindowsHelloV4lConfig {
        rgb_request: rgb_request(&cli),
        ir_request: ir_request(&cli),
        ir_metadata_id: None,
        decoded_rgb_format: PixelFormat::from(cli.decode_format),
        decoded_ir_format: PixelFormat::Bgra8,
    })?;
    eprintln!(
        "tron-playground: opened rgb {} {:?} {}x{}",
        streams.rgb_info.id,
        streams.rgb_info.format,
        streams.rgb_info.size.width,
        streams.rgb_info.size.height
    );
    anyhow::ensure!(
        streams.rgb_info.format == CaptureFormat::Mjpeg,
        "playground RGB feed currently requires MJPEG"
    );

    eprintln!(
        "tron-playground: opened ir {} {:?} {}x{}",
        streams.ir_info.id,
        streams.ir_info.format,
        streams.ir_info.size.width,
        streams.ir_info.size.height
    );
    let ir_device_id = streams.ir_device_id.clone();

    let rgb_latest = tron::latest::LatestFrameSource::spawn("rgb", Box::new(streams.rgb_stream));
    let ir_latest = tron::latest::LatestFrameSource::spawn("ir", Box::new(streams.ir_stream));
    let camera_roi = if cli.camera_roi_from_detection {
        Some(CameraRoiDriver::new(
            CameraRoiConfig {
                min_edge: cli.camera_roi_min_edge,
                update_interval: camera_roi_update_interval(&cli),
            },
            Box::new(V4lCameraRoiControl::open(&ir_device_id)?),
        ))
    } else {
        None
    };
    let mut sinks = window::ComboSink::new();
    if !cli.no_metadata_http {
        let metadata = HttpJsonSink::bind_available(("127.0.0.1", cli.metadata_port), 20)?;
        eprintln!(
            "tron-playground: metadata http://{}/metadata",
            metadata.local_addr()
        );
        sinks.push_box(Box::new(metadata));
    }
    window::run(
        rgb_latest,
        ir_latest,
        sinks,
        camera_roi,
        PlaygroundPipelineConfig {
            roi_threshold: cli.roi_threshold,
            exposure_roi_threshold: cli.exposure_roi_threshold,
            rgb_mediapipe_model: cli
                .rgb_mediapipe_roi
                .then_some(cli.rgb_mediapipe_model.clone()),
            rgb_mediapipe_min_score: cli.rgb_mediapipe_min_score,
            rgb_mediapipe_box_scale: cli.rgb_mediapipe_box_scale,
        },
    )
}

fn camera_roi_update_interval(cli: &Cli) -> Option<Duration> {
    if cli.no_camera_roi_throttle || cli.camera_roi_update_ms == 0 {
        None
    } else {
        Some(Duration::from_millis(cli.camera_roi_update_ms))
    }
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
