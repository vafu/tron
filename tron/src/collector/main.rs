use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tron::capture::{WindowsHelloV4lConfig, open_windows_hello_v4l_streams};
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{
    CameraOpenRequest, CaptureFormat, CheckerboardStereoCalibration, DepthSource, PixelFormat,
    SensorKind, Size,
};
use tron_core::capture::{
    LitIrFrameStream, V4lUvcmMetadataSource, v4l_control::V4lCameraRoiControl,
};
use tron_core::projection::CheckerboardDepthProjection;
use tron_core::projection::{HandProjectionConfig, HandProjectionProcessor};
use tron_core::render::http::HttpJsonSink;
use tron_core::roi::camera::CameraRoiFollowConfig;
use tron_core::roi::mediapipe::{MediaPipeHandLandmarkConfig, MediaPipeRoiConfig};
use tron_core::sensor::vl53l5cx_serial::Vl53l5cxSerialDepthSource;
use tron_core::transform::MirroredFrameSource;

mod aggregate;
mod camera_roi;
mod persistence;
mod pipeline;
mod renderer;
mod sink;
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
    #[arg(long, default_value = "models/google_hand_detector/model.onnx")]
    rgb_mediapipe_model: PathBuf,

    /// ONNX model path for RGB MediaPipe hand landmark extraction.
    #[arg(long, default_value = "models/google_hand_landmark/hand_landmark.onnx")]
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

    /// Stereo calibration JSON used to project the RGB ROI onto the IR frame.
    #[arg(long)]
    stereo_calibration: Option<PathBuf>,

    /// Assumed RGB-camera hand depth, in millimeters, for RGB-to-IR ROI projection.
    #[arg(long, default_value_t = 700.0)]
    roi_projection_depth_mm: f64,

    /// Millimeters per MediaPipe relative-z unit for per-landmark projection.
    #[arg(long, default_value_t = 1000.0)]
    landmark_z_scale_mm: f64,

    /// VL53L5CX serial port to use as live ROI projection depth.
    #[arg(long)]
    tof_serial: Option<PathBuf>,

    /// Baud rate for --tof-serial.
    #[arg(long, default_value_t = 115200)]
    tof_baud: u32,

    /// Serial read timeout for --tof-serial, in milliseconds.
    #[arg(long, default_value_t = 1)]
    tof_timeout_ms: u64,

    /// Drive the IR camera exposure ROI from the center of the RGB palm detection.
    #[arg(long)]
    camera_roi_from_palm: bool,

    /// Minimum edge size for the palm-following camera exposure ROI rectangle.
    #[arg(long, default_value_t = 40)]
    camera_roi_min_edge: u32,

    /// Minimum interval between palm-following camera exposure ROI updates. Set 0 to disable throttling.
    #[arg(long, default_value_t = 100)]
    camera_roi_update_ms: u64,

    /// Disable palm-following camera exposure ROI update throttling.
    #[arg(long)]
    no_camera_roi_throttle: bool,
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
    anyhow::ensure!(
        cli.roi_projection_depth_mm >= 0.0,
        "--roi-projection-depth-mm must be non-negative"
    );
    anyhow::ensure!(
        cli.landmark_z_scale_mm >= 0.0,
        "--landmark-z-scale-mm must be non-negative"
    );
    anyhow::ensure!(
        cli.tof_serial.is_none() || cli.stereo_calibration.is_some(),
        "--tof-serial requires --stereo-calibration"
    );

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

    let hand_projection = if let Some(path) = cli.stereo_calibration.as_ref() {
        let file = std::fs::File::open(path)
            .map(std::io::BufReader::new)
            .map_err(anyhow::Error::from)?;
        let calibration: CheckerboardStereoCalibration = serde_json::from_reader(file)?;
        eprintln!(
            "collector: projecting RGB ROI onto IR using {} at {:.1} mm",
            path.display(),
            cli.roi_projection_depth_mm
        );
        Some(HandProjectionProcessor::new(
            CheckerboardDepthProjection::new(calibration),
            HandProjectionConfig {
                fallback_depth_mm: cli.roi_projection_depth_mm,
                landmark_z_scale_mm: cli.landmark_z_scale_mm,
                source_mirrored_x: true,
                target_mirrored_x: true,
            },
        )?)
    } else {
        None
    };
    let roi_depth_source: Option<Box<dyn DepthSource + Send>> =
        if let Some(port) = cli.tof_serial.as_ref() {
            eprintln!(
                "collector: using TOF depth from {} at {} baud",
                port.display(),
                cli.tof_baud
            );
            Some(Box::new(Vl53l5cxSerialDepthSource::open(
                port,
                cli.tof_baud,
                Duration::from_millis(cli.tof_timeout_ms),
            )?))
        } else {
            None
        };

    let ir_device_id = streams.ir_device_id.clone();
    let ir_metadata_id = streams.ir_metadata_id.clone();
    let rgb = MirroredFrameSource::horizontal(streams.rgb_stream);
    let ir_metadata = V4lUvcmMetadataSource::open(&ir_metadata_id)?;
    let ir = MirroredFrameSource::horizontal(LitIrFrameStream::new(streams.ir_stream, ir_metadata));
    let camera_roi_update_interval = camera_roi_update_interval(&cli);
    let pipeline = pipeline::Pipeline::new(
        rgb,
        ir,
        pipeline::PipelineConfig {
            max_sync_delta_us: cli.max_sync_delta_us,
            palm_model: cli.rgb_mediapipe_model,
            palm: MediaPipeRoiConfig {
                min_score: cli.rgb_mediapipe_min_score,
                box_scale: cli.rgb_mediapipe_box_scale,
            },
            landmark_model: cli.rgb_mediapipe_landmark_model,
            landmarks: MediaPipeHandLandmarkConfig {
                min_presence: cli.rgb_mediapipe_landmark_min_presence,
                roi_scale: cli.rgb_mediapipe_landmark_roi_scale,
            },
            camera_roi: cli.camera_roi_from_palm.then_some(CameraRoiFollowConfig {
                min_edge: cli.camera_roi_min_edge,
            }),
            hand_projection,
            depth_source: roi_depth_source,
        },
    )?;
    let persistence = sink::ToggleSink::new(persistence::Persistence::new_tmp()?, false);
    let mut sinks = sink::ComboSink::new();
    if cli.camera_roi_from_palm {
        sinks.push_box(Box::new(camera_roi::CameraRoiSink::new(
            Box::new(V4lCameraRoiControl::open(&ir_device_id)?),
            camera_roi_update_interval,
        )));
    }
    let metadata = HttpJsonSink::bind_available(("127.0.0.1", 8765), 100)?;
    eprintln!(
        "collector: metadata http://{}/metadata",
        metadata.local_addr()
    );
    sinks.push_box(Box::new(metadata));
    window::run(pipeline, sinks, persistence)
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
