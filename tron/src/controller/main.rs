use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use tron::capture::open_v4l_stream;
use tron::config::{CameraArgs, PixelFormatArg};
use tron_api::{
    CameraOpenRequest, CaptureFormat, FrameSource, PixelFormat, SensorKind, Size,
    spawn_event_channels,
};
use tron_core::capture::FromFileFrameSource;
use tron_core::pointer::{
    AbsolutePointerProducer, JoystickPointerProducer, RelativePointerProducer,
};
use tron_core::render::http::HttpJsonSink;
use tron_core::roi::mediapipe::{MediaPipeHandLandmarkConfig, MediaPipeRoiConfig};
use tron_core::transform::{FpsThrottledFrameSource, MirroredFrameSource};

mod capture;
mod linux_pointer;
mod pipeline;
mod pointer_sink;
mod renderer;
mod runtime;
mod window;

#[derive(Debug, Parser)]
#[command(name = "controller")]
#[command(about = "RGB hand tracking controller prototype")]
struct Cli {
    #[command(flatten)]
    camera: CameraArgs,

    /// Pixel format produced when decoding MJPEG.
    #[arg(long, value_enum, default_value = "bgra8")]
    decode_format: PixelFormatArg,

    /// Replay RGB frames from an image file or directory instead of opening a camera.
    #[arg(long)]
    input: Option<PathBuf>,

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

    /// Pointer producer used by the controller demo.
    #[arg(long, value_enum, default_value = "absolute")]
    pointer_mode: PointerMode,

    /// Emit pointer events through a Linux uinput virtual mouse.
    #[arg(long)]
    linux_pointer: bool,

    /// Show the controller preview/debug window.
    #[arg(long)]
    preview: bool,

    /// Scale normalized pointer delta to Linux relative pointer units.
    #[arg(long, default_value_t = 1400.0)]
    linux_pointer_units_per_delta: f64,

    /// Limit controller frame processing to this FPS.
    #[arg(long)]
    max_fps: Option<f64>,

    /// Directory for RGB frame captures when landmarked pinch state changes.
    #[arg(long)]
    capture_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PointerMode {
    Absolute,
    Joystick,
    Relative,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    run(Cli::parse())
}

fn run(cli: Cli) -> Result<()> {
    if let Some(camera) = &cli.camera.camera {
        eprintln!("controller: resolving --camera {camera:?}");
    }

    let pipeline_config = pipeline::PipelineConfig {
        palm_model: cli.rgb_mediapipe_model.clone(),
        palm: MediaPipeRoiConfig {
            min_score: cli.rgb_mediapipe_min_score,
            box_scale: cli.rgb_mediapipe_box_scale,
        },
        landmark_model: cli.rgb_mediapipe_landmark_model.clone(),
        landmarks: MediaPipeHandLandmarkConfig {
            min_presence: cli.rgb_mediapipe_landmark_min_presence,
            roi_scale: cli.rgb_mediapipe_landmark_roi_scale,
        },
    };

    let pipeline = if let Some(input) = cli.input.as_ref() {
        let stream = FromFileFrameSource::open(input)?;
        let info = stream.info();
        eprintln!(
            "controller: replaying rgb {} {}x{}",
            info.id, info.size.width, info.size.height
        );
        pipeline::ControllerTicker::Replay(pipeline::ReplayPipeline::new(pipeline::Pipeline::new(
            stream,
            pipeline_config,
        )?))
    } else {
        let (info, stream) =
            open_v4l_stream(rgb_request(&cli), PixelFormat::from(cli.decode_format))?;
        anyhow::ensure!(
            info.format == CaptureFormat::Mjpeg,
            "controller RGB feed currently requires MJPEG"
        );
        eprintln!(
            "controller: opened rgb {} {:?} {}x{}",
            info.id, info.format, info.size.width, info.size.height
        );
        let stream = MirroredFrameSource::horizontal(stream);
        let stream: Box<dyn tron_api::FrameSource + Send> = if let Some(max_fps) = cli.max_fps {
            Box::new(FpsThrottledFrameSource::new(stream, max_fps)?)
        } else {
            Box::new(stream)
        };
        pipeline::ControllerTicker::Live(pipeline::Pipeline::new(stream, pipeline_config)?)
    };
    let pointer = match cli.pointer_mode {
        PointerMode::Absolute => spawn_event_channels(AbsolutePointerProducer::default(), 8, 32),
        PointerMode::Joystick => spawn_event_channels(JoystickPointerProducer::default(), 8, 32),
        PointerMode::Relative => spawn_event_channels(RelativePointerProducer::default(), 8, 32),
    };
    let mut sinks = runtime::ComboSink::new();
    let metadata = HttpJsonSink::bind_available(("127.0.0.1", 8765), 100)?;
    eprintln!(
        "controller: metadata http://{}/metadata",
        metadata.local_addr()
    );
    sinks.push_box(Box::new(metadata));
    if let Some(path) = cli.capture_dir {
        sinks.push_box(Box::new(capture::FrameImageCaptureSink::new(path)?));
    }
    let mut pointer_sinks = runtime::PointerSink::new();
    if cli.linux_pointer {
        pointer_sinks.push_box(Box::new(linux_pointer::LinuxPointerSink::new(
            linux_pointer::LinuxPointerConfig {
                units_per_delta: cli.linux_pointer_units_per_delta,
            },
        )?));
    }

    if cli.preview {
        window::run(pipeline, pointer, sinks, pointer_sinks)
    } else {
        runtime::run(pipeline, pointer, sinks, pointer_sinks)
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
