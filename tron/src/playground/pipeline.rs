use std::sync::Arc;
use std::time::Instant;

use crate::exposure_roi::{ClippedExposureRoiConfig, ClippedExposureRoiDetector};
use crate::metadata::{CameraStatsProcessor, PlaygroundMetadata};
use anyhow::Result;
use serde::Serialize;
use std::path::PathBuf;
use tron_api::{Frame, NoContext, OwnedFrame, Processor, RoiResult};
use tron_core::process::frame_diff::{
    FrameDiffConfig, FrameDiffMode, FrameDiffOutputPolicy, FrameDiffProcessor,
    FrameDiffReferencePolicy,
};
use tron_core::roi::hand::{HandRoiInput, HandRoiTracker, HandRoiTrackerConfig};
use tron_core::roi::mediapipe::{MediaPipeRoiConfig, MediaPipeRoiProcessor};
use tron_core::roi::opencv::{OpenCvRoiConfig, OpenCvRoiDetector};

pub struct PlaygroundInput {
    pub rgb: Option<Arc<OwnedFrame>>,
    pub ir: Option<Arc<OwnedFrame>>,
}

#[derive(Serialize)]
pub struct PlaygroundOutput<'a> {
    pub rgb: Option<Arc<OwnedFrame>>,
    pub ir_diff: Option<Frame<'a>>,
    pub depth_cue: Option<Frame<'a>>,
    pub roi: Option<RoiResult>,
    pub rgb_roi: Option<RoiResult>,
    pub exposure_roi: Option<RoiResult>,
    pub metadata: PlaygroundMetadata,
}

pub struct PlaygroundPipeline {
    rgb_stats: CameraStatsProcessor,
    ir_stats: CameraStatsProcessor,
    ambient_reject: FrameDiffProcessor,
    depth_cue: FrameDiffProcessor,
    roi: OpenCvRoiDetector,
    rgb_roi: Option<MediaPipeRoiProcessor>,
    hand_roi: HandRoiTracker,
    exposure_roi: ClippedExposureRoiDetector,
}

#[derive(Clone, Debug)]
pub struct PlaygroundPipelineConfig {
    pub roi_threshold: u8,
    pub exposure_roi_threshold: u8,
    pub rgb_mediapipe_model: Option<PathBuf>,
    pub rgb_mediapipe_min_score: f32,
    pub rgb_mediapipe_box_scale: f32,
}

impl Default for PlaygroundPipelineConfig {
    fn default() -> Self {
        Self {
            roi_threshold: 32,
            exposure_roi_threshold: 250,
            rgb_mediapipe_model: None,
            rgb_mediapipe_min_score: 0.75,
            rgb_mediapipe_box_scale: 2.6,
        }
    }
}

impl PlaygroundPipeline {
    pub fn new(config: PlaygroundPipelineConfig) -> Result<Self> {
        let rgb_roi = config
            .rgb_mediapipe_model
            .as_ref()
            .map(|model| {
                MediaPipeRoiProcessor::new(
                    model,
                    MediaPipeRoiConfig {
                        min_score: config.rgb_mediapipe_min_score,
                        box_scale: config.rgb_mediapipe_box_scale,
                        ..MediaPipeRoiConfig::default()
                    },
                )
            })
            .transpose()?;
        Ok(Self {
            rgb_stats: CameraStatsProcessor::new(),
            ir_stats: CameraStatsProcessor::new(),
            ambient_reject: FrameDiffProcessor::new(FrameDiffConfig {
                mode: FrameDiffMode::BrighterOnly,
                reference_policy: FrameDiffReferencePolicy::AlternatingPair,
                output_policy: FrameDiffOutputPolicy::MeaningfulOnly,
                min_output_value: 8,
                min_output_pixels: 64,
            }),
            depth_cue: FrameDiffProcessor::new(FrameDiffConfig {
                mode: FrameDiffMode::Absolute,
                reference_policy: FrameDiffReferencePolicy::PreviousFrame,
                output_policy: FrameDiffOutputPolicy::MeaningfulOnly,
                min_output_value: 8,
                min_output_pixels: 64,
            }),
            roi: OpenCvRoiDetector::new(OpenCvRoiConfig {
                threshold: config.roi_threshold,
                min_area: 96,
                padding: 24,
                ..OpenCvRoiConfig::default()
            }),
            rgb_roi,
            hand_roi: HandRoiTracker::new(HandRoiTrackerConfig::default()),
            exposure_roi: ClippedExposureRoiDetector::new(ClippedExposureRoiConfig {
                threshold: config.exposure_roi_threshold,
                ..ClippedExposureRoiConfig::default()
            }),
        })
    }

    pub fn process(&mut self, input: PlaygroundInput) -> Result<PlaygroundOutput<'_>> {
        let now = Instant::now();
        let metadata = PlaygroundMetadata {
            rgb: self.rgb_stats.process(input.rgb.as_deref(), now),
            ir: self.ir_stats.process(input.ir.as_deref(), now),
            rgb_ir_delta_us: camera_delta_us(input.rgb.as_deref(), input.ir.as_deref()),
        };
        let ambient_rejected = input
            .ir
            .as_ref()
            .map(|frame| self.ambient_reject.process(frame.as_frame()))
            .transpose()?;
        let depth_cue = ambient_rejected
            .map(|frame| self.depth_cue.process(frame))
            .transpose()?;
        let roi_candidates = input
            .ir
            .as_ref()
            .map(|frame| self.roi.detect_candidates(frame.as_frame()))
            .transpose()?
            .unwrap_or_default();
        let roi = self.hand_roi.process(
            HandRoiInput {
                candidates: &roi_candidates,
                motion: depth_cue,
            },
            NoContext,
        )?;
        let rgb_roi = if let Some(processor) = self.rgb_roi.as_mut() {
            input
                .rgb
                .as_ref()
                .map(|frame| processor.process(frame.as_frame(), NoContext))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let exposure_roi = input
            .ir
            .as_ref()
            .map(|frame| self.exposure_roi.detect(frame.as_frame(), roi))
            .transpose()?
            .flatten();

        Ok(PlaygroundOutput {
            rgb: input.rgb,
            ir_diff: ambient_rejected,
            depth_cue,
            roi,
            rgb_roi,
            exposure_roi,
            metadata,
        })
    }
}

fn camera_delta_us(rgb: Option<&OwnedFrame>, ir: Option<&OwnedFrame>) -> Option<i64> {
    let rgb = rgb?;
    let ir = ir?;
    Some(rgb.meta.timestamp.camera_monotonic_us? - ir.meta.timestamp.camera_monotonic_us?)
}
