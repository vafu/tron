use std::sync::Arc;
use std::time::Instant;

use crate::metadata::{CameraStatsProcessor, PlaygroundMetadata};
use anyhow::Result;
use tron_api::{Frame, OwnedFrame};
use tron_core::process::frame_diff::{
    FrameDiffConfig, FrameDiffMode, FrameDiffOutputPolicy, FrameDiffProcessor,
    FrameDiffReferencePolicy,
};

pub struct PlaygroundInput {
    pub rgb: Option<Arc<OwnedFrame>>,
    pub ir: Option<Arc<OwnedFrame>>,
}

pub struct PlaygroundOutput<'a> {
    pub rgb: Option<Arc<OwnedFrame>>,
    pub ir_diff: Option<Frame<'a>>,
    pub depth_cue: Option<Frame<'a>>,
    pub metadata: PlaygroundMetadata,
}

pub struct PlaygroundPipeline {
    rgb_stats: CameraStatsProcessor,
    ir_stats: CameraStatsProcessor,
    ambient_reject: FrameDiffProcessor,
    depth_cue: FrameDiffProcessor,
}

impl PlaygroundPipeline {
    pub fn new() -> Self {
        Self {
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
        }
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

        Ok(PlaygroundOutput {
            rgb: input.rgb,
            ir_diff: ambient_rejected,
            depth_cue,
            metadata,
        })
    }
}

fn camera_delta_us(rgb: Option<&OwnedFrame>, ir: Option<&OwnedFrame>) -> Option<i64> {
    let rgb = rgb?;
    let ir = ir?;
    Some(rgb.meta.timestamp.camera_monotonic_us? - ir.meta.timestamp.camera_monotonic_us?)
}
