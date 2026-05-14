use std::path::PathBuf;

use anyhow::Result;
use tron_api::{DepthSource, FrameSource, NoContext, Processor};
use tron_core::StereoFrameSource;
use tron_core::projection::{
    CheckerboardDepthProjection, HandProjectionInput, HandProjectionProcessor,
};
use tron_core::roi::landmark::{LandmarkRoiInput, LandmarkRoiProcessor};
use tron_core::roi::mediapipe::{
    MediaPipeHandLandmarkConfig, MediaPipeHandLandmarkInput, MediaPipeHandLandmarkProcessor,
    MediaPipeRoiConfig, MediaPipeRoiProcessor,
};

use crate::aggregate::Aggregate;

pub struct PipelineConfig {
    pub max_sync_delta_us: u64,
    pub palm_model: PathBuf,
    pub palm: MediaPipeRoiConfig,
    pub landmark_model: PathBuf,
    pub landmarks: MediaPipeHandLandmarkConfig,
    pub hand_projection: Option<HandProjectionProcessor<CheckerboardDepthProjection>>,
    pub depth_source: Option<Box<dyn DepthSource + Send>>,
}

pub trait Tick {
    fn tick(&mut self) -> Result<Option<Aggregate<'_>>>;
}

pub struct Pipeline<R, I> {
    frames: StereoFrameSource<R, I>,
    palm: MediaPipeRoiProcessor,
    landmarks: MediaPipeHandLandmarkProcessor,
    landmark_roi: LandmarkRoiProcessor,
    hand_projection: Option<HandProjectionProcessor<CheckerboardDepthProjection>>,
    depth_source: Option<Box<dyn DepthSource + Send>>,
}

impl<R, I> Pipeline<R, I>
where
    R: FrameSource + Send,
    I: FrameSource + Send,
{
    pub fn new(rgb: R, ir: I, config: PipelineConfig) -> Result<Self> {
        let landmark_roi = LandmarkRoiProcessor::new(config.landmarks.roi_scale);
        Ok(Self {
            frames: StereoFrameSource::new(rgb, ir, config.max_sync_delta_us),
            palm: MediaPipeRoiProcessor::new(config.palm_model, config.palm)?,
            landmarks: MediaPipeHandLandmarkProcessor::new(
                config.landmark_model,
                config.landmarks,
            )?,
            landmark_roi,
            hand_projection: config.hand_projection,
            depth_source: config.depth_source,
        })
    }

    async fn tick_async(&mut self) -> Result<Option<Aggregate<'_>>> {
        let Some(pair) = self.frames.next_pair().await? else {
            return Ok(None);
        };
        let rgb = pair.left;
        let ir = pair.right;

        let palm_roi = self.palm.process(rgb, NoContext)?;
        let landmarks = self.landmarks.process(
            MediaPipeHandLandmarkInput {
                frame: rgb,
                roi: palm_roi,
            },
            NoContext,
        )?;

        if let Some(ref landmarks) = landmarks {
            let valid_count = landmarks
                .points
                .iter()
                .filter(|point| point.x.is_finite())
                .count();
            tracing::info!("Detected {} valid landmarks", valid_count);
        }

        let landmark_roi = self.landmark_roi.process(
            LandmarkRoiInput {
                landmarks: landmarks.as_ref(),
                frame_size: rgb.meta.size,
            },
            NoContext,
        )?;
        if let Some(ref roi) = landmark_roi {
            tracing::info!("Landmark ROI: {:?}", roi.rect);
        }

        let rgb_roi = landmark_roi.or(palm_roi);
        let depth_sample = match self.depth_source.as_mut() {
            Some(depth_source) => {
                depth_source
                    .depth_at(rgb.meta.timestamp.received_at)
                    .await?
            }
            None => None,
        };
        let projection = match self.hand_projection.as_mut() {
            Some(hand_projection) => Some(hand_projection.process(
                HandProjectionInput {
                    roi: rgb_roi,
                    landmarks: landmarks.as_ref(),
                    depth_sample,
                    source_size: rgb.meta.size,
                    target_size: ir.meta.size,
                },
                NoContext,
            )?),
            None => None,
        };

        Ok(Some(Aggregate {
            rgb,
            ir,
            sync_delta_us: pair.delta_us,
            palm_roi,
            landmarks,
            rgb_roi,
            depth_sample,
            projection,
        }))
    }
}

impl<R, I> Tick for Pipeline<R, I>
where
    R: FrameSource + Send,
    I: FrameSource + Send,
{
    fn tick(&mut self) -> Result<Option<Aggregate<'_>>> {
        pollster::block_on(self.tick_async())
    }
}
