use std::path::PathBuf;

use anyhow::Result;
use tron_api::{Frame, FrameSource, NoContext, Processor, RoiResult};
use tron_core::roi::landmark::{LandmarkRoiInput, LandmarkRoiProcessor};
use tron_core::roi::mediapipe::{
    HandLandmarks, MediaPipeHandLandmarkConfig, MediaPipeHandLandmarkInput,
    MediaPipeHandLandmarkProcessor, MediaPipeRoiConfig, MediaPipeRoiProcessor,
};

pub struct PipelineConfig {
    pub palm_model: PathBuf,
    pub palm: MediaPipeRoiConfig,
    pub landmark_model: PathBuf,
    pub landmarks: MediaPipeHandLandmarkConfig,
}

pub struct ControllerFrame<'a> {
    pub rgb: Frame<'a>,
    pub palm_roi: Option<RoiResult>,
    pub landmarks: Option<HandLandmarks>,
    pub rgb_roi: Option<RoiResult>,
}

impl ControllerFrame<'_> {
    pub fn frame_id(&self) -> u64 {
        self.rgb.meta.id
    }
}

pub trait Tick {
    fn tick(&mut self) -> Result<Option<ControllerFrame<'_>>>;
}

pub struct Pipeline<S> {
    source: S,
    palm: MediaPipeRoiProcessor,
    landmarks: MediaPipeHandLandmarkProcessor,
    landmark_roi: LandmarkRoiProcessor,
}

impl<S> Pipeline<S>
where
    S: FrameSource + Send,
{
    pub fn new(source: S, config: PipelineConfig) -> Result<Self> {
        let landmark_roi = LandmarkRoiProcessor::new(config.landmarks.roi_scale);
        Ok(Self {
            source,
            palm: MediaPipeRoiProcessor::new(config.palm_model, config.palm)?,
            landmarks: MediaPipeHandLandmarkProcessor::new(
                config.landmark_model,
                config.landmarks,
            )?,
            landmark_roi,
        })
    }

    async fn tick_async(&mut self) -> Result<Option<ControllerFrame<'_>>> {
        let Some(rgb) = self.source.next_frame().await? else {
            return Ok(None);
        };

        let palm_roi = self.palm.process(rgb, NoContext)?;
        let landmarks = self.landmarks.process(
            MediaPipeHandLandmarkInput {
                frame: rgb,
                roi: palm_roi,
            },
            NoContext,
        )?;
        let landmark_roi = self.landmark_roi.process(
            LandmarkRoiInput {
                landmarks: landmarks.as_ref(),
                frame_size: rgb.meta.size,
            },
            NoContext,
        )?;
        let rgb_roi = landmark_roi.or(palm_roi);

        Ok(Some(ControllerFrame {
            rgb,
            palm_roi,
            landmarks,
            rgb_roi,
        }))
    }
}

impl<S> Tick for Pipeline<S>
where
    S: FrameSource + Send,
{
    fn tick(&mut self) -> Result<Option<ControllerFrame<'_>>> {
        pollster::block_on(self.tick_async())
    }
}
