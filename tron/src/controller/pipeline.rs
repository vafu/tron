use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;
use tron_api::{
    Frame, FrameSource, GestureFrame, HandGesture, IterableFrameSource, NoContext, Processor,
    RoiResult,
};
use tron_core::gesture::{GesturePreprocessor, GesturePreprocessorInput};
use tron_core::process::landmark_velocity::{HandLandmarkMotion, LandmarkVelocityProcessor};
use tron_core::process::one_euro_landmarks::OneEuroLandmarkFilter;
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

#[derive(Serialize)]
pub struct ControllerFrame<'a> {
    pub rgb: Frame<'a>,
    pub palm_roi: Option<RoiResult>,
    pub landmark_input_roi: Option<RoiResult>,
    pub landmarks: Option<HandLandmarks>,
    pub landmark_motion: Option<HandLandmarkMotion>,
    pub output_roi: Option<RoiResult>,
    pub gesture: GestureFrame,
}

impl ControllerFrame<'_> {
    pub fn frame_id(&self) -> u64 {
        self.rgb.meta.id
    }
}

// TODO move to API, make generic, name "Pipeline"?
pub trait Tick {
    fn tick(&mut self) -> Result<Option<ControllerFrame<'_>>>;

    fn next_frame(&mut self) -> Result<bool> {
        Ok(false)
    }

    fn prev_frame(&mut self) -> Result<bool> {
        Ok(false)
    }
}

pub enum ControllerTicker<L, R> {
    Live(Pipeline<L>),
    Replay(ReplayPipeline<R>),
}

pub struct ReplayPipeline<S> {
    pipeline: Pipeline<S>,
    pending_frame: bool,
}

impl<S> ReplayPipeline<S> {
    pub fn new(pipeline: Pipeline<S>) -> Self {
        Self {
            pipeline,
            pending_frame: true,
        }
    }
}

pub struct Pipeline<S> {
    source: S,
    palm: MediaPipeRoiProcessor,
    landmarks: MediaPipeHandLandmarkProcessor,
    landmark_filter: OneEuroLandmarkFilter,
    landmark_velocity: LandmarkVelocityProcessor,
    landmark_roi: LandmarkRoiProcessor,
    prev_roi: Option<RoiResult>,
    last_pinch_state: Option<bool>,
    gesture: GesturePreprocessor,
}

impl<S> Pipeline<S>
where
    S: FrameSource + Send,
{
    pub fn new(source: S, config: PipelineConfig) -> Result<Self> {
        let landmark_roi = LandmarkRoiProcessor::new();
        Ok(Self {
            source,
            palm: MediaPipeRoiProcessor::new(config.palm_model, config.palm)?,
            landmarks: MediaPipeHandLandmarkProcessor::new(
                config.landmark_model,
                config.landmarks,
            )?,
            landmark_filter: OneEuroLandmarkFilter::default(),
            landmark_velocity: LandmarkVelocityProcessor::new(),
            landmark_roi,
            prev_roi: None,
            last_pinch_state: None,
            gesture: GesturePreprocessor,
        })
    }

    async fn tick_async(&mut self) -> Result<Option<ControllerFrame<'_>>> {
        let Some(rgb) = self.source.next_frame().await? else {
            return Ok(None);
        };

        let mut _palm_roi: Option<RoiResult> = None;
        let processing_roi = self.prev_roi.or_else(|| {
            _palm_roi = self.palm.process(rgb, NoContext).unwrap();
            _palm_roi
        });

        let landmarks = self.landmarks.process(
            MediaPipeHandLandmarkInput {
                frame: rgb,
                roi: processing_roi,
            },
            NoContext,
        )?;

        // let landmarks = self.landmark_filter.process(landmarks, NoContext)?;
        let landmark_motion = self
            .landmark_velocity
            .process(landmarks.clone(), NoContext)?;
        self.prev_roi = None;
        let output_roi = self.landmark_roi.process(
            LandmarkRoiInput {
                landmarks: landmarks.as_ref(),
                frame_size: rgb.meta.size,
            },
            NoContext,
        )?;

        let gesture = self.gesture.process(
            GesturePreprocessorInput {
                landmarks: landmarks.as_ref(),
                palm_roi: processing_roi,
                frame_size: rgb.meta.size,
                timestamp: rgb.meta.timestamp.received_at,
            },
            NoContext,
        )?;
        if landmarks.is_some() {
            let pinch = gesture.signal(HandGesture::Pinch).is_some();
            let should_dump = self
                .last_pinch_state
                .is_some_and(|previous| previous != pinch);
            self.last_pinch_state = Some(pinch);
            if should_dump {
                self.landmarks.dump_last_debug()?;
            }
        } else {
            self.last_pinch_state = None;
        }

        Ok(Some(ControllerFrame {
            rgb,
            palm_roi: _palm_roi,
            landmark_input_roi: processing_roi,
            output_roi,
            landmarks,
            landmark_motion,
            gesture,
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

impl<S> Tick for ReplayPipeline<S>
where
    S: IterableFrameSource + Send,
{
    fn tick(&mut self) -> Result<Option<ControllerFrame<'_>>> {
        if !self.pending_frame {
            return Ok(None);
        }
        self.pending_frame = false;
        self.pipeline.tick()
    }

    fn next_frame(&mut self) -> Result<bool> {
        self.pending_frame = true;
        Ok(true)
    }

    fn prev_frame(&mut self) -> Result<bool> {
        let moved = self.pipeline.source.prev_frame()?;
        if moved {
            self.pending_frame = true;
        }
        Ok(moved)
    }
}

impl<L, R> Tick for ControllerTicker<L, R>
where
    L: FrameSource + Send,
    R: IterableFrameSource + Send,
{
    fn tick(&mut self) -> Result<Option<ControllerFrame<'_>>> {
        match self {
            Self::Live(pipeline) => pipeline.tick(),
            Self::Replay(pipeline) => pipeline.tick(),
        }
    }

    fn next_frame(&mut self) -> Result<bool> {
        match self {
            Self::Live(_) => Ok(false),
            Self::Replay(pipeline) => pipeline.next_frame(),
        }
    }

    fn prev_frame(&mut self) -> Result<bool> {
        match self {
            Self::Live(_) => Ok(false),
            Self::Replay(pipeline) => pipeline.prev_frame(),
        }
    }
}
