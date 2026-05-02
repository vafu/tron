use crate::types::Image;
use anyhow::Result;
use std::sync::{Arc, Mutex};

pub mod v4l;

pub type SharedImage = Arc<Mutex<Option<Image>>>;

#[derive(Clone, Copy, Debug)]
pub enum StreamFormat {
    Rgb,
    Ir,
}

#[derive(Clone, Debug)]
pub struct StreamConfig {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub format: StreamFormat,
}

impl StreamConfig {
    pub fn rgb(path: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            path: path.into(),
            width,
            height,
            format: StreamFormat::Rgb,
        }
    }

    pub fn ir(path: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            path: path.into(),
            width,
            height,
            format: StreamFormat::Ir,
        }
    }
}

/// Camera backend boundary.
///
/// The rest of the app consumes `SharedImage` and `StreamConfig`; backend
/// details such as V4L, libcamera, or IPU6-specific setup should live behind
/// this trait.
pub trait CameraBackend {
    fn spawn_stream(&self, config: StreamConfig) -> Result<SharedImage>;
}

pub fn spawn_stream<B: CameraBackend>(backend: &B, config: StreamConfig) -> Result<SharedImage> {
    backend.spawn_stream(config)
}

pub fn spawn_rgb(path: &str, width: u32, height: u32) -> Result<SharedImage> {
    spawn_stream(&v4l::Backend, StreamConfig::rgb(path, width, height))
}

pub fn spawn_ir(path: &str, width: u32, height: u32) -> Result<SharedImage> {
    spawn_stream(&v4l::Backend, StreamConfig::ir(path, width, height))
}
