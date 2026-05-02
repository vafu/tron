use anyhow::{Context, Result, anyhow};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct OrtConfig {
    pub optimization: GraphOptimizationLevel,
    pub intra_threads: usize,
}

impl OrtConfig {
    pub fn cpu(intra_threads: usize) -> Self {
        Self {
            optimization: GraphOptimizationLevel::Level3,
            intra_threads,
        }
    }
}

/// ONNX Runtime construction boundary.
///
/// Execution provider selection, GPU/NPU enablement, and shared runtime knobs
/// should be added here so model-specific code stays focused on tensor
/// preprocessing and output decoding.
pub fn load_ort_session(path: impl AsRef<Path>, config: OrtConfig) -> Result<Session> {
    let path = path.as_ref();
    Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(config.optimization)
        .map_err(ort_err)?
        .with_intra_threads(config.intra_threads)
        .map_err(ort_err)?
        .commit_from_file(path)
        .map_err(ort_err)
        .with_context(|| format!("load ONNX model {}", path.display()))
}

fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("ort: {e}")
}
