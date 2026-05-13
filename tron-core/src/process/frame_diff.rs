use anyhow::{Context, Result};
use tron_api::{Frame, FrameMeta, OwnedFrame, PixelFormat};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameDiffMode {
    BrighterOnly,
    Absolute,
}

impl Default for FrameDiffMode {
    fn default() -> Self {
        Self::Absolute
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameDiffReferencePolicy {
    PreviousFrame,
    AlternatingPair,
}

impl Default for FrameDiffReferencePolicy {
    fn default() -> Self {
        Self::PreviousFrame
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameDiffOutputPolicy {
    EveryFrame,
    MeaningfulOnly,
}

impl Default for FrameDiffOutputPolicy {
    fn default() -> Self {
        Self::EveryFrame
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FrameDiffConfig {
    pub mode: FrameDiffMode,
    pub reference_policy: FrameDiffReferencePolicy,
    pub output_policy: FrameDiffOutputPolicy,
    pub min_output_value: u8,
    pub min_output_pixels: usize,
}

impl Default for FrameDiffConfig {
    fn default() -> Self {
        Self {
            mode: FrameDiffMode::default(),
            reference_policy: FrameDiffReferencePolicy::default(),
            output_policy: FrameDiffOutputPolicy::default(),
            min_output_value: 1,
            min_output_pixels: 1,
        }
    }
}

pub struct FrameDiffProcessor {
    config: FrameDiffConfig,
    previous: Option<OwnedFrame>,
    output: Option<OwnedFrame>,
    scratch: Vec<u8>,
}

impl FrameDiffProcessor {
    pub fn new(config: FrameDiffConfig) -> Self {
        Self {
            config,
            previous: None,
            output: None,
            scratch: Vec::new(),
        }
    }

    pub fn process(&mut self, frame: Frame<'_>) -> Result<Frame<'_>> {
        anyhow::ensure!(
            frame.format == PixelFormat::Gray8,
            "frame diff requires Gray8 frames, got {:?}",
            frame.format
        );
        anyhow::ensure!(
            frame.buffer.stride == frame.meta.size.width as usize,
            "frame diff requires tightly packed Gray8 frames"
        );
        self.ensure_buffers(frame)?;

        let previous = self.previous.as_mut().expect("previous buffer initialized");
        let output = self.output.as_mut().expect("output buffer initialized");

        compute_diff(
            self.config.mode,
            self.config.reference_policy,
            frame.buffer.data,
            &previous.data,
            &mut self.scratch,
        )
        .context("compute frame diff")?;

        let should_update_output = self.config.output_policy == FrameDiffOutputPolicy::EveryFrame
            || is_meaningful_output(
                &self.scratch,
                self.config.min_output_value,
                self.config.min_output_pixels,
            );
        if should_update_output {
            output.data.copy_from_slice(&self.scratch);
            output.meta = FrameMeta {
                id: frame.meta.id,
                sensor: frame.meta.sensor,
                size: frame.meta.size,
                timestamp: frame.meta.timestamp,
                sequence: frame.meta.sequence,
            };
        }

        previous.meta = frame.meta;
        previous.data.copy_from_slice(frame.buffer.data);

        Ok(output.as_frame())
    }

    fn ensure_buffers(&mut self, frame: Frame<'_>) -> Result<()> {
        let len = frame.buffer.data.len();

        if let Some(previous) = &self.previous {
            anyhow::ensure!(
                previous.meta.size == frame.meta.size
                    && previous.format == frame.format
                    && previous.stride == frame.buffer.stride
                    && previous.data.len() == len,
                "frame diff shape changed from {:?}/stride {}/len {} to {:?}/stride {}/len {}",
                previous.meta.size,
                previous.stride,
                previous.data.len(),
                frame.meta.size,
                frame.buffer.stride,
                len
            );
        } else {
            self.previous = Some(OwnedFrame {
                meta: frame.meta,
                format: frame.format,
                stride: frame.buffer.stride,
                data: frame.buffer.data.to_vec(),
            });
        }

        if self.output.is_none() {
            self.output = Some(OwnedFrame {
                meta: frame.meta,
                format: PixelFormat::Gray8,
                stride: frame.buffer.stride,
                data: vec![0; len],
            });
        }
        if self.scratch.len() != len {
            self.scratch.resize(len, 0);
        }

        Ok(())
    }
}

fn compute_diff(
    mode: FrameDiffMode,
    reference_policy: FrameDiffReferencePolicy,
    current: &[u8],
    previous: &[u8],
    output: &mut [u8],
) -> Result<()> {
    anyhow::ensure!(
        current.len() == previous.len() && current.len() == output.len(),
        "frame diff buffer length mismatch: current={} previous={} output={}",
        current.len(),
        previous.len(),
        output.len()
    );

    match reference_policy {
        FrameDiffReferencePolicy::PreviousFrame => {
            compute_directional_diff(mode, current, previous, output)
        }
        FrameDiffReferencePolicy::AlternatingPair => {
            if frame_sum(current) >= frame_sum(previous) {
                compute_directional_diff(mode, current, previous, output);
            } else {
                compute_directional_diff(mode, previous, current, output);
            }
        }
    }
    Ok(())
}

fn compute_directional_diff(
    mode: FrameDiffMode,
    current: &[u8],
    reference: &[u8],
    output: &mut [u8],
) {
    match mode {
        FrameDiffMode::BrighterOnly => compute_brighter_diff(current, reference, output),
        FrameDiffMode::Absolute => compute_absolute_diff(current, reference, output),
    }
}

fn compute_brighter_diff(current: &[u8], reference: &[u8], output: &mut [u8]) {
    for ((out, current), reference) in output.iter_mut().zip(current).zip(reference) {
        *out = current.saturating_sub(*reference);
    }
}

fn compute_absolute_diff(current: &[u8], reference: &[u8], output: &mut [u8]) {
    for ((out, current), reference) in output.iter_mut().zip(current).zip(reference) {
        *out = current.abs_diff(*reference);
    }
}

fn frame_sum(data: &[u8]) -> u64 {
    data.iter().map(|value| *value as u64).sum()
}

fn is_meaningful_output(output: &[u8], min_value: u8, min_pixels: usize) -> bool {
    let min_pixels = min_pixels.max(1);
    output
        .iter()
        .filter(|value| **value >= min_value)
        .take(min_pixels)
        .count()
        == min_pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brighter_diff_saturates_negative_values() {
        let current = [10, 30, 200, 0];
        let previous = [20, 10, 200, 255];
        let mut output = [0; 4];

        compute_diff(
            FrameDiffMode::BrighterOnly,
            FrameDiffReferencePolicy::PreviousFrame,
            &current,
            &previous,
            &mut output,
        )
        .unwrap();

        assert_eq!(output, [0, 20, 0, 0]);
    }

    #[test]
    fn absolute_diff_keeps_both_directions() {
        let current = [10, 30, 200, 0];
        let previous = [20, 10, 200, 255];
        let mut output = [0; 4];

        compute_diff(
            FrameDiffMode::Absolute,
            FrameDiffReferencePolicy::PreviousFrame,
            &current,
            &previous,
            &mut output,
        )
        .unwrap();

        assert_eq!(output, [10, 20, 0, 255]);
    }

    #[test]
    fn alternating_pair_subtracts_dimmer_from_brighter() {
        let dim = [10, 20, 30, 40];
        let bright = [20, 30, 40, 50];
        let mut output = [0; 4];

        compute_diff(
            FrameDiffMode::BrighterOnly,
            FrameDiffReferencePolicy::AlternatingPair,
            &dim,
            &bright,
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [10, 10, 10, 10]);

        compute_diff(
            FrameDiffMode::BrighterOnly,
            FrameDiffReferencePolicy::AlternatingPair,
            &bright,
            &dim,
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [10, 10, 10, 10]);
    }

    #[test]
    fn meaningful_output_requires_threshold_and_count() {
        assert!(!is_meaningful_output(&[0, 1, 2, 3], 8, 1));
        assert!(!is_meaningful_output(&[8, 0, 1, 9], 8, 3));
        assert!(is_meaningful_output(&[8, 0, 1, 9], 8, 2));
    }
}
