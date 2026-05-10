use std::time::Instant;

use serde::Serialize;
use tron_api::{FrameId, OwnedFrame, SensorKind};

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct PlaygroundMetadata {
    pub rgb: CameraMetadata,
    pub ir: CameraMetadata,
    pub rgb_ir_delta_us: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CameraMetadata {
    pub sensor: Option<SensorKind>,
    pub frame_id: Option<FrameId>,
    pub sequence: Option<u64>,
    pub fps: Option<f32>,
    pub frame_delta_us: Option<i64>,
    pub age_us: Option<u128>,
    pub camera_monotonic_us: Option<i64>,
}

pub struct CameraStatsProcessor {
    last_frame_id: Option<FrameId>,
    last_camera_monotonic_us: Option<i64>,
    last_received_at: Option<Instant>,
    frame_delta_us: Option<i64>,
    fps: Option<f32>,
    fps_window_started_at: Instant,
    fps_window_frames: u32,
}

impl CameraStatsProcessor {
    pub fn new() -> Self {
        Self {
            last_frame_id: None,
            last_camera_monotonic_us: None,
            last_received_at: None,
            frame_delta_us: None,
            fps: None,
            fps_window_started_at: Instant::now(),
            fps_window_frames: 0,
        }
    }

    pub fn process(&mut self, frame: Option<&OwnedFrame>, now: Instant) -> CameraMetadata {
        let Some(frame) = frame else {
            return CameraMetadata {
                fps: self.fps,
                frame_delta_us: self.frame_delta_us,
                ..CameraMetadata::default()
            };
        };

        if self.last_frame_id != Some(frame.meta.id) {
            self.frame_delta_us = match (
                self.last_camera_monotonic_us,
                frame.meta.timestamp.camera_monotonic_us,
            ) {
                (Some(prev), Some(current)) => Some(current - prev),
                _ => self
                    .last_received_at
                    .and_then(|prev| {
                        frame
                            .meta
                            .timestamp
                            .received_at
                            .checked_duration_since(prev)
                    })
                    .map(|duration| duration.as_micros() as i64),
            };
            self.last_frame_id = Some(frame.meta.id);
            self.last_camera_monotonic_us = frame.meta.timestamp.camera_monotonic_us;
            self.last_received_at = Some(frame.meta.timestamp.received_at);
            self.fps_window_frames += 1;
        }

        let fps_window_elapsed = now.duration_since(self.fps_window_started_at);
        if fps_window_elapsed.as_secs_f32() >= 1.0 {
            self.fps = Some(self.fps_window_frames as f32 / fps_window_elapsed.as_secs_f32());
            self.fps_window_started_at = now;
            self.fps_window_frames = 0;
        }

        CameraMetadata {
            sensor: Some(frame.meta.sensor),
            frame_id: Some(frame.meta.id),
            sequence: frame.meta.sequence,
            fps: self.fps,
            frame_delta_us: self.frame_delta_us,
            age_us: Some(
                now.duration_since(frame.meta.timestamp.received_at)
                    .as_micros(),
            ),
            camera_monotonic_us: frame.meta.timestamp.camera_monotonic_us,
        }
    }
}
