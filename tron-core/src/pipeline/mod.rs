use anyhow::Result;
use tron_api::{Frame, FrameSource};

#[derive(Clone, Copy, Debug)]
pub struct SyncedFramePair<'a> {
    pub left: Frame<'a>,
    pub right: Frame<'a>,
    pub delta_us: i64,
}

pub struct StereoFrameSource<L, R> {
    left: L,
    right: R,
    max_delta_us: i64,
}

impl<L, R> StereoFrameSource<L, R>
where
    L: FrameSource,
    R: FrameSource,
{
    pub fn new(left: L, right: R, max_delta_us: u64) -> Self {
        Self {
            left,
            right,
            max_delta_us: max_delta_us.min(i64::MAX as u64) as i64,
        }
    }

    /// Yield the next synchronized pair, dropping stale frames from the older
    /// stream until the timestamps are within the configured delta.
    pub async fn next_pair(&mut self) -> Result<Option<SyncedFramePair<'_>>> {
        let mut left = match next_timestamped_frame(&mut self.left).await? {
            Some(frame) => frame,
            None => return Ok(None),
        };
        let mut right = match next_timestamped_frame(&mut self.right).await? {
            Some(frame) => frame,
            None => return Ok(None),
        };

        loop {
            let delta_us = frame_delta_us(&left, &right);
            if delta_us.abs() <= self.max_delta_us {
                return Ok(Some(SyncedFramePair {
                    left,
                    right,
                    delta_us,
                }));
            }

            if delta_us > 0 {
                let _ = right;
                right = match next_timestamped_frame(&mut self.right).await? {
                    Some(frame) => frame,
                    None => return Ok(None),
                };
            } else {
                let _ = left;
                left = match next_timestamped_frame(&mut self.left).await? {
                    Some(frame) => frame,
                    None => return Ok(None),
                };
            }
        }
    }
}

async fn next_timestamped_frame<'a, S>(source: &mut S) -> Result<Option<Frame<'a>>>
where
    S: FrameSource,
{
    loop {
        let Some(frame) = source.next_frame().await? else {
            return Ok(None);
        };
        // SAFETY: the returned frame is either returned to the caller, or dropped
        // before this source is polled again. Callers must preserve that shape:
        // when a frame is stale, stop using it before asking the same source for
        // another frame.
        let frame = unsafe { detach_frame_lifetime(frame) };
        if frame.meta.timestamp.camera_monotonic_us.is_some() {
            return Ok(Some(frame));
        }
    }
}

fn frame_delta_us(left: &Frame<'_>, right: &Frame<'_>) -> i64 {
    let left_us = left
        .meta
        .timestamp
        .camera_monotonic_us
        .expect("left frame is timestamped");
    let right_us = right
        .meta
        .timestamp
        .camera_monotonic_us
        .expect("right frame is timestamped");
    left_us.saturating_sub(right_us)
}

/// Reborrow a `Frame` under a caller-chosen lifetime.
///
/// SAFETY: caller must ensure the underlying buffer in the originating
/// `FrameSource` is not invalidated for the duration of `'a`. In this module,
/// that means a stale frame must be dropped before polling the same source for
/// another frame, and a matched frame is returned immediately to the caller.
unsafe fn detach_frame_lifetime<'a>(frame: Frame<'_>) -> Frame<'a> {
    Frame {
        meta: frame.meta,
        format: frame.format,
        stride: frame.stride,
        data: unsafe { std::slice::from_raw_parts(frame.data.as_ptr(), frame.data.len()) },
    }
}

