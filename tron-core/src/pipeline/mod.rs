use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tron_api::{Frame, FrameSource, OpenedCameraInfo, OwnedFrame};

pub trait FramePairSource {
    fn next_pair(&mut self) -> Result<Option<SyncedFramePair>>;
}

#[derive(Clone, Debug)]
pub struct SyncedFramePair {
    pub left: OwnedFrame,
    pub right: OwnedFrame,
    pub delta_us: i64,
}

pub struct BufferedFrameSource {
    info: OpenedCameraInfo,
    shared: Arc<Mutex<BufferedState>>,
    current: Option<OwnedFrame>,
}

impl BufferedFrameSource {
    pub fn spawn<S>(name: &'static str, mut source: S, capacity: usize) -> Self
    where
        S: FrameSource + Send + 'static,
    {
        let info = source.info().clone();
        let shared = Arc::new(Mutex::new(BufferedState {
            frames: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            error: None,
        }));
        let thread_shared = shared.clone();

        thread::spawn(move || {
            loop {
                match source.next_frame() {
                    Ok(Some(frame)) => {
                        let frame = own_frame(frame);
                        if let Ok(mut state) = thread_shared.lock() {
                            if state.frames.len() >= state.capacity {
                                state.frames.pop_front();
                            }
                            state.frames.push_back(frame);
                        }
                    }
                    Ok(None) => thread::yield_now(),
                    Err(err) => {
                        if let Ok(mut state) = thread_shared.lock() {
                            state.error = Some(err);
                        }
                        eprintln!("buffered frame source {name}: stopped");
                        return;
                    }
                }
            }
        });

        Self {
            info,
            shared,
            current: None,
        }
    }
}

impl FrameSource for BufferedFrameSource {
    fn info(&self) -> &OpenedCameraInfo {
        &self.info
    }

    fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        let mut state = self
            .shared
            .lock()
            .map_err(|_| anyhow::anyhow!("buffered frame source lock poisoned"))?;
        if let Some(err) = state.error.take() {
            return Err(err);
        }
        self.current = state.frames.pop_front();
        Ok(self.current.as_ref().map(|frame| frame.as_frame()))
    }
}

struct BufferedState {
    frames: VecDeque<OwnedFrame>,
    capacity: usize,
    error: Option<anyhow::Error>,
}

pub struct FrameSynchronizer<L, R> {
    left: L,
    right: R,
    left_pending: VecDeque<OwnedFrame>,
    right_pending: VecDeque<OwnedFrame>,
    max_delta_us: i64,
    dropped_left: u64,
    dropped_right: u64,
}

impl<L, R> FrameSynchronizer<L, R>
where
    L: FrameSource,
    R: FrameSource,
{
    pub fn new(left: L, right: R, max_delta_us: i64) -> Self {
        Self {
            left,
            right,
            left_pending: VecDeque::new(),
            right_pending: VecDeque::new(),
            max_delta_us: max_delta_us.max(0),
            dropped_left: 0,
            dropped_right: 0,
        }
    }

    pub fn next_unsynchronized_pair(&mut self) -> Result<Option<SyncedFramePair>> {
        self.fill_empty_buffers()?;
        let Some(left) = self.left_pending.pop_front() else {
            return Ok(None);
        };
        let Some(right) = self.right_pending.pop_front() else {
            self.left_pending.push_front(left);
            return Ok(None);
        };
        let delta_us = frame_delta_us(&left, &right);
        Ok(Some(SyncedFramePair {
            left,
            right,
            delta_us,
        }))
    }

    pub fn dropped_counts(&self) -> (u64, u64) {
        (self.dropped_left, self.dropped_right)
    }

    fn fill_empty_buffers(&mut self) -> Result<()> {
        if self.left_pending.is_empty() {
            if let Some(frame) = next_owned_frame(&mut self.left)? {
                self.left_pending.push_back(frame);
            }
        }
        if self.right_pending.is_empty() {
            if let Some(frame) = next_owned_frame(&mut self.right)? {
                self.right_pending.push_back(frame);
            }
        }
        Ok(())
    }
}

impl<L, R> FramePairSource for FrameSynchronizer<L, R>
where
    L: FrameSource,
    R: FrameSource,
{
    fn next_pair(&mut self) -> Result<Option<SyncedFramePair>> {
        const MAX_DEQUEUE_STEPS: usize = 32;

        for _ in 0..MAX_DEQUEUE_STEPS {
            self.fill_empty_buffers()?;
            let Some(left) = self.left_pending.front() else {
                return Ok(None);
            };
            let Some(right) = self.right_pending.front() else {
                return Ok(None);
            };

            let delta_us = frame_delta_us(left, right);
            if delta_us.abs() <= self.max_delta_us {
                return Ok(Some(SyncedFramePair {
                    left: self
                        .left_pending
                        .pop_front()
                        .expect("left front disappeared"),
                    right: self
                        .right_pending
                        .pop_front()
                        .expect("right front disappeared"),
                    delta_us,
                }));
            }

            if delta_us < 0 {
                self.left_pending.pop_front();
                self.dropped_left += 1;
            } else {
                self.right_pending.pop_front();
                self.dropped_right += 1;
            }
        }

        Ok(None)
    }
}

fn next_owned_frame(source: &mut impl FrameSource) -> Result<Option<OwnedFrame>> {
    source.next_frame().map(|frame| frame.map(own_frame))
}

fn own_frame(frame: Frame<'_>) -> OwnedFrame {
    OwnedFrame {
        meta: frame.meta,
        format: frame.format,
        stride: frame.stride,
        data: frame.data.to_vec(),
    }
}

fn frame_delta_us(left: &OwnedFrame, right: &OwnedFrame) -> i64 {
    match (
        left.meta.timestamp.camera_monotonic_us,
        right.meta.timestamp.camera_monotonic_us,
    ) {
        (Some(left_us), Some(right_us)) => left_us.saturating_sub(right_us),
        _ => instant_delta_us(
            left.meta.timestamp.received_at,
            right.meta.timestamp.received_at,
        ),
    }
}

fn instant_delta_us(left: Instant, right: Instant) -> i64 {
    if left >= right {
        duration_to_i64_us(left.duration_since(right))
    } else {
        -duration_to_i64_us(right.duration_since(left))
    }
}

fn duration_to_i64_us(duration: Duration) -> i64 {
    duration.as_micros().min(i64::MAX as u128) as i64
}
