use anyhow::Result;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tron_api::{CapturedFrame, Frame, FrameDecoder, FrameSource, OwnedFrame};

pub trait FrameStream {
    fn next_frame(&mut self) -> Result<Option<Frame<'_>>>;
}

pub trait FramePairStream {
    fn next_pair(&mut self) -> Result<Option<SyncedFramePair>>;
}

#[derive(Clone, Debug)]
pub struct SyncedFramePair {
    pub left: OwnedFrame,
    pub right: OwnedFrame,
    pub delta_us: i64,
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
    L: FrameStream,
    R: FrameStream,
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

impl<L, R> FramePairStream for FrameSynchronizer<L, R>
where
    L: FrameStream,
    R: FrameStream,
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

pub struct PassthroughStream<S> {
    source: S,
}

impl<S> PassthroughStream<S> {
    pub fn new(source: S) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

fn next_owned_frame(stream: &mut impl FrameStream) -> Result<Option<OwnedFrame>> {
    stream.next_frame().map(|frame| frame.map(own_frame))
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

impl<S> FrameStream for PassthroughStream<S>
where
    S: FrameSource,
{
    fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        match self.source.next_frame()? {
            Some(CapturedFrame::Frame(frame)) => Ok(Some(frame)),
            Some(CapturedFrame::Encoded(frame)) => anyhow::bail!(
                "passthrough stream received encoded frame {:?} from sensor {:?}",
                frame.format,
                frame.meta.sensor
            ),
            None => Ok(None),
        }
    }
}

pub struct DecodeStream<S, D> {
    source: S,
    decoder: D,
}

impl<S, D> DecodeStream<S, D> {
    pub fn new(source: S, decoder: D) -> Self {
        Self { source, decoder }
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    pub fn decoder(&self) -> &D {
        &self.decoder
    }

    pub fn decoder_mut(&mut self) -> &mut D {
        &mut self.decoder
    }
}

impl<S, D> FrameStream for DecodeStream<S, D>
where
    S: FrameSource,
    D: FrameDecoder,
{
    fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        match self.source.next_frame()? {
            Some(CapturedFrame::Encoded(frame)) => self.decoder.decode(frame).map(Some),
            Some(CapturedFrame::Frame(frame)) => anyhow::bail!(
                "decode stream received pixel frame {:?} from sensor {:?}",
                frame.format,
                frame.meta.sensor
            ),
            None => Ok(None),
        }
    }
}
