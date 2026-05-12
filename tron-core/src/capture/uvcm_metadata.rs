use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex, TryLockError};

use anyhow::{Context, Result};
use tokio::runtime::Handle;
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};
use v4l::buffer::{Metadata, Type};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::{v4l_sys, v4l2};

const DEFAULT_BUFFERS: u32 = 4;
const DEFAULT_SAMPLE_CAPACITY: usize = 256;
const DEFAULT_LOOKUP_TIMEOUT: Duration = Duration::from_millis(120);
const UVCM_METADATA_ID_FRAME_ILLUMINATION: u32 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UvcmFrameIllumination {
    pub sequence: Option<u64>,
    pub illumination_on: bool,
}

pub struct V4lUvcmMetadataSource {
    shared: Arc<Mutex<SharedMetadata>>,
    reader: JoinHandle<()>,
}

struct SharedMetadata {
    samples: VecDeque<UvcmFrameIllumination>,
    capacity: usize,
    error: Option<anyhow::Error>,
}

impl V4lUvcmMetadataSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_capacity(path, DEFAULT_SAMPLE_CAPACITY)
    }

    pub fn open_with_capacity(path: impl AsRef<Path>, capacity: usize) -> Result<Self> {
        let path = path.as_ref();
        let dev = Device::with_path(path).with_context(|| format!("open {}", path.display()))?;
        set_uvcm_meta_format(&dev)
            .with_context(|| format!("set UVCM metadata format on {}", path.display()))?;
        let stream = MmapStream::with_buffers(&dev, Type::MetaCapture, DEFAULT_BUFFERS)
            .with_context(|| format!("create V4L metadata mmap stream on {}", path.display()))?;
        let shared = Arc::new(Mutex::new(SharedMetadata {
            samples: VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            error: None,
        }));
        let reader_shared = shared.clone();
        let reader = Handle::try_current()
            .context("open V4L UVCM metadata source inside a Tokio runtime")?
            .spawn_blocking(move || read_metadata(stream, reader_shared));
        Ok(Self { shared, reader })
    }

    pub async fn illumination_for_sequence(&self, sequence: u64) -> Result<Option<bool>> {
        let deadline = Instant::now() + DEFAULT_LOOKUP_TIMEOUT;
        loop {
            match self.try_illumination_for_sequence(sequence)? {
                LookupState::Found(illumination_on) => return Ok(Some(illumination_on)),
                LookupState::Missed => return Ok(None),
                LookupState::Pending if Instant::now() >= deadline => return Ok(None),
                LookupState::Pending => tokio::task::yield_now().await,
            }
        }
    }

    fn try_illumination_for_sequence(&self, sequence: u64) -> Result<LookupState> {
        let mut shared = match self.shared.try_lock() {
            Ok(shared) => shared,
            Err(TryLockError::WouldBlock) => return Ok(LookupState::Pending),
            Err(TryLockError::Poisoned(_)) => anyhow::bail!("V4L UVCM metadata lock poisoned"),
        };
        if let Some(err) = shared.error.take() {
            return Err(err);
        }
        let Some(index) = shared
            .samples
            .iter()
            .position(|sample| sample.sequence == Some(sequence))
        else {
            if shared
                .samples
                .back()
                .and_then(|sample| sample.sequence)
                .is_some_and(|latest_sequence| latest_sequence > sequence)
            {
                while shared
                    .samples
                    .front()
                    .and_then(|sample| sample.sequence)
                    .is_some_and(|sample_sequence| sample_sequence < sequence)
                {
                    shared.samples.pop_front();
                }
                return Ok(LookupState::Missed);
            }
            return Ok(LookupState::Pending);
        };
        let illumination_on = shared.samples[index].illumination_on;
        shared.samples.drain(..=index);
        Ok(LookupState::Found(illumination_on))
    }
}

enum LookupState {
    Found(bool),
    Missed,
    Pending,
}

impl Drop for V4lUvcmMetadataSource {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

fn read_metadata(mut stream: MmapStream<'static>, shared: Arc<Mutex<SharedMetadata>>) {
    loop {
        match next_illuminations(&mut stream) {
            Ok(samples) => {
                if let Ok(mut shared) = shared.lock() {
                    for sample in samples {
                        if shared.samples.len() >= shared.capacity {
                            shared.samples.pop_front();
                        }
                        shared.samples.push_back(sample);
                    }
                }
            }
            Err(err) => {
                if let Ok(mut shared) = shared.lock() {
                    shared.error = Some(err);
                }
                return;
            }
        }
    }
}

fn next_illuminations(stream: &mut MmapStream<'static>) -> Result<Vec<UvcmFrameIllumination>> {
    let (buf, meta) = stream.next().context("dequeue V4L UVCM metadata")?;
    let used_len = (meta.bytesused as usize).min(buf.len());
    parse_uvcm_frame_illumination(&buf[..used_len], meta)
}

fn set_uvcm_meta_format(dev: &Device) -> Result<()> {
    let mut format = v4l_sys::v4l2_format {
        type_: Type::MetaCapture as u32,
        fmt: v4l_sys::v4l2_format__bindgen_ty_1 {
            meta: v4l_sys::v4l2_meta_format {
                dataformat: fourcc_bytes(*b"UVCM"),
                buffersize: 0,
                width: 0,
                height: 0,
                bytesperline: 0,
            },
        },
    };
    unsafe {
        v4l2::ioctl(
            dev.handle().fd(),
            v4l2::vidioc::VIDIOC_S_FMT,
            &mut format as *mut _ as *mut std::os::raw::c_void,
        )
        .context("VIDIOC_S_FMT UVCM metadata")?;
    }
    Ok(())
}

fn parse_uvcm_frame_illumination(
    data: &[u8],
    meta: &Metadata,
) -> Result<Vec<UvcmFrameIllumination>> {
    let mut output = Vec::new();
    parse_raw_uvcm_blocks(data, meta.sequence as u64, &mut output);
    if !output.is_empty() {
        return Ok(output);
    }

    if let Some(illumination_on) = find_direct_frame_illumination_item(data) {
        output.push(UvcmFrameIllumination {
            sequence: Some(meta.sequence as u64),
            illumination_on,
        });
    }
    Ok(output)
}

fn parse_raw_uvcm_blocks(data: &[u8], sequence: u64, output: &mut Vec<UvcmFrameIllumination>) {
    let mut offset = 0;
    while offset + 12 <= data.len() {
        let length = data[offset + 10] as usize;
        if length < 2 {
            offset += 1;
            continue;
        }
        let Some(block_end) = offset
            .checked_add(10)
            .and_then(|start| start.checked_add(length))
        else {
            break;
        };
        if block_end > data.len() {
            offset += 1;
            continue;
        }

        if length > 12 {
            let extra_start = offset + 22;
            if extra_start <= block_end {
                parse_extra_metadata_items(&data[extra_start..block_end], sequence, output);
            }
        }
        offset = block_end;
    }
}

fn parse_extra_metadata_items(data: &[u8], sequence: u64, output: &mut Vec<UvcmFrameIllumination>) {
    let mut offset = 0;
    while offset + 8 <= data.len() {
        let id = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .expect("metadata id slice length"),
        );
        let size = u32::from_le_bytes(
            data[offset + 4..offset + 8]
                .try_into()
                .expect("metadata size slice length"),
        );
        let size = size as usize;
        if size < 8 || offset + size > data.len() {
            break;
        }

        let payload = &data[offset + 8..offset + size];
        if id == UVCM_METADATA_ID_FRAME_ILLUMINATION && payload.len() >= 8 {
            let frame_flags = u32::from_le_bytes(payload[0..4].try_into().expect("payload len"));
            output.push(UvcmFrameIllumination {
                sequence: Some(sequence),
                illumination_on: (frame_flags & 0x01) != 0,
            });
        }
        offset += size;
    }
}

fn find_direct_frame_illumination_item(data: &[u8]) -> Option<bool> {
    let mut output = Vec::new();
    parse_extra_metadata_items(data, 0, &mut output);
    output
        .first()
        .map(|illumination| illumination.illumination_on)
}

fn fourcc_bytes(bytes: [u8; 4]) -> u32 {
    u32::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_uvcm_frame_illumination_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(&UVCM_METADATA_ID_FRAME_ILLUMINATION.to_le_bytes());
        data.extend_from_slice(&16_u32.to_le_bytes());
        data.extend_from_slice(&1_u32.to_le_bytes());
        data.extend_from_slice(&0_u32.to_le_bytes());

        assert_eq!(find_direct_frame_illumination_item(&data), Some(true));
        data[8] = 0;
        assert_eq!(find_direct_frame_illumination_item(&data), Some(false));
    }

    #[test]
    fn parses_raw_uvcm_blocks() {
        let mut data = Vec::new();
        data.extend_from_slice(&123_u64.to_le_bytes());
        data.extend_from_slice(&45_u16.to_le_bytes());
        data.push(28);
        data.push(0);
        data.extend_from_slice(&[0; 10]);
        data.extend_from_slice(&UVCM_METADATA_ID_FRAME_ILLUMINATION.to_le_bytes());
        data.extend_from_slice(&16_u32.to_le_bytes());
        data.extend_from_slice(&1_u32.to_le_bytes());
        data.extend_from_slice(&0_u32.to_le_bytes());

        let mut output = Vec::new();
        parse_raw_uvcm_blocks(&data, 7, &mut output);

        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0],
            UvcmFrameIllumination {
                sequence: Some(7),
                illumination_on: true,
            }
        );
    }
}
