//! IR emitter control for the UVC camera at `/dev/video51`.
//!
//! The emitter is exposed as a UVC Extension Unit control at unit=4,
//! selector=6, with a 9-byte payload. Encoding was reverse-engineered by
//! probing — see `strobetest.sh` in the linux-enable-ir-emitter scratch dir.
//! Byte 0 selects the command set, byte 1 the mode, byte 2 a sub-mode.

use anyhow::{Context, Result};
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU8, Ordering};

const UNIT: u8 = 4;
const SELECTOR: u8 = 6;
const PAYLOAD_LEN: usize = 9;

const UVC_SET_CUR: u8 = 0x01;
const UVCIOC_CTRL_QUERY: libc::c_ulong = 0xC010_5521;

#[repr(C)]
struct UvcXuControlQuery {
    unit: u8,
    selector: u8,
    query: u8,
    size: u16,
    data: *mut u8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IrMode {
    Off,
    Strobe,
    Steady,
}

impl IrMode {
    fn payload(self) -> [u8; PAYLOAD_LEN] {
        match self {
            IrMode::Off    => [1, 1, 0, 0, 0, 0, 0, 0, 0],
            IrMode::Strobe => [1, 1, 1, 0, 0, 0, 0, 0, 0],
            IrMode::Steady => [1, 3, 1, 0, 0, 0, 0, 0, 0],
        }
    }

    fn next(self) -> Self {
        match self {
            IrMode::Off => IrMode::Strobe,
            IrMode::Strobe => IrMode::Steady,
            IrMode::Steady => IrMode::Off,
        }
    }
}

static CURRENT: AtomicU8 = AtomicU8::new(IrMode::Steady as u8);

fn from_u8(v: u8) -> IrMode {
    match v {
        0 => IrMode::Off,
        1 => IrMode::Strobe,
        _ => IrMode::Steady,
    }
}

pub fn current() -> IrMode {
    from_u8(CURRENT.load(Ordering::Relaxed))
}

pub fn set(device: &str, mode: IrMode) -> Result<()> {
    let mut payload = mode.payload();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(device)
        .with_context(|| format!("open {device}"))?;
    let mut q = UvcXuControlQuery {
        unit: UNIT,
        selector: SELECTOR,
        query: UVC_SET_CUR,
        size: PAYLOAD_LEN as u16,
        data: payload.as_mut_ptr(),
    };
    let rc = unsafe { libc::ioctl(file.as_raw_fd(), UVCIOC_CTRL_QUERY, &mut q) };
    if rc < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("UVCIOC_CTRL_QUERY on {device} for mode {mode:?}"));
    }
    CURRENT.store(mode as u8, Ordering::Relaxed);
    eprintln!("ir_emitter: {mode:?}");
    Ok(())
}

pub fn cycle(device: &str) -> Result<()> {
    set(device, current().next())
}
