use anyhow::{Context, Result};
use rustix::{fd, fs, ioctl};
use std::path::Path;

const FACE_AUTH_UNIT: u8 = 4;
const FACE_AUTH_SELECTOR: u8 = 6;

#[derive(Clone, Copy, Debug)]
pub enum FaceAuthMode {
    Default,
    Mode2,
}

impl FaceAuthMode {
    fn bytes(self) -> [u8; 9] {
        match self {
            Self::Default => [1, 3, 1, 0, 0, 0, 0, 0, 0],
            Self::Mode2 => [1, 3, 2, 0, 0, 0, 0, 0, 0],
        }
    }
}

pub fn set_face_auth_mode(path: impl AsRef<Path>, mode: FaceAuthMode) -> Result<()> {
    let path = path.as_ref();
    let device = UvcDevice::open(path)?;
    let len = device.len(FACE_AUTH_UNIT, FACE_AUTH_SELECTOR)?;
    let mut value = mode.bytes().to_vec();
    anyhow::ensure!(
        value.len() == len as usize,
        "face-auth UVC control length {} does not match expected {} on {}",
        len,
        value.len(),
        path.display()
    );
    device
        .set(FACE_AUTH_UNIT, FACE_AUTH_SELECTOR, &mut value)
        .with_context(|| format!("set face-auth {mode:?} on {}", path.display()))
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum UvcQuery {
    SetCur = 0x01,
    GetLen = 0x85,
}

#[repr(C)]
struct UvcXuControlQuery {
    unit: u8,
    selector: u8,
    query: u8,
    size: u16,
    data: *mut u8,
}

impl UvcXuControlQuery {
    fn new(unit: u8, selector: u8, query: UvcQuery, data: &mut [u8]) -> Self {
        Self {
            unit,
            selector,
            query: query as u8,
            size: data.len() as u16,
            data: data.as_mut_ptr(),
        }
    }
}

const UVCIOC_CTRL_QUERY: ioctl::Opcode = ioctl::opcode::read_write::<UvcXuControlQuery>(b'u', 0x21);

struct UvcDevice {
    fd: fd::OwnedFd,
}

impl UvcDevice {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            fd: fs::open(path, fs::OFlags::RDWR, fs::Mode::empty())
                .with_context(|| format!("open {}", path.display()))?,
        })
    }

    fn len(&self, unit: u8, selector: u8) -> Result<u16> {
        let mut data = vec![0_u8; 2];
        self.query(unit, selector, UvcQuery::GetLen, &mut data)?;
        Ok(u16::from_le_bytes([data[0], data[1]]))
    }

    fn set(&self, unit: u8, selector: u8, data: &mut [u8]) -> Result<()> {
        self.query(unit, selector, UvcQuery::SetCur, data)
    }

    fn query(&self, unit: u8, selector: u8, query: UvcQuery, data: &mut [u8]) -> Result<()> {
        let mut query = UvcXuControlQuery::new(unit, selector, query, data);
        let updater =
            unsafe { ioctl::Updater::<UVCIOC_CTRL_QUERY, UvcXuControlQuery>::new(&mut query) };
        unsafe { ioctl::ioctl(&self.fd, updater) }.map_err(std::io::Error::from)?;
        Ok(())
    }
}
