use anyhow::{Context, Result};
use rustix::{fd, fs, ioctl};

#[derive(Debug)]
pub struct UvcStepper {
    device: UvcDevice,
    path: String,
    unit: u8,
    selector: u8,
    values: Vec<Vec<u8>>,
    next_index: usize,
}

impl UvcStepper {
    pub fn new(path: String, unit: u8, selector: u8) -> Result<Self> {
        let device = UvcDevice::open(&path)?;
        let len = device
            .len(unit, selector)
            .with_context(|| format!("read UVC length for unit={unit} selector={selector}"))?;
        let cur = device.get(unit, selector, len, UvcQuery::GetCur)?;
        eprintln!("uvc-step: {path} unit={unit} selector={selector} cur={cur:?}");
        let values = values_for(unit, selector, len)?;
        Ok(Self {
            device,
            path,
            unit,
            selector,
            values,
            next_index: 0,
        })
    }

    pub fn step(&mut self) -> Result<()> {
        let mut value = self.values[self.next_index].clone();
        self.next_index = (self.next_index + 1) % self.values.len();
        self.device.set(self.unit, self.selector, &mut value)?;
        let cur = self.device.get(
            self.unit,
            self.selector,
            value.len() as u16,
            UvcQuery::GetCur,
        )?;
        eprintln!(
            "uvc-step: {} unit={} selector={} set={value:?} cur={cur:?}",
            self.path, self.unit, self.selector
        );
        Ok(())
    }
}

fn values_for(unit: u8, selector: u8, len: u16) -> Result<Vec<Vec<u8>>> {
    match (unit, selector, len) {
        (4, 6, 9) | (2, 6, 9) => Ok(vec![
            vec![1, 3, 1, 0, 0, 0, 0, 0, 0],
            vec![1, 3, 2, 0, 0, 0, 0, 0, 0],
            vec![1, 3, 3, 0, 0, 0, 0, 0, 0],
        ]),
        (4, 9, 4) => Ok(vec![vec![0, 0, 0, 0], vec![1, 0, 0, 0]]),
        _ => anyhow::bail!("no built-in step values for unit={unit} selector={selector} len={len}"),
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum UvcQuery {
    SetCur = 0x01,
    GetCur = 0x81,
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

#[derive(Debug)]
struct UvcDevice {
    fd: fd::OwnedFd,
}

impl UvcDevice {
    fn open(path: &str) -> Result<Self> {
        Ok(Self {
            fd: fs::open(path, fs::OFlags::RDWR, fs::Mode::empty())
                .with_context(|| format!("open {path}"))?,
        })
    }

    fn len(&self, unit: u8, selector: u8) -> Result<u16> {
        let mut data = vec![0_u8; 2];
        self.query(unit, selector, UvcQuery::GetLen, &mut data)?;
        Ok(u16::from_le_bytes([data[0], data[1]]))
    }

    fn get(&self, unit: u8, selector: u8, len: u16, query: UvcQuery) -> Result<Vec<u8>> {
        let mut data = vec![0_u8; len as usize];
        self.query(unit, selector, query, &mut data)?;
        Ok(data)
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
