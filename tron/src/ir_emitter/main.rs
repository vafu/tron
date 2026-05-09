use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use rustix::{fd, fs, ioctl};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "ir-emitter-control")]
#[command(about = "Controlled UVC extension control writes for IR emitter research")]
struct Cli {
    /// Video device path.
    #[arg(short, long)]
    device: PathBuf,

    /// UVC extension unit.
    #[arg(long, default_value_t = 4)]
    unit: u8,

    /// UVC extension selector.
    #[arg(long, default_value_t = 6)]
    selector: u8,

    /// Read current/min/max/default values before setting.
    #[arg(long)]
    read: bool,

    /// Read-only scan of unit/selector controls on this device.
    #[arg(long)]
    scan: bool,

    /// Set a known NexiGo unit=4 selector=6 mode.
    #[arg(long, value_enum)]
    mode: Option<EmitterMode>,

    /// Set an explicit comma-separated byte vector, e.g. 1,3,2,0,0,0,0,0,0.
    #[arg(long, value_parser = parse_bytes)]
    value: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EmitterMode {
    /// Observed default value: [1,3,1,0,0,0,0,0,0].
    Default,
    /// Observed emitter-enabling value: [1,3,2,0,0,0,0,0,0].
    Mode2,
    /// Observed alternate emitter-enabling value: [1,3,3,0,0,0,0,0,0].
    Mode3,
}

impl EmitterMode {
    fn bytes(self) -> Vec<u8> {
        match self {
            Self::Default => vec![1, 3, 1, 0, 0, 0, 0, 0, 0],
            Self::Mode2 => vec![1, 3, 2, 0, 0, 0, 0, 0, 0],
            Self::Mode3 => vec![1, 3, 3, 0, 0, 0, 0, 0, 0],
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = UvcDevice::open(&cli.device)?;

    if cli.scan {
        scan_controls(&device);
        return Ok(());
    }

    let len = device
        .len(cli.unit, cli.selector)
        .with_context(|| format!("get length for unit={} selector={}", cli.unit, cli.selector))?;

    if cli.read {
        print_query(
            "cur",
            device.get(cli.unit, cli.selector, len, UvcQuery::GetCur)?,
        );
        print_query(
            "min",
            device.get(cli.unit, cli.selector, len, UvcQuery::GetMin)?,
        );
        print_query(
            "max",
            device.get(cli.unit, cli.selector, len, UvcQuery::GetMax)?,
        );
        print_query(
            "res",
            device.get(cli.unit, cli.selector, len, UvcQuery::GetRes)?,
        );
        print_query(
            "def",
            device.get(cli.unit, cli.selector, len, UvcQuery::GetDef)?,
        );
    }

    let value = match (cli.mode, cli.value) {
        (Some(mode), None) => Some(mode.bytes()),
        (None, Some(value)) => Some(value),
        (None, None) => None,
        (Some(_), Some(_)) => anyhow::bail!("use either --mode or --value, not both"),
    };

    if let Some(mut value) = value {
        anyhow::ensure!(
            value.len() == len as usize,
            "value length {} does not match UVC control length {}",
            value.len(),
            len
        );
        eprintln!(
            "setting {} unit={} selector={} value={:?}",
            cli.device.display(),
            cli.unit,
            cli.selector,
            value
        );
        device.set(cli.unit, cli.selector, &mut value)?;
        print_query(
            "cur",
            device
                .get(cli.unit, cli.selector, len, UvcQuery::GetCur)
                .context("read back current value after set")?,
        );
    }

    Ok(())
}

fn parse_bytes(value: &str) -> std::result::Result<Vec<u8>, String> {
    let bytes = value
        .split(',')
        .map(str::trim)
        .map(|part| {
            part.parse::<u8>()
                .map_err(|err| format!("invalid byte {part:?}: {err}"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if bytes.is_empty() {
        Err("value must contain at least one byte".to_string())
    } else {
        Ok(bytes)
    }
}

fn print_query(label: &str, value: Vec<u8>) {
    println!("{label}={value:?}");
}

fn scan_controls(device: &UvcDevice) {
    for unit in 0..=u8::MAX {
        for selector in 0..=u8::MAX {
            let Ok(len) = device.len(unit, selector) else {
                continue;
            };
            let Ok(cur) = device.get(unit, selector, len, UvcQuery::GetCur) else {
                continue;
            };
            let min = device.get(unit, selector, len, UvcQuery::GetMin).ok();
            let max = device.get(unit, selector, len, UvcQuery::GetMax).ok();
            let res = device.get(unit, selector, len, UvcQuery::GetRes).ok();
            let def = device.get(unit, selector, len, UvcQuery::GetDef).ok();
            println!(
                "unit={unit} selector={selector} len={len} cur={cur:?} min={min:?} max={max:?} res={res:?} def={def:?}"
            );
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum UvcQuery {
    SetCur = 0x01,
    GetCur = 0x81,
    GetMin = 0x82,
    GetMax = 0x83,
    GetRes = 0x84,
    GetLen = 0x85,
    GetDef = 0x87,
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
    fn open(path: &PathBuf) -> Result<Self> {
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
