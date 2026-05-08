//! POC: send arbitrary UVC Extension Unit control transfers via libusb.
//!
//! Usage:
//!   ir_xu [--iface N] <vid:pid> <unit> <selector> getlen
//!   ir_xu [--iface N] <vid:pid> <unit> <selector> get
//!   ir_xu [--iface N] <vid:pid> <unit> <selector> set <byte,byte,...>
//!   ir_xu [--iface N] <vid:pid> scan <unit>            # GET_LEN over selectors 0..255
//!
//! --iface selects the VideoControl interface number (default 0). Cameras with
//! both an RGB and IR sensor expose two VC interfaces — pick the right one.
//!
//! Examples:
//!   sudo ir_xu 17ef:4839 6 1 getlen
//!   sudo ir_xu --iface 2 3443:c803 4 6 get
//!   sudo ir_xu 17ef:4839 6 1 set 1,3,1,0,0,0,0,0,0

use anyhow::{Context, Result, bail};
use rusb::UsbContext;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_millis(500);

const REQ_SET_CUR: u8 = 0x01;
const REQ_GET_CUR: u8 = 0x81;
const REQ_GET_MIN: u8 = 0x82;
const REQ_GET_MAX: u8 = 0x83;
const REQ_GET_RES: u8 = 0x84;
const REQ_GET_LEN: u8 = 0x85;
const REQ_GET_DEF: u8 = 0x87;

const BM_IN: u8 = 0xa1; // device-to-host, class, interface
const BM_OUT: u8 = 0x21; // host-to-device, class, interface

fn parse_vidpid(s: &str) -> Result<(u16, u16)> {
    let (v, p) = s.split_once(':').context("expected vid:pid")?;
    Ok((u16::from_str_radix(v, 16)?, u16::from_str_radix(p, 16)?))
}

fn parse_payload(s: &str) -> Result<Vec<u8>> {
    s.split(',')
        .map(|x| x.trim().parse::<u8>().context("bad u8"))
        .collect()
}

struct Xu {
    handle: rusb::DeviceHandle<rusb::Context>,
    iface: u8,
}

impl Xu {
    fn open(vid: u16, pid: u16, iface: u8) -> Result<Self> {
        let ctx = rusb::Context::new()?;
        let handle = ctx
            .devices()?
            .iter()
            .find(|d| {
                d.device_descriptor()
                    .map(|desc| desc.vendor_id() == vid && desc.product_id() == pid)
                    .unwrap_or(false)
            })
            .with_context(|| format!("device {vid:04x}:{pid:04x} not found"))?
            .open()
            .context("open device — try sudo")?;
        let _ = handle.set_auto_detach_kernel_driver(true);
        handle
            .claim_interface(iface)
            .with_context(|| format!("claim VC interface {iface}"))?;
        Ok(Self { handle, iface })
    }

    fn w_value(selector: u8) -> u16 {
        (selector as u16) << 8
    }

    fn w_index(&self, unit: u8) -> u16 {
        ((unit as u16) << 8) | (self.iface as u16)
    }

    fn get_len(&self, unit: u8, selector: u8) -> Result<u16> {
        let mut buf = [0u8; 2];
        let n = self.handle.read_control(
            BM_IN,
            REQ_GET_LEN,
            Self::w_value(selector),
            self.w_index(unit),
            &mut buf,
            TIMEOUT,
        )?;
        if n != 2 {
            bail!("GET_LEN returned {n} bytes, expected 2");
        }
        Ok(u16::from_le_bytes(buf))
    }

    fn read_q(&self, unit: u8, selector: u8, req: u8, len: u16) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len as usize];
        let n = self.handle.read_control(
            BM_IN,
            req,
            Self::w_value(selector),
            self.w_index(unit),
            &mut buf,
            TIMEOUT,
        )?;
        buf.truncate(n);
        Ok(buf)
    }

    fn set_cur(&self, unit: u8, selector: u8, payload: &[u8]) -> Result<usize> {
        Ok(self.handle.write_control(
            BM_OUT,
            REQ_SET_CUR,
            Self::w_value(selector),
            self.w_index(unit),
            payload,
            TIMEOUT,
        )?)
    }
}

impl Drop for Xu {
    fn drop(&mut self) {
        let _ = self.handle.release_interface(2);
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage:\n  {0} <vid:pid> <unit> <selector> getlen\n  \
             {0} <vid:pid> <unit> <selector> get\n  \
             {0} <vid:pid> <unit> <selector> set <b,b,...>\n  \
             {0} <vid:pid> scan <unit>",
            args[0]
        );
        std::process::exit(1);
    }

    let (vid, pid) = parse_vidpid(&args[1])?;
    let xu = Xu::open(vid, pid, 2)?;

    if args[2] == "scan" {
        let unit: u8 = args[3].parse()?;
        let mut found = 0;
        for sel in 0u8..=255 {
            match xu.get_len(unit, sel) {
                Ok(len) => {
                    let cur = xu.read_q(unit, sel, REQ_GET_CUR, len).ok();
                    println!(
                        "  unit={unit} selector={sel} len={len} cur={:?}",
                        cur.unwrap_or_default()
                    );
                    found += 1;
                }
                Err(_) => {}
            }
        }
        println!("found {found} controls on unit {unit}");
        return Ok(());
    }

    let unit: u8 = args[2].parse()?;
    let selector: u8 = args[3].parse()?;
    let op = &args[4];

    match op.as_str() {
        "getlen" => {
            let len = xu.get_len(unit, selector)?;
            println!("GET_LEN unit={unit} sel={selector} -> {len} bytes");
        }
        "get" => {
            let len = xu.get_len(unit, selector)?;
            let cur = xu.read_q(unit, selector, REQ_GET_CUR, len)?;
            let min = xu.read_q(unit, selector, REQ_GET_MIN, len).ok();
            let max = xu.read_q(unit, selector, REQ_GET_MAX, len).ok();
            let res = xu.read_q(unit, selector, REQ_GET_RES, len).ok();
            let def = xu.read_q(unit, selector, REQ_GET_DEF, len).ok();
            println!("unit={unit} selector={selector} len={len}");
            println!("  cur={cur:?}");
            if let Some(v) = min {
                println!("  min={v:?}");
            }
            if let Some(v) = max {
                println!("  max={v:?}");
            }
            if let Some(v) = res {
                println!("  res={v:?}");
            }
            if let Some(v) = def {
                println!("  def={v:?}");
            }
        }
        "set" => {
            if args.len() < 6 {
                bail!("set requires payload, e.g. 1,3,1,0,0,0,0,0,0");
            }
            let payload = parse_payload(&args[5])?;
            let n = xu.set_cur(unit, selector, &payload)?;
            println!("SET_CUR unit={unit} sel={selector} sent {n} bytes: {payload:?}");
        }
        _ => bail!("unknown op: {op}"),
    }

    Ok(())
}
