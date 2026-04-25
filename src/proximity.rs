use anyhow::{anyhow, Context, Result};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub type SharedProx = Arc<Mutex<Option<i64>>>;

pub fn spawn(device: &str, channel: &str) -> Result<SharedProx> {
    let device = device.to_string();
    let channel = channel.to_string();
    let shared: SharedProx = Arc::new(Mutex::new(None));
    let out = shared.clone();

    thread::Builder::new()
        .name("prox".into())
        .spawn(move || {
            if let Err(e) = run(&device, &channel, shared) {
                eprintln!("proximity thread exited: {e:#}");
            }
        })?;
    Ok(out)
}

fn run(device_name: &str, channel_id: &str, shared: SharedProx) -> Result<()> {
    let ctx = industrial_io::Context::new().context("create iio context")?;
    let dev = ctx
        .find_device(device_name)
        .ok_or_else(|| anyhow!("iio device {device_name:?} not found"))?;
    let chan = dev
        .find_channel(channel_id, industrial_io::Direction::Input)
        .ok_or_else(|| anyhow!("iio channel {channel_id:?} not found"))?;

    loop {
        match chan.attr_read_int("raw") {
            Ok(v) => {
                *shared.lock().unwrap() = Some(v);
            }
            Err(e) => eprintln!("prox read err: {e}"),
        }
        thread::sleep(Duration::from_millis(50));
    }
}
