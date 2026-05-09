use super::{CameraBackend, SharedImage, StreamConfig, StreamFormat};
use crate::types::{Image, PixelFormat};
use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use v4l::FourCC;
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::video::capture::Parameters;

pub struct Backend;

impl CameraBackend for Backend {
    fn spawn_stream(&self, config: StreamConfig) -> Result<SharedImage> {
        let (fourcc, decoder) = match config.format {
            StreamFormat::Rgb => (FourCC::new(b"YUYV"), Decoder::Yuyv),
            StreamFormat::Ir => (FourCC::new(b"GREY"), Decoder::Grey),
        };
        spawn(config, fourcc, decoder)
    }
}

#[derive(Copy, Clone)]
enum Decoder {
    Yuyv,
    Grey,
}

fn spawn(config: StreamConfig, fourcc: FourCC, decoder: Decoder) -> Result<SharedImage> {
    let shared: SharedImage = Arc::new(Mutex::new(None));
    let out = shared.clone();
    let path = config.path.clone();

    thread::Builder::new()
        .name(format!("cap:{path}"))
        .spawn(move || {
            if let Err(e) = run(&config, fourcc, decoder, shared) {
                eprintln!("camera {} thread exited: {e:#}", config.path);
            }
        })?;
    Ok(out)
}

fn run(config: &StreamConfig, fourcc: FourCC, decoder: Decoder, shared: SharedImage) -> Result<()> {
    let dev = Device::with_path(&config.path).with_context(|| format!("open {}", config.path))?;
    let mut fmt = dev.format()?;
    fmt.width = config.width;
    fmt.height = config.height;
    fmt.fourcc = fourcc;
    let fmt = dev.set_format(&fmt)?;
    if let Some(fps) = config.fps {
        match dev.set_params(&Parameters::with_fps(fps)) {
            Ok(params) => tracing::debug!(
                target: "tron::camera",
                path = %config.path,
                requested_fps = fps,
                interval = %params.interval,
                "camera frame interval configured"
            ),
            Err(err) => tracing::warn!(
                target: "tron::camera",
                path = %config.path,
                requested_fps = fps,
                error = %err,
                "camera frame interval request failed"
            ),
        }
    }
    eprintln!(
        "{}: negotiated {}x{} {}",
        config.path, fmt.width, fmt.height, fmt.fourcc
    );

    let w = fmt.width;
    let h = fmt.height;

    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4)?;
    // We always publish RGBA (camera frames double as a renderable texture).
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let mut seq: u64 = 0;
    let mut last_log = Instant::now();
    let mut frames: u32 = 0;
    let mut decode_us: u64 = 0;
    let mut publish_us: u64 = 0;

    loop {
        let (buf, _meta) = stream.next()?;
        let t_decode = Instant::now();
        {
            let _span = tracing::debug_span!("camera.decode", path = %config.path).entered();
            match decoder {
                Decoder::Yuyv => yuyv_to_rgba(buf, &mut rgba),
                Decoder::Grey => grey_to_rgba(buf, &mut rgba),
            }
        }
        decode_us += t_decode.elapsed().as_micros() as u64;
        seq = seq.wrapping_add(1);
        let t_publish = Instant::now();
        {
            let _span = tracing::debug_span!("camera.publish", path = %config.path, seq).entered();
            let img = Image {
                data: rgba.clone(),
                width: w,
                height: h,
                format: PixelFormat::Rgba8,
                timestamp: Instant::now(),
                seq,
            };
            *shared.lock().unwrap() = Some(img);
        }
        publish_us += t_publish.elapsed().as_micros() as u64;
        frames += 1;

        if last_log.elapsed() >= Duration::from_secs(2) {
            let n = frames.max(1) as f32;
            tracing::debug!(
                target: "tron::camera",
                path = %config.path,
                fps = frames as f32 / last_log.elapsed().as_secs_f32(),
                decode_ms = decode_us as f32 / n / 1000.0,
                publish_ms = publish_us as f32 / n / 1000.0,
                "camera timing"
            );
            last_log = Instant::now();
            frames = 0;
            decode_us = 0;
            publish_us = 0;
        }
    }
}

fn yuyv_to_rgba(yuyv: &[u8], rgba: &mut [u8]) {
    for (i, c) in yuyv.chunks_exact(4).enumerate() {
        let y0 = c[0] as i32;
        let u = c[1] as i32 - 128;
        let y1 = c[2] as i32;
        let v = c[3] as i32 - 128;
        let (r0, g0, b0) = yuv_to_rgb(y0, u, v);
        let (r1, g1, b1) = yuv_to_rgb(y1, u, v);
        let o = i * 8;
        rgba[o] = r0;
        rgba[o + 1] = g0;
        rgba[o + 2] = b0;
        rgba[o + 3] = 255;
        rgba[o + 4] = r1;
        rgba[o + 5] = g1;
        rgba[o + 6] = b1;
        rgba[o + 7] = 255;
    }
}

fn grey_to_rgba(grey: &[u8], rgba: &mut [u8]) {
    for (i, &g) in grey.iter().enumerate() {
        let o = i * 4;
        rgba[o] = g;
        rgba[o + 1] = g;
        rgba[o + 2] = g;
        rgba[o + 3] = 255;
    }
}

fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
    let c = y - 16;
    let r = (298 * c + 409 * v + 128) >> 8;
    let g = (298 * c - 100 * u - 208 * v + 128) >> 8;
    let b = (298 * c + 516 * u + 128) >> 8;
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}
