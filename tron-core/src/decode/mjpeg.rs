use anyhow::{Context, Result};
use tron_api::{Frame, PixelFormat};

use crate::decode::{EncodedFormat, EncodedFrame, FrameDecoder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MjpegHeader {
    pub width: usize,
    pub height: usize,
}

pub struct TurboMjpegDecoder {
    decompressor: turbojpeg::Decompressor,
    output_format: PixelFormat,
    turbojpeg_format: turbojpeg::PixelFormat,
    buffer: Vec<u8>,
}

impl TurboMjpegDecoder {
    pub fn new(output_format: PixelFormat) -> Result<Self> {
        let turbojpeg_format = turbojpeg_pixel_format(output_format)?;
        Ok(Self {
            decompressor: turbojpeg::Decompressor::new()?,
            output_format,
            turbojpeg_format,
            buffer: Vec::new(),
        })
    }

    pub fn read_header(&mut self, data: &[u8]) -> Result<MjpegHeader> {
        let header = self
            .decompressor
            .read_header(data)
            .context("read MJPEG header")?;
        Ok(MjpegHeader {
            width: header.width,
            height: header.height,
        })
    }
}

impl FrameDecoder for TurboMjpegDecoder {
    fn decode<'a>(&'a mut self, frame: EncodedFrame<'_>) -> Result<Frame<'a>> {
        if frame.format != EncodedFormat::Mjpeg {
            anyhow::bail!("TurboJPEG decoder only supports MJPEG frames");
        }
        let header = self.read_header(frame.data)?;
        let header_width = header.width as u32;
        let header_height = header.height as u32;
        anyhow::ensure!(
            frame.meta.size.width == header_width && frame.meta.size.height == header_height,
            "MJPEG payload dimensions {}x{} do not match frame metadata {}x{}",
            header_width,
            header_height,
            frame.meta.size.width,
            frame.meta.size.height
        );

        let stride = header.width * self.turbojpeg_format.size();
        let required_len = stride * header.height;
        self.buffer.resize(required_len, 0);
        self.decompressor
            .decompress(
                frame.data,
                turbojpeg::Image {
                    pixels: self.buffer.as_mut_slice(),
                    width: header.width,
                    pitch: stride,
                    height: header.height,
                    format: self.turbojpeg_format,
                },
            )
            .context("decode MJPEG frame")?;

        Ok(Frame {
            meta: frame.meta,
            format: self.output_format,
            stride,
            data: &self.buffer,
        })
    }
}

fn turbojpeg_pixel_format(format: PixelFormat) -> Result<turbojpeg::PixelFormat> {
    match format {
        PixelFormat::Gray8 => Ok(turbojpeg::PixelFormat::GRAY),
        PixelFormat::Bgra8 => Ok(turbojpeg::PixelFormat::BGRA),
        PixelFormat::Yuyv422 => {
            anyhow::bail!("unsupported TurboJPEG output format: {format:?}")
        }
    }
}
