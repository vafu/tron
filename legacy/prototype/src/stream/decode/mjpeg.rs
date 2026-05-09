use anyhow::{Context, Result};

use crate::stream::decode::FrameDecoder;
use crate::stream::frame::{EncodedFormat, EncodedFrame, Frame, PixelFormat};

pub struct TurboMjpegDecoder {
    decompressor: turbojpeg::Decompressor,
    output_format: PixelFormat,
    turbojpeg_format: turbojpeg::PixelFormat,
    buffer: Vec<u8>,
}

impl TurboMjpegDecoder {
    pub fn new(output_format: PixelFormat) -> Result<Self> {
        let turbojpeg_format = turbojpeg::PixelFormat::try_from(output_format)?;
        Ok(Self {
            decompressor: turbojpeg::Decompressor::new()?,
            output_format,
            turbojpeg_format,
            buffer: Vec::new(),
        })
    }
}

impl FrameDecoder for TurboMjpegDecoder {
    fn decode<'a>(&'a mut self, frame: EncodedFrame<'_>) -> Result<Frame<'a>> {
        if frame.format != EncodedFormat::Mjpeg {
            anyhow::bail!("TurboJPEG decoder only supports MJPEG frames");
        }
        let header = self
            .decompressor
            .read_header(frame.data)
            .context("read MJPEG header")?;
        let header_width = header.width as u32;
        let header_height = header.height as u32;
        anyhow::ensure!(
            frame.meta.width == header_width && frame.meta.height == header_height,
            "MJPEG payload dimensions {}x{} do not match frame metadata {}x{}",
            header_width,
            header_height,
            frame.meta.width,
            frame.meta.height
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

impl TryFrom<PixelFormat> for turbojpeg::PixelFormat {
    type Error = anyhow::Error;

    fn try_from(format: PixelFormat) -> Result<Self> {
        match format {
            PixelFormat::Gray8 => Ok(turbojpeg::PixelFormat::GRAY),
            PixelFormat::Bgra8 => Ok(turbojpeg::PixelFormat::BGRA),
            PixelFormat::Yuyv422 => {
                anyhow::bail!("unsupported TurboJPEG output format: {format:?}")
            }
        }
    }
}
