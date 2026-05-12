use tron_api::Size;

#[derive(Clone, Debug)]
pub struct FrameProjectionMap {
    pub input_size: Size,
    pub output_size: Size,
    pub pixels: Vec<Option<(u32, u32)>>,
}

impl FrameProjectionMap {
    pub fn get(&self, x: u32, y: u32) -> Option<(u32, u32)> {
        if x >= self.output_size.width || y >= self.output_size.height {
            return None;
        }
        self.pixels[y as usize * self.output_size.width as usize + x as usize]
    }
}
