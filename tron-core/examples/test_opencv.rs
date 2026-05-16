use opencv::core::{self, Mat, Vec4b};
use opencv::imgproc;
use opencv::prelude::*;

fn main() {
    let data = vec![Vec4b::all(0); 100 * 100];
    let src = Mat::new_rows_cols_with_data(100, 100, &data).unwrap();
    let mut resized = Mat::default();
    imgproc::resize(
        &src,
        &mut resized,
        core::Size::new(50, 50),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .unwrap();
    let mut padded = Mat::default();
    core::copy_make_border(
        &resized,
        &mut padded,
        25,
        25,
        25,
        25,
        core::BORDER_CONSTANT,
        core::Scalar::all(0.0),
    )
    .unwrap();
    let bytes = padded.data_bytes().unwrap();
    println!("{}", bytes.len());
}
