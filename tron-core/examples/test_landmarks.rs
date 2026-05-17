use ort::session::Session;
use ort::value::TensorRef;

fn main() -> anyhow::Result<()> {
    let model_path = "../models/google_hand_landmark/hand_landmark.onnx";
    let mut session = Session::builder()?.commit_from_file(model_path)?;

    let input_size = 256;
    let input = vec![0.5f32; 3 * input_size * input_size];
    let tensor = TensorRef::from_array_view(([1, 3, input_size, input_size], &*input))?;

    let outputs = session.run(vec![("image", tensor)])?;
    let landmarks = outputs["landmarks"].try_extract_tensor::<f32>()?.1;

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in landmarks.iter() {
        min = min.min(v);
        max = max.max(v);
    }

    println!("Landmarks range: [{}, {}]", min, max);
    println!("First 6 values: {:?}", &landmarks[..6]);

    Ok(())
}
