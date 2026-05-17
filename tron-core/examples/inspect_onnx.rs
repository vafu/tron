use ort::session::Session;
use ort::value::ValueType;

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/google_hand_detector/model.onnx".to_string());
    let session = Session::builder()?.commit_from_file(model_path)?;

    println!("Inputs:");
    for (i, input) in session.inputs().iter().enumerate() {
        let shape = match input.dtype() {
            ValueType::Tensor { shape, .. } => format!("{:?}", shape),
            _ => "not a tensor".to_string(),
        };
        println!("  Index {}: name={}, shape={}", i, input.name(), shape);
    }

    println!("\nOutputs:");
    for (i, output) in session.outputs().iter().enumerate() {
        let shape = match output.dtype() {
            ValueType::Tensor { shape, .. } => format!("{:?}", shape),
            _ => "not a tensor".to_string(),
        };
        println!("  Index {}: name={}, shape={}", i, output.name(), shape);
    }

    if let Ok(metadata) = session.metadata() {
        println!("\nMetadata:");
        if let Ok(keys) = metadata.custom_keys() {
            for key in keys {
                if let Some(val) = metadata.custom(&key) {
                    println!("  {}: {:?}", key, val);
                }
            }
        }
    }

    Ok(())
}
