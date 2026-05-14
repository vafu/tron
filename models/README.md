# Models

Drop a MediaPipe Hands hand-landmark ONNX export here as `hand_landmark.onnx`.

The loader (`tron-core/src/roi/mediapipe/landmark.rs`) expects:
- input: NCHW `[1, 3, 224, 224]` f32 in `[0, 1]`, RGB
- output: 21×3 landmarks, exactly 63 f32 values, normalized 0..1 in input space (raw pixel space is also supported)
- output named like `*presence*` / `*score*`: scalar f32 confidence (optional)
- output named like `*handed*`: scalar f32 (>0.5 ⇒ right hand) (optional)

The landmark loader rejects detector models. A palm detector usually exposes
outputs like `box_coords` and `box_scores`; those are not landmark tensors and
must not be used as `hand_landmark.onnx`.

Sources to pick from (verify shapes against the contract above):
- PINTO_model_zoo: <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/033_Hand_Detection_and_Tracking>
- Qualcomm AI Hub: <https://huggingface.co/qualcomm/MediaPipe-Hand>
- `mediapipe` TFLite → ONNX via `tf2onnx` (the original Google weights)

Palm detection is wired separately under `models/hand_detector/model.onnx`.
