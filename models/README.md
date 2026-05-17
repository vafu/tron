# Models

The default RGB hand pipeline uses Google MediaPipe task weights converted from
TFLite to ONNX with `tf2onnx`:

- palm detector: `models/google_hand_detector/model.onnx`
- hand landmarks: `models/google_hand_landmark/hand_landmark.onnx`

The landmark loader (`tron-core/src/roi/mediapipe/landmark.rs`) accepts:

- NCHW `[1, 3, H, W]` or NHWC `[1, H, W, 3]` f32 input in `[0, 1]`, RGB
- output: 21x3 landmarks, exactly 63 f32 values, normalized 0..1 in input space
  or raw pixel space
- output named like `*presence*` / `*score*`: scalar f32 confidence (optional)
- output named like `*handed*`: scalar f32 (>0.5 means right hand) (optional)

The landmark loader rejects detector models. Palm detection is wired separately
and expects detector outputs compatible with `tron-core/src/roi/mediapipe/palm.rs`.
