# Models

Drop a MediaPipe Hands hand-landmark ONNX export here as `hand_landmark.onnx`.

The loader (`src/landmarker/mediapipe.rs`) expects:
- input: NCHW `[1, 3, 224, 224]` f32 in `[0, 1]`, RGB
- output[largest]: 21×3 landmarks, normalized 0..1 in input space (auto-detected; raw pixel space also supported)
- output named like `*presence*` / `*score*`: scalar f32 confidence (optional)
- output named like `*handed*`: scalar f32 (>0.5 ⇒ right hand) (optional)

If the file is absent or fails to load, the app falls back to the deterministic
`MockLandmarker` so the rest of the pipeline keeps running.

Sources to pick from (verify shapes against the contract above):
- PINTO_model_zoo: <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/033_Hand_Detection_and_Tracking>
- Qualcomm AI Hub: <https://huggingface.co/qualcomm/MediaPipe-Hand>
- `mediapipe` TFLite → ONNX via `tf2onnx` (the original Google weights)

Palm detection (`palm_detection_full.onnx`) is **not yet wired** — IR ROI +
last-frame tracking are the current ROI sources. See plan file
`~/.claude/plans/ok-next-task-would-deep-fern.md` for the full design.
