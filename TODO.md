# TODO

Things to try / fix later. Keep entries terse.

## Pipeline correctness
- Bilinear letterbox preprocessing in `MediaPipeHandLandmarker::preprocess` (currently nearest-neighbour stretch — distorts hand aspect).
- Square `square_crop` in **pixel** space, not normalized — pad rather than stretch when the ROI is non-square.
- Decouple gesture tick rate from RGB seq — RGB updates can starve when we want IR-rate inference.
- Reset `TrackFromLastRoi` when consecutive low-presence frames suggest the tracker has drifted.

## IR/RGB fusion experiments
- Proper IR↔RGB calibration via checkerboard homography (`opencv::calib3d`); replace `calib::IR_TO_RGB` affine.
- Background-suppression fusion: `rgb *= 0.5 + 0.5 * ir_mask` before inference, with aligned IR.
- Compare presence/jitter: vanilla RGB vs IR-mono vs fusion vs masked-RGB.
- Move "combined" buffer creation into the pipeline (single source of truth for what the model + renderer see); drop the duplicate RGB upload in `gfx.rs`.

## Models
- Wire `PalmDetectorRoi` (anchor decode + NMS) so far-field hands work without IR.
- Try alternate hand-landmark exports (PINTO_model_zoo, vladmandic/human) to compare stability.
- Quantized int8 variant — check accuracy vs latency trade-off.

## Performance
- OpenVINO execution provider on Meteor Lake iGPU/NPU.
- Move YUYV→RGB to a wgpu compute shader; skip the CPU step + RGBA upload.
- Reuse the IR R8 buffer instead of triplicating to RGBA; sample with R8 swizzle in the shader.

## UX / debug
- Real on-screen text (glyphon or fontdue) for gesture label and stats.
- Live-tunable calibration constants (keyboard arrows nudge `IR_TO_RGB`).
- FPS / latency HUD (capture→publish ms in the gesture thread).

## Robustness
- Two-handed support — drop the single-hand assumption from `MockLandmarker` API too.
- Recover gracefully when a camera disappears (USB unplug) instead of thread-exit.
- Log presence/handedness once per second only when the value class changes.
