# AGENTS.md

## Project

Working title: **Leverage Windows Hello camera stack to use as a 3D/2D pointer system.**

`tron` is a research prototype for turning a Windows Hello-style camera stack
into a low-latency pointer system. The near-term demo is a 3D scene with:

- live RGB/IR camera debug views
- a cube rendered in 3D space
- a hand representation with tracked `x`, `y`, and `z`
- interaction logic that lets the hand drag the cube in 3D

The longer-term research goal is to learn which sensing/model pipeline can
support reliable 2D and 3D pointing. Expect custom model training once the
demo produces enough instrumentation and labeled examples.

## Research Notes

Use `zk` for research notes. This repo has a notebook at `notes/`.

Every research note must include:

- an ISO-8601 timestamp
- the `#tron` tag
- the question or hypothesis being tested
- the observed result, even if inconclusive
- follow-up tasks or instrumentation gaps

Task notes must include both `#tron` and `#task`.

Useful commands:

```sh
zk --notebook-dir notes -W notes new --title "Title" --print-path
zk --notebook-dir notes -W notes list --tag tron --format oneline
zk --notebook-dir notes -W notes list --tag tron --tag task --format oneline
zk --notebook-dir notes -W notes index
```

## Architecture Direction

Keep the runtime modular. The goal is to replace individual pieces without
rewriting the application shell.

Camera capture belongs behind `camera::CameraBackend`. The current backend is
V4L, but future backends may use libcamera, IPU6-specific paths, or direct
Windows Hello-adjacent device configuration.

ONNX Runtime setup belongs in `inference`. Execution provider choices, GPU/NPU
enablement, model session configuration, and shared runtime policy should live
there rather than in model-specific code.

Pipeline stages should remain trait-driven:

- `FrameContextRefiner`: sensor fusion and derived frame products
- `RoiHinter`: hand ROI acquisition and tracking
- `HandLandmarker`: landmark extraction
- `LandmarkFilter`: temporal smoothing/post-processing
- `GestureClassifier`: pointer/gesture interpretation

Rendering should stay separate from tracking logic. The 3D cube demo should
consume a hand/pointer state produced by the pipeline rather than reaching into
camera or model internals.

## Demo Milestones

1. Establish a `PointerState` with normalized `x`, `y`, `z`, confidence, and
   interaction state.
2. Render a 3D cube alongside the existing camera debug views.
3. Render a hand/pointer representation in the same 3D coordinate space.
4. Map hand landmarks plus IR/proximity cues into initial depth estimates.
5. Add grab/drag semantics for moving the cube.
6. Instrument latency, jitter, drift, confidence, and failure modes.
7. Capture data for custom-model research.

## Engineering Constraints

Prefer narrow changes that preserve the current live demo. Avoid hard-coding
research assumptions into shared types until an experiment has produced useful
evidence.

When adding a model or backend, expose it through the existing module boundary
first. If the boundary does not fit, adjust the boundary deliberately and note
the reason in `zk`.

Do not make hardware-specific code a dependency of the core pipeline. Keep it
behind camera, sensor, or inference adapters.
