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

Use `remarked-cli` for research notes. Always target this project with
`--proj tron`.

As work progresses, save important notes with `remarked-cli`. Do this when
discovering hardware behavior, making or changing research assumptions,
choosing an implementation direction, observing model/pipeline behavior,
finding a bug or limitation that affects future work, or leaving a follow-up
task for the next session. Prefer short notes captured close to the observation
over reconstructing context later.

Do not log administrative events. Notes should be related to actual research or
technical decisions.

Every research note must include:

- an ISO-8601 timestamp
- the question or hypothesis being tested
- the observed result, even if inconclusive
- follow-up tasks or instrumentation gaps

Always set a meaningful `--title`. Do not include a markdown `# title` heading
inside the note body.

Always set `--tags` to include `codex` plus extra comma-separated tags that
describe the work, such as `calibration`, `tracking`, `depth`, `camera`,
`diagnostics`, or `task`.

Useful commands:

```sh
remarked-cli --proj tron --title "Meaningful title" --tags "codex,calibration" "Timestamp: $(date --iso-8601=seconds)

Question / hypothesis: ...

Observed result: ...

Follow-up tasks / instrumentation gaps: ..."
```

Quick disposable-note check:

```sh
remarked-cli --proj tron --title "Codex remarked test" --tags "codex,test" "Timestamp: $(date --iso-8601=seconds)

Question / hypothesis: Verify remarked-cli can create notes for tron from Codex.

Observed result: Test note created through remarked-cli.

Follow-up tasks / instrumentation gaps: Remove this disposable test note after verification."
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
the reason with `remarked-cli`.

Do not make hardware-specific code a dependency of the core pipeline. Keep it
behind camera, sensor, or inference adapters.
