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

Prefer building new architecture beside the current demo, then porting proven
pieces into `main`. `tron-pipeline` is the primary new executable for this
work. Older binaries such as `tron-stream` may contain research/playground
wiring; avoid turning them into larger catch-all modules.

Use `clap` derive for command-line interfaces. Avoid hand-rolled argument
parsers in project binaries.

Directory structure should reflect implementation structure. Use a plain module
file such as `stream/frame.rs` when the module is only vocabulary, types, or a
single trait surface. Use a module directory with `mod.rs` only when that
logical part has multiple implementation files behind the same trait, such as
`stream/decode/{mod.rs,mjpeg.rs}` or `stream/source/{mod.rs,v4l.rs}`.

Camera capture belongs behind `camera::CameraBackend`. The current backend is
V4L, but future backends may use libcamera, IPU6-specific paths, or direct
Windows Hello-adjacent device configuration.

The newer streaming architecture lives under `stream`. Keep the source,
decode, processing, and rendering concerns separate:

- `source`: capture backends produce `CapturedFrame`
- `decode`: decoders convert encoded frames into pixel-accessible frames
- `process`: processing traits consume frame/context views
- `render`: render sinks consume frame/context views

Capture format and pixel format are different concepts. `CaptureFormat`
describes what a source requests from hardware (`Mjpeg`, `Gray8`, `Yuyv422`).
`PixelFormat` describes pixel-accessible frame memory (`Gray8`, `Yuyv422`,
`Bgra8`). Do not put decoded/render-only formats such as `Bgra8` into source
configuration. Avoid enum shapes that permit invalid states; prefer distinct
types and `From`/`TryFrom` conversions.

Use `CapturedFrame::Encoded(EncodedFrame)` for compressed payloads and
`CapturedFrame::Frame(Frame)` for already pixel-accessible camera output.
Only encoded frames should go through `FrameDecoder`. GREY/YUYV V4L output is
already a `Frame`; do not add passthrough decoders for pixel-accessible camera
data.

Use borrowed frame views in hot paths. `Frame<'a>` is the normal
pixel-accessible view passed through processing and rendering. Owned storage,
such as `OwnedFrame`, is only backing storage and should expose a borrowed
frame view. Avoid cloning or allocating frame-sized buffers in per-frame paths.
Decoders should own and reuse pre-allocated buffers or buffer pools.

Prefer Rust conversion traits for format/adaptor boundaries:

- use `From`/`.into()` only for infallible conversions
- use `TryFrom`/`.try_into()?` for conversions that can reject a format
- use `derive_more::From` for repeated enum wrapper boilerplate

For V4L, source config is the requested mode; the driver-returned format is the
effective source of truth. After negotiation, update/validate effective width,
height, and FourCC. If V4L returns an unexpected format, fail loudly rather
than letting metadata lie.

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
