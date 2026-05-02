# Task profile CPU usage

Timestamp: 2026-05-02T14:53:22-07:00

Tags: #tron #task

Question / hypothesis:

The live demo is consuming about 88% CPU across four cores. The likely sources
are the always-polling winit event loop, continuous redraw requests, per-frame
camera buffer cloning/conversion, and per-frame ONNX detector/landmarker work.

Observed result:

Instrumentation was added and a 10-second live run was captured with
`--classifier-only`. The current code now reports rolling two-second timings
for:

- event-loop wait/redraw rate
- render lock/upload/overlay/encode/submit cost
- V4L camera decode and publish cost per stream
- palm-detector tensor prep, ONNX runtime, and output decode cost

Measured on 2026-05-02T14:54:00-07:00:

- event loop redraws at about 60 FPS
- render timing is about 60 FPS; `submit_ms` is about 16 ms, consistent with
  vsync wait rather than CPU-bound rendering
- RGB camera publishes about 30 FPS; YUYV decode is about 0.9-1.0 ms/frame
- IR camera publishes about 15 FPS; GREY decode is about 0.4-0.55 ms/frame
- palm detector runs about 30 FPS; tensor prep is about 1.8-1.9 ms/frame and
  ONNX Runtime is about 17.7-19.7 ms/frame
- pipeline trace shows total frame time about 25-40 ms, with landmarker around
  6.3-9.5 ms/frame

Initial conclusion:

The abnormal CPU load is most likely dominated by running palm detection plus
landmark inference on every RGB frame. Continuous 60 FPS rendering is present,
but WGPU submit time appears to be mostly waiting on presentation. Camera
conversion/cloning is measurable but not the main cost.

2026-05-02T15:08:00-07:00 update:

Perfetto output was wired through `tracing-perfetto` and verified with a short
live run. The trace file was written to `/tmp/tron.pftrace` and includes spans
for camera decode/publish, pipeline stages, ROI detector prep/ORT/decode,
MediaPipe landmarker prep/ORT/decode, and render lock/upload/overlay/encode/
submit.

2026-05-02T15:25:00-07:00 update:

Perfetto UI loading was investigated. The public UI does not require uploading
the trace to a remote service; Perfetto's own `open_trace_in_ui` helper serves
the trace from `127.0.0.1:9001` and opens `https://ui.perfetto.dev` with a
`url=` query parameter. `tron` now mirrors that flow with `--perfetto-open`.
On shutdown it spawns a helper mode, exits to close the trace file, and the
helper serves the completed trace once while opening the browser.

Run command:

```sh
RUST_LOG=tron::event_loop=debug,tron::gfx=debug,tron::camera=debug,tron::roi=debug cargo run -- --classifier-only
```

Perfetto command:

```sh
RUST_LOG=tron=debug cargo run -- --classifier-only --perfetto /tmp/tron.pftrace
```

Perfetto auto-open command:

```sh
RUST_LOG=tron=debug cargo run -- --classifier-only --perfetto /tmp/tron.pftrace --perfetto-open
```

Follow-up tasks / instrumentation gaps:

- Add a pipeline-stage timing summary for landmarker, filter, and classifier
  similar to the detector summary.
- Try ROI tracking reuse so the palm detector does not run every frame after
  lock-on.
- Cap classifier/research mode to camera FPS and avoid 60 FPS redraws when no
  visible state changed.
- Consider moving detector inference to a lower cadence, e.g. 10-15 FPS, while
  tracking landmarks every frame.
- Measure landmarker/filter/pipeline timing separately after the first pass.
- Measure CPU after disabling cube rendering with `--classifier-only`.
