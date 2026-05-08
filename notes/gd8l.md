# IR brightness depth experiment integration plan

2026-05-07T17:33:09-07:00

#tron #task

Question/hypothesis: What trait and pipeline changes are needed to test
relative IR brightness depth with auto-exposure compensation, without
implementing yet.

Observed result: Existing `FrameContextRefiner` flow already produces
`ctx.ir_diff` before ROI/landmark stages. However, the proposed metrics need the
final hand ROI and/or landmarks, so metric extraction belongs after landmark
filtering/classification in `GesturePipeline::step`, not inside
`TemporalSubtractionRefiner`.

Observed result: No camera backend change is required for a first experiment.
IR GREY frames are currently expanded to RGBA, but `Image::grey_iter()` recovers
the luminance channel. A later optimization could preserve R8 IR frames.

Observed result: Likely type additions: `IrDepthMetrics` or `DepthCue` with
hand ROI signal, background signal, clipped fraction, corrected signal,
temporal delta, confidence, and maybe landmark scale. Add it to `HandState`
and/or a separate shared pipeline output for rendering/logging. `PointerState.z`
can remain unchanged until the metric is validated.

Follow-up tasks or instrumentation gaps: Add an experimental depth estimator
trait or helper after landmarks; map RGB ROI/landmarks into IR coordinates via
`calib`; log metrics per frame and expose a compact HUD/window-title/debug
output. Keep ROI, landmarker, filter, and gesture traits stable initially.

2026-05-07T17:46:42-07:00

#tron #task

Question/hypothesis: Implement the first-pass `DepthCueEstimator` architecture
without changing pointer behavior.

Observed result: Added `depth_cue` module with `DepthCueContext`,
`DepthCueEstimator`, and `IrBrightnessDepthEstimator`. `GesturePipeline` now
accepts an optional estimator and runs it after landmark filtering and gesture
classification. Metrics are attached to `HandState.ir_depth` and logged every
30 pipeline frames; window title debug shows corrected signal, delta, clipping,
and confidence when classifier debug is enabled.

Observed result: `IrBrightnessDepthEstimator` maps the RGB ROI to IR coordinates
using `calib::AffineCalib::unmap_rect`, computes hand/background statistics from
`ctx.ir_diff`, computes raw hand mean and clipping from `ctx.ir`, and derives a
background-normalized corrected signal plus temporal delta. `PointerState.z`
remains unchanged.

Follow-up tasks or instrumentation gaps: Run live near/far motion tests and
record whether `corrected_signal` is monotonic. If ROI padding contaminates hand
statistics, switch hand region from ROI rectangle to a landmark-derived mask or
tighter landmark bounding box.
