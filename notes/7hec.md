# RGB IR resolution and FOV calibration problem

# RGB IR resolution and FOV calibration problem

Timestamp: 2026-05-07T18:34:19-07:00

Tags: #tron #task

Question / hypothesis: Current RGB/IR tracking errors may come from mismatched
camera FOV/resolution and possibly an incorrect RGB aspect/proportion in the
capture/render path.

Observed result: Resolution selection is currently static/preferred inside
`camera::select`: RGB prefers YUYV 640x480; IR prefers GREY 640x360 then
640x480. The V4L backend requests that size and uses the negotiated result for
captured `Image` dimensions, while the renderer texture sizes are still created
from the selected config. `--list-cameras` currently reports Lenovo as RGB
640x480 + IR 640x480 and NexiGo as RGB 640x480 + IR 640x360. Direct
`v4l2-ctl --list-formats-ext` for `/dev/video53` and `/dev/video49` failed with
`Cannot open device`, so exact available format lists were not confirmed in this
pass.

Follow-up tasks / instrumentation gaps: Publish selected vs negotiated stream
size/format/stride in diagnostics. Add a calibration view that can overlay RGB
landmarks/ROI transformed into IR space, then replace the current affine-only
mapping with a measured RGB->IR calibration model.
