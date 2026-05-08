# Camera mode selection should preserve RGB IR aspect

# Camera mode selection should preserve RGB IR aspect

Timestamp: 2026-05-07T18:36:51-07:00

Tags: #tron #task

Question / hypothesis: Hardcoded RGB/IR preferred resolutions can produce wrong
RGB proportions when paired with an IR stream of a different aspect ratio. The
camera selector should choose from the modes advertised by each camera and
prefer RGB/IR pairs with matching aspect ratios.

Observed result: Replaced independent preferred-size selection with pair-wise
mode selection. The selector now enumerates all advertised YUYV RGB and GREY IR
modes, scores RGB/IR mode pairs by aspect-ratio match first and VGA-class pixel
count second, then chooses the best pair. `--list-cameras` now reports NexiGo as
RGB `/dev/video49` 960x540 paired with IR `/dev/video51` 640x360; Lenovo remains
RGB `/dev/video53` 640x480 paired with IR `/dev/video55` 640x480.

Follow-up tasks / instrumentation gaps: Add selected-vs-negotiated stream size
to HTTP diagnostics, and handle V4L `bytesperline` explicitly to rule out
stride-related distortion in YUYV decode.

Update: 2026-05-07T18:44:28-07:00

Question / hypothesis: Lenovo RGB still looks morphed because the first
pair-wise selector incorrectly forced RGB toward IR's 4:3 aspect, even though
Lenovo RGB advertises native 16:9 modes.

Observed result: Detailed mode listing showed Lenovo RGB has YUYV 16:9 modes up
to 1920x1080, while Lenovo IR only has GREY 640x480. Revised selection to choose
RGB and IR independently from each stream's advertised native aspect instead of
matching RGB aspect to IR aspect. `--list-cameras` now selects Lenovo RGB
`/dev/video53` 1024x576 and IR `/dev/video55` 640x480; NexiGo remains RGB
960x540 and IR 640x360.

Follow-up tasks / instrumentation gaps: Verify Lenovo live view geometry. RGB
and IR will now have different aspect/FOV by design, so the next calibration
step should estimate an RGB->IR transform rather than trying to make modes
match.
