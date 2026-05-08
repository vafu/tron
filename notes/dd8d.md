# Stereo calibration can support depth cue test

# Stereo calibration can support depth cue test

Timestamp: 2026-05-07T19:42:45-07:00

Tags: #tron

Question / hypothesis: Can the RGB/IR stereo-calibration derivation also become
a depth cue for hand tracking?

Observed result: User identified that once true RGB/IR stereo calibration exists,
the parallax/disparity between the two camera views can be leveraged for depth,
not just alignment. This would complement the transient IR brightness cue and
landmark-scale prior.

Follow-up tasks / instrumentation gaps: After stereo calibration capture works,
test sparse depth from known correspondences: checkerboard corners first, then
hand landmarks/ROI features if reliable in both RGB and IR. Compare stereo depth
against IR brightness transient signal and proximity/landmark-scale cues.
