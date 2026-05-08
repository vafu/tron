---
title: "Task improve gesture classifier"
tags:
  - project/tron
  - kind/fleeting
  - status/inbox
---

# Task improve gesture classifier

Timestamp: 2026-05-02T14:19:12-07:00

Tags: #tron #task

Task:

Improve the gesture classifier after the RGB-first cube demo checkpoint.

Current issue:

The hand-rule classifier is brittle. Recent live testing showed inconsistent labels: fist could be read as thumbs-up, pinch could be read as fist, and pinch was not reliable enough for cube grabbing.

Immediate direction:

- Commit the current cube/pointer work as a checkpoint.
- Add live classifier diagnostics so tuning is based on observed features rather than guesswork.
- Prefer a smaller, explicit classifier focused on the demo states first: open/free and fist/grab.
- Keep pinch available for later, but do not depend on it for drag until measured features are stable.

Acceptance:

- Demo can reliably distinguish free hand from fist/grab.
- Window/debug output exposes enough classifier features to tune thresholds.
- Classifier changes are documented with observed behavior.

Progress:

2026-05-02T14:19:12-07:00

- Added `GestureFeatures` and `GestureClassification` so each classified frame carries tunable feature values.
- Window title now exposes `pinch`, `extended`, `curled`, `fist_score`, and `thumb_up`.
- Reworked the rule classifier to prioritize the demo states: fist/grab, open/free, point, then pinch as non-critical.
- Fist detection now uses a weighted score from curled-finger count and fingertip compactness around the palm.
- Added runtime flags for focused classifier work: `--classifier-only`, `--no-cube`, `--no-skeleton`, and `--no-classifier-debug`.

Next tuning loop:

Run the app and record the title values for open hand, fist, and accidental pinch-like poses. Tune `fist_score` and curl thresholds from those observed numbers.

2026-05-02T15:57:47-07:00

- Replaced the coarse curled-count fist rule with per-finger curl scores for
  index, middle, ring, and pinky.
- Added classifier hysteresis: fist now has a stricter enter threshold and a
  looser stay threshold to reduce frame-to-frame flicker while dragging.
- Fist classification now requires most fingers curled and at most one extended
  finger, so pinch-like poses should no longer become fist from compactness
  alone.
- Debug title now includes per-finger curl values as `c=[index,middle,ring,pinky]`.

Next tuning loop:

Use `--classifier-only` and capture title values for open hand, fist, point,
and the problematic pinch pose. If fist misses, lower fist enter/stay slightly;
if pinch still becomes fist, raise the per-finger curled threshold or require
all four fingers curled for fist entry.

2026-05-02T16:02:37-07:00

- Live classifier-only smoke test reached the camera/pipeline path after
  clearing stale incremental build artifacts from an interrupted `timeout`.
- The run showed ORT/WGPU info logs drowning out classifier diagnostics under
  the default tracing filter, so the default filter was tightened to
  `warn,tron=info`. `RUST_LOG=...` still overrides it for profiling.

2026-05-02T16:09:02-07:00

- Live feedback: pinch was not detected at all after the per-finger fist
  rewrite.
- Pinch no longer requires an extended-finger count and is evaluated before
  open/point. It still loses to a strong fist.
- Added pinch hysteresis: enter at `pinch <= threshold`, stay until
  `pinch <= threshold * 1.35`, unless the hand becomes a strong multi-finger
  fist.

2026-05-02T16:20:25-07:00

- Moved gesture feedback from the window title into the camera overlay.
- The skeleton renderer now draws a small bitmap label at the ROI top-right
  corner when a concrete gesture is detected.
- Window title is back to coarse process/proximity status only.

2026-05-02T16:26:20-07:00

- Added gesture color states to the skeleton/ROI/label overlay.
- Fist remains red for grab feedback.
- Pinch now renders green to distinguish it from fist while tuning classifier
  behavior.
