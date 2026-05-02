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
