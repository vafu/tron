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

