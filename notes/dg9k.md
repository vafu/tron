# Observation IR mask improves hand edges near face

Timestamp: 2026-05-02T17:13:51-07:00

Tags: #tron

Question / hypothesis:

Does re-enabling the IR foreground mask help MediaPipe tracking enough to keep
using it as an auxiliary indoor signal?

Observed result:

Yes, in the current indoor setup the IR mask appears useful when the hand is in
front of the face. The hand edges are noticeably clearer in the masked/debug
view, which should give ROI and landmark stages a cleaner separation between
hand, face, and background.

Attached visual evidence:

- Conversation screenshot Image #1: full app view showing IR/masked RGB panes
  and the tracked open hand.
- Conversation screenshot Image #2: cropped main view showing the hand skeleton
  over RGB with clearer hand boundaries when the hand overlaps the face.

Follow-up tasks / instrumentation gaps:

- Save future screenshots into `notes/assets/` or another repo-local path so
  they can be embedded directly in zk notes.
- Compare tracking stability with `I` toggled on and off while the hand crosses
  the face.
- Investigate whether IR camera exposure can be controlled or stabilized.
