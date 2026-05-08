---
title: "Auto exposure compensated IR depth hypothesis"
tags:
  - project/tron
  - kind/fleeting
  - status/inbox
---

# Auto exposure compensated IR depth hypothesis

2026-05-07T17:24:35-07:00

#tron

Question/hypothesis: Can relative IR brightness still support depth if standard
manual exposure is unreliable, by modeling auto-exposure reaction rather than
requiring fixed exposure.

Observed result: User hypothesis: detect hand, compute successive brightness
diff masks, then adjust the diff by a calibrated model of how quickly the
sensor reacts. User observation from experiments: exposure seems to adjust
mainly when the sensor starts clipping.

Follow-up tasks or instrumentation gaps: Instrument per-frame hand ROI mean,
background mean, clipped-pixel fraction, strobe/on-off diff, and temporal
response after deliberate near/far hand motion. Test whether compensation can
be driven by clipped fraction and global/background brightness rather than
hidden exposure values.

