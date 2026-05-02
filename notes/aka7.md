# Task investigate IR strobe timing

Timestamp: 2026-05-02T13:22:25-07:00

Tags: #tron #task

Task:

Measure the IR flashlight strobe pattern and effective usable frame rate.

Why:

The current assumption is that a 30 fps IR camera yields only about 15 fps of lit frames because the emitter strobes. Pointer interaction needs lower latency and smoother motion than that if IR is a primary signal.

Experiment:

- Log IR frame sequence, timestamp, mean intensity, and flashlight-on classification.
- Estimate on/off cadence and jitter.
- Check whether RGB and IR timestamps can be aligned well enough to use stale lit frames.
- Test whether camera controls or linux-enable-ir-emitter settings can alter strobe cadence.

Output:

A note with measured cadence, effective lit FPS, and whether IR should be used for per-frame tracking or lower-rate correction/reacquisition.

