---
title: "Task implement RGB first cube demo"
tags:
  - project/tron
  - kind/fleeting
  - status/inbox
---

# Task implement RGB first cube demo

Timestamp: 2026-05-02T13:48:13-07:00

Tags: #tron #task

Task:

Implement the first RGB-primary pointer foundation and cube interaction demo.

Implementation notes:

- Keep `FrameContext` multimodal with both RGB and IR frames captured.
- Use RGB as the active source for palm ROI and landmark input.
- Keep IR foreground generation available as auxiliary/debug data, but remove direct IR masking from the default tracking path.
- Add a generic `PointerState` with `x`, `y`, reserved `z = 0`, confidence, timestamp, and pinch/grab state.
- Add a real WGPU cube renderer and rotate the cube only while pinch is active.

Acceptance:

- `cargo check` passes.
- `cargo fmt --check` passes.
- The app still renders camera/debug panes and skeleton overlays.
- The cube renders in the main window.
- Pinch publishes `grabbed = true` and drives cube rotation from pointer x/y deltas.

