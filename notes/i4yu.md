# Task split rendering modules

Timestamp: 2026-05-02T16:32:43-07:00

Tags: #tron #task

Task:

Split oversized rendering modules so future graphics and interaction work has
clear component boundaries.

Question / hypothesis:

`gfx/mod.rs` and the standalone `skeleton_render` module had grown large enough
to slow navigation. Moving component renderers and shaders into focused files
should make the demo easier to extend without changing behavior.

Observed result:

- `gfx/mod.rs` is now the module/type hub.
- Renderer initialization moved to `gfx/setup.rs`.
- Frame rendering and timing moved to `gfx/render.rs`.
- Cube rendering moved to `gfx/cube.rs`, mesh/math helpers to
  `gfx/cube/mesh.rs`, and WGSL to `gfx/cube.wgsl`.
- Texture and solid quad helpers moved to `gfx/texture.rs`.
- Depth texture moved to `gfx/depth.rs`.
- `skeleton_render` moved under `gfx/skeleton`, split into renderer,
  geometry, label, and WGSL files.

Follow-up tasks / instrumentation gaps:

- Consider splitting `roi/detector.rs` if detector tuning starts expanding.
- Keep future render components under `gfx/` instead of adding new top-level
  rendering modules.
