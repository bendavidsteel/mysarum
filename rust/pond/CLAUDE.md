# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Pond is a real-time simulation combining particle life with Graph Rewriting Automata (GRA) — small graph structures that grow via discrete cellular automata rules on a toroidal 2D world. Particles and GRA nodes interact through repulsion forces. The system includes GPU-accelerated audio synthesis driven by simulation state.

Built with [nannou](https://nannou.cc/) (creative coding framework) using a local fork at `/home/ben/Projects/personal/repos/nannou/`.

## Build & Run

```bash
cargo build --release
cargo run --release
```

Uses the `mold` linker on Linux (configured in `.cargo/config.toml`). Debug builds work but are significantly slower for the GPU compute pipeline setup.

## Architecture

**nannou app lifecycle** (`main.rs`): Standard `model → update → render` loop.
- `model()` — initializes window, audio stream, GRA trials, and simulation parameters
- `update()` — runs CPU-side logic (Barnes-Hut repulsion, trial management, discrete CA steps), then dispatches GPU compute passes and handles audio buffer management
- `render::render()` — GPU render pass via instanced quads

**GPU compute** (`gpu.rs`): All GPU buffers, pipelines, and dispatch functions. Shaders are concatenated at compile time via `include_str!` with a `shader_with_common!` macro that prepends `common.wgsl` to each shader.

**Compute pipeline stages** (all in `src/shaders/`):
1. Spatial binning + prefix-sum sort (separate for particles and GRA nodes)
2. Particle forces (particle↔particle interaction + particle↔GRA repulsion)
3. GRA spring forces + integration
4. Phase/oscillator updates for audio synthesis
5. Audio buffer generation on GPU, staged readback to CPU ring buffer

**Barnes-Hut** (`barnes_hut.rs`): CPU-side O(N log N) repulsion between GRA nodes using Morton-code sorted flat quadtree with dual-tree traversal. Force law: `f = charge * dx / (1 + r²)`.

**Rendering** (`render.rs`): Instanced quad rendering with separate pipelines for particles (point sprites), GRA edges (line-style quads), and GRA nodes (discs with state-based coloring).

**GRA trial system**: Multiple independent graph rewriting trials run simultaneously. Each trial has random discrete CA rules (state transitions + topology actions: no-op/split/merge). Trials are culled if they die out or grow too large; new ones spawn periodically. Graph topology loaded from `graphs.json`.

## Key data flow

- GRA node repulsion: CPU Barnes-Hut → positions uploaded to GPU each frame
- GRA spring forces + integration: GPU compute
- Particle simulation: entirely GPU (spatial hash binning → force computation → integration)
- Audio: GPU compute generates samples → staged buffer readback → ring buffer → audio thread

## Shader conventions

- `common.wgsl` defines all shared structs (`Particle`, `SimParams`, `AudioParams`, `RenderUniforms`) and utility functions (spatial binning, toroidal wrapping, audio helpers)
- Every compute/render shader includes `common.wgsl` as a prefix — struct definitions must stay in sync between `common.wgsl` and `gpu.rs` (Rust `#[repr(C)]` structs with bytemuck)
- Workgroup sizes: 256 for particle shaders, 64 for GRA shaders

## Controls

The app has an egui parameter panel and mouse controls for pan/zoom. Audio can be toggled at runtime.
