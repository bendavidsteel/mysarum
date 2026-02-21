# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build          # Build debug
cargo build --release # Build release
cargo run            # Run the application
cargo test           # Run tests
```

## Project Overview

A growing mesh simulation using **Nannou** (bevy-refactor branch) + **Bevy 0.17**. A half-edge mesh starts as a small disc and grows by splitting edges, with vertex states evolving via Lenia-style Gaussian growth functions and message passing along mesh edges.

## Dependencies

- `nannou` / `bevy_nannou` from git (`bevy-refactor` branch) — requires Bevy 0.17, NOT 0.18
- Custom WGSL render shader at `src/shaders/mesh_render.wgsl` (vertex pulling + fragment lighting)

## Architecture

Single-file app (`src/main.rs`) with Nannou's `app(model).update(update).render(render).run()` pattern.

### Half-Edge Mesh (structure-of-arrays)
- `HalfEdgeMesh` — fixed-size arrays (MAX_VERTICES=8000, MAX_HALF_EDGES=48000, MAX_FACES=16000) with `-1` sentinel for inactive/null indices
- Uses `i32` indices throughout (not `Option<usize>`) — check `>= 0` before use
- Allocation via watermark counters (`next_vertex`, `next_face`, `next_half_edge`)
- Three growth operations: `add_external_triangle` (boundary), `add_internal_edge_triangle` (one side boundary), `add_internal_triangles` (fully internal edge split)
- `flip_edge` and `refine_mesh` for Delaunay-style valence optimization (target valence 6)

### Simulation Loop (`update()`)
1. **Physics** — spring forces, pairwise repulsion (O(n²)), bulge force (boundary outward push), planar force (Laplacian smoothing)
2. **State evolution** — neighbor-averaging message pass → Gaussian growth function on vertex states
3. **Mesh growth** (every 10 frames) — vertices with high growth potential split adjacent edges
4. **Mesh refinement** (every 20 frames) — edge flips to regularize valences

### Rendering (`render()` — GPU-direct)
- Custom wgpu render pipeline bypasses Nannou's draw API — no per-frame GPU→CPU readback
- Vertex shader pulls positions/states directly from GPU compute storage buffers via vertex index
- Index buffer rebuilt only on topology-change frames (every 10/20); bounding box cached
- Fragment shader (`mesh_render.wgsl`) does: flat-shaded Lambert lighting via `dpdx`/`dpdy` normals, HSV state→color mapping, barycentric wireframe overlay
- `RenderState` (pipeline, bind group) initialized lazily on first render frame via `Arc<Mutex<...>>`
- Model wrapped in `Arc<HalfEdgeMesh>` for efficient clone to Bevy render world

## Controls

- **R** — Randomize parameters and reset mesh
- **T** — Reset mesh (keep current parameters)
- **S** — Save screenshot
- Debug mesh validation runs every 100 frames in debug builds (`#[cfg(debug_assertions)]`)
