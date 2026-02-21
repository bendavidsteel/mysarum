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
- Custom WGSL fragment shader at `assets/lit_mesh.wgsl` using `#[shader_model]` macro

## Architecture

Single-file app (`src/main.rs`, ~1480 lines) with Nannou's `app(model).update(update).run()` pattern.

### Half-Edge Mesh (structure-of-arrays)
- `HalfEdgeMesh` — fixed-size arrays (MAX_VERTICES=2000, MAX_HALF_EDGES=12000, MAX_FACES=4000) with `-1` sentinel for inactive/null indices
- Uses `i32` indices throughout (not `Option<usize>`) — check `>= 0` before use
- Allocation via watermark counters (`next_vertex`, `next_face`, `next_half_edge`)
- Three growth operations: `add_external_triangle` (boundary), `add_internal_edge_triangle` (one side boundary), `add_internal_triangles` (fully internal edge split)
- `flip_edge` and `refine_mesh` for Delaunay-style valence optimization (target valence 6)

### Simulation Loop (`update()`)
1. **Physics** — spring forces, pairwise repulsion (O(n²)), bulge force (boundary outward push), planar force (Laplacian smoothing)
2. **State evolution** — neighbor-averaging message pass → Gaussian growth function on vertex states
3. **Mesh growth** (every 10 frames) — vertices with high growth potential split adjacent edges
4. **Mesh refinement** (every 20 frames) — edge flips to regularize valences

### Rendering (`view()`)
- Auto-fits mesh to window via bounding box
- Encodes barycentric coords in vertex color RGB, vertex state in alpha
- Fragment shader (`lit_mesh.wgsl`) does: flat-shaded Lambert lighting via `dpdx`/`dpdy` normals, HSV state→color mapping, barycentric wireframe overlay

## Controls

- **R** — Randomize parameters and reset mesh
- **T** — Reset mesh (keep current parameters)
- **S** — Save screenshot
- Debug mesh validation runs every 100 frames in debug builds (`#[cfg(debug_assertions)]`)
