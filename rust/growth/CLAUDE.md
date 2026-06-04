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
- `HalfEdgeMesh` — fixed-size arrays (MAX_VERTICES=32000, MAX_HALF_EDGES=192000, MAX_FACES=64000) with `-1` sentinel for inactive/null indices
- Uses `i32` indices throughout (not `Option<usize>`) — check `>= 0` before use
- Allocation via watermark counters (`next_vertex`, `next_face`, `next_half_edge`)
- Three growth operations: `add_external_triangle` (boundary), `add_internal_edge_triangle` (one side boundary), `add_internal_triangles` (fully internal edge split)
- **Intrinsic triangulations**: per-half-edge `half_edge_intrinsic_len` stores intrinsic edge lengths, decoupled from 3D embedding. Growth modifies intrinsic lengths; XPBD springs use them as rest lengths.
- `flip_edge` computes new intrinsic edge length via planar unfold; `refine_mesh` uses intrinsic Delaunay criterion (opposite angle sum > π)
- Split operations compute new intrinsic lengths via the median formula

### Simulation Loop (`update()`)
1. **Physics (XPBD)** — pairwise repulsion + bulge forces applied as overdamped position prediction, then XPBD constraint projection (Jacobi, N iterations): spring constraints enforce intrinsic edge lengths, dihedral angle bending constraints resist folding. No velocity — first-order overdamped system.
2. **State evolution** — Chebyshev polynomial convolution (Lenia-style ring kernel) → Gaussian growth function updates vertex states
3. **Differential growth** — intrinsic edge lengths grow continuously based on vertex state: each vertex walks its half-edge fan and grows outgoing edges proportionally to `max(2s-1, 0)^2`. The spring shader averages he and twin intrinsic lengths for each edge's rest length.
4. **Adaptive subdivision** (every 10 frames) — edges exceeding `max_edge_len` (intrinsic) are split (longest-in-face priority)
5. **Mesh refinement** (every 20 frames) — intrinsic Delaunay edge flips

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
