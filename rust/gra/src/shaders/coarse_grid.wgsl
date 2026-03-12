// Coarse grid accumulation for far-field repulsion.
// Builds a 16x16 summary grid with center-of-mass and count per cell.
// Uses fixed-point i32 atomics for position accumulation.

@group(0) @binding(0) var<storage, read_write> coarse_sum_x: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> coarse_sum_y: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> coarse_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> node_pos: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

const COARSE_DIM: u32 = 16u;
const COARSE_CELLS: u32 = 256u;
const FIXED_SCALE: f32 = 32.0;

@compute @workgroup_size(64)
fn coarse_clear(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= COARSE_CELLS {
        return;
    }
    atomicStore(&coarse_sum_x[id.x], 0);
    atomicStore(&coarse_sum_y[id.x], 0);
    atomicStore(&coarse_count[id.x], 0u);
}

@compute @workgroup_size(64)
fn coarse_accumulate(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_nodes {
        return;
    }
    let pos = node_pos[id.x];
    if pos.w < 0.0 { return; }

    let cx = u32(clamp((pos.x - params.origin_x) / params.coarse_bin_size, 0.0, f32(COARSE_DIM - 1u)));
    let cy = u32(clamp((pos.y - params.origin_y) / params.coarse_bin_size, 0.0, f32(COARSE_DIM - 1u)));
    let cell = cy * COARSE_DIM + cx;

    atomicAdd(&coarse_sum_x[cell], i32(pos.x * FIXED_SCALE));
    atomicAdd(&coarse_sum_y[cell], i32(pos.y * FIXED_SCALE));
    atomicAdd(&coarse_count[cell], 1u);
}
