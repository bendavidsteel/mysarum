// Far-field repulsion from coarse grid summary.
// Each node repels from distant coarse cells (skipping own cell).
// Uses 1/r force law for smooth long-range pressure.

@group(0) @binding(0) var<storage, read_write> coarse_sum_x: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> coarse_sum_y: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> coarse_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> node_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> node_force: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

const COARSE_DIM: u32 = 16u;
const FIXED_SCALE: f32 = 32.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_nodes {
        return;
    }
    let pos = node_pos[id.x];
    if pos.w < 0.0 { return; }

    // Which coarse cell is this node in?
    let my_cx = i32(clamp((pos.x - params.origin_x) / params.coarse_bin_size, 0.0, f32(COARSE_DIM - 1u)));
    let my_cy = i32(clamp((pos.y - params.origin_y) / params.coarse_bin_size, 0.0, f32(COARSE_DIM - 1u)));

    var force = vec2f(0.0);

    for (var cy = 0u; cy < COARSE_DIM; cy += 1u) {
        for (var cx = 0u; cx < COARSE_DIM; cx += 1u) {
            // Skip own cell (near-field handles it)
            if i32(cx) == my_cx && i32(cy) == my_cy {
                continue;
            }

            let cell = cy * COARSE_DIM + cx;
            let count = atomicLoad(&coarse_count[cell]);
            if count == 0u { continue; }

            // Reconstruct center of mass from fixed-point sums
            let com_x = f32(atomicLoad(&coarse_sum_x[cell])) / (FIXED_SCALE * f32(count));
            let com_y = f32(atomicLoad(&coarse_sum_y[cell])) / (FIXED_SCALE * f32(count));

            let diff = pos.xy - vec2f(com_x, com_y);
            let dist_sq = dot(diff, diff) + EPSILON;
            let dist = sqrt(dist_sq);

            // 1/r force scaled by cell population
            let f = params.far_repulsion_strength * f32(count) / dist;
            force += (diff / dist) * f;
        }
    }

    // Add to existing force (from near-field repulsion)
    node_force[id.x] = node_force[id.x] + vec4f(force, 0.0, 0.0);
}
