// XPBD spring constraint projection (Jacobi style).
// Each vertex walks its half-edge fan, computes spring corrections from all neighbors,
// and applies the averaged correction. Dispatch N times for convergence.

@group(0) @binding(0) var<storage, read_write> vertex_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> he_packed: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> vertex_he: array<i32>;
@group(0) @binding(3) var<storage, read> he_intrinsic_len: array<f32>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }
    let pos = vertex_pos[id.x];
    if pos.w < 0.0 { return; }

    let start_he = vertex_he[id.x];
    if start_he < 0 { return; }

    // Compliance is pre-scaled (α̃ in XPBD notation) so behavior is dt-invariant.
    // 0 == hard constraint; increase for softer springs.
    let alpha_tilde = params.compliance;

    var correction = vec3f(0.0);
    var num_edges = 0u;

    let start_data = he_packed[start_he];
    let start_dest = start_data.x;
    var he = start_he;
    var first = true;

    for (var iter = 0u; iter < 20u; iter += 1u) {
        let data = he_packed[he];
        let dest = data.x;
        let twin = data.y;

        if dest >= 0 {
            let np = vertex_pos[dest].xyz;
            let d = pos.xyz - np;
            // Floor the distance at a small fraction of spring_len so d/dist
            // stays well-defined when two vertices coincide (e.g. just after a split).
            let dist_floor = params.spring_len * 0.01;
            let dist = max(length(d), dist_floor);
            // Rest length = average of both half-edges' intrinsic lengths
            // (each side grown independently by its source vertex). Floor to the
            // same value so a newly-allocated half-edge (rest == 0) can still heal.
            var rest = he_intrinsic_len[he];
            if twin >= 0 {
                rest = (rest + he_intrinsic_len[twin]) * 0.5;
            }
            rest = max(rest, dist_floor);
            let C = dist - rest;

            // XPBD correction: w_i = w_j = 1, so denominator = 2 + alpha_tilde
            let dlambda = -C / (2.0 + alpha_tilde);
            var edge_corr = dlambda * (d / dist);
            // Clamp per-edge correction to prevent explosion after splits
            let corr_mag = length(edge_corr);
            let max_corr = rest * 0.2;
            if corr_mag > max_corr {
                edge_corr = edge_corr * (max_corr / corr_mag);
            }
            correction += edge_corr;
            num_edges += 1u;
        }

        // Walk fan: he -> twin -> next
        if twin < 0 { break; }
        let twin_next = he_packed[twin].z;
        if twin_next < 0 { break; }
        he = twin_next;
        if he_packed[he].x == start_dest && !first { break; }
        first = false;
    }

    // Jacobi relaxation: divide by edge count to prevent overcorrection,
    // then apply SOR under-relaxation so chained projection passes don't overshoot.
    if num_edges > 0u {
        let step = params.relaxation * correction / f32(num_edges);
        vertex_pos[id.x] = vec4f(pos.xyz + step, pos.w);
    }
}
