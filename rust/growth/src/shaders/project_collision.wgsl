// XPBD collision constraint projection (Jacobi style).
// Uses spatial hash built at frame start to find nearby vertices.
// Pushes apart non-adjacent vertices closer than repulsion_distance.

@group(0) @binding(0) var<storage, read_write> vertex_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sorted_idx: array<u32>;
@group(0) @binding(2) var<storage, read> bin_offset: array<u32>;
// Per-vertex 1-ring neighbour list, NEIGHBORS_K slots each, -1 terminated.
// Precomputed on the CPU on topology frames (see upload_mesh_to_gpu).
@group(0) @binding(3) var<storage, read> vertex_neighbors: array<i32>;
@group(1) @binding(0) var<uniform> params: SimParams;

// MUST match VERTEX_NEIGHBORS_MAX in gpu.rs.
const NEIGHBORS_K: u32 = 20u;

// Check if vertex j is a mesh neighbor of vertex i (connected by an edge).
// Skip collision for direct mesh neighbors — springs handle those. A fixed
// contiguous scan of the precomputed list replaces the old half-edge fan walk:
// no global-memory pointer chase, no warp divergence. The list is at most
// valence-long (it ends at the -1 sentinel), so the common non-neighbor case
// — distant folds touching during buckling — pays only ~valence cheap reads.
fn is_mesh_neighbor(i: u32, j: u32) -> bool {
    let base = i * NEIGHBORS_K;
    for (var s = 0u; s < NEIGHBORS_K; s += 1u) {
        let nb = vertex_neighbors[base + s];
        if nb < 0 { return false; }
        if nb == i32(j) { return true; }
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }
    let pos = vertex_pos[id.x];
    // Skip inactive (w < 0) and pinned (w == 0) vertices; pinned vertices
    // still repel free vertices (they remain in the spatial bins below).
    if pos.w < 0.5 { return; }

    let info = get_bin_info(pos.x, pos.y, pos.z, params);

    var correction = vec3f(0.0);
    var num_collisions = 0u;

    for (var dz: i32 = -1; dz <= 1; dz += 1) {
        let nz = i32(info.bin_z) + dz;
        if nz < 0 || u32(nz) >= params.num_bins_z { continue; }
        for (var dy: i32 = -1; dy <= 1; dy += 1) {
            let ny = i32(info.bin_y) + dy;
            if ny < 0 || u32(ny) >= params.num_bins_y { continue; }
            for (var dx: i32 = -1; dx <= 1; dx += 1) {
                let nx = i32(info.bin_x) + dx;
                if nx < 0 || u32(nx) >= params.num_bins_x { continue; }
                let bi = (u32(nz) * params.num_bins_y + u32(ny)) * params.num_bins_x + u32(nx);
                let start = bin_offset[bi];
                let end = bin_offset[bi + 1u];

                for (var k = start; k < end; k += 1u) {
                    let j = sorted_idx[k];
                    if j == id.x { continue; }

                    let other = vertex_pos[j];
                    if other.w < 0.0 { continue; }

                    let diff = pos.xyz - other.xyz;
                    let dist = max(length(diff), EPSILON);

                    if dist < params.repulsion_distance {
                        // Skip direct mesh neighbors — springs handle those
                        if is_mesh_neighbor(id.x, j) { continue; }

                        // XPBD collision constraint: C = repulsion_distance - dist
                        // Gradient for vertex i: diff / dist (unit direction away from j)
                        // w_i = w_j = 1, so denominator = 2
                        let C = params.repulsion_distance - dist;
                        let dlambda = -C / 2.0;  // no compliance — hard collision
                        var corr = -dlambda * (diff / dist);  // push i away from j

                        // Scale by penetration depth for smooth falloff
                        let ratio = C / params.repulsion_distance;
                        corr *= ratio;

                        let corr_mag = length(corr);
                        let max_corr = params.repulsion_distance * 0.2;
                        if corr_mag > max_corr {
                            corr = corr * (max_corr / corr_mag);
                        }

                        correction += corr;
                        num_collisions += 1u;
                    }
                }
            }
        }
    }

    if num_collisions > 0u {
        // SOR under-relaxation (see SimParams.relaxation). No strength multiplier:
        // the XPBD correction already resolves the constraint exactly, and any
        // gain > 1 injects energy instead of removing violation.
        let step = params.relaxation * correction / f32(num_collisions);
        vertex_pos[id.x] = vec4f(apply_floor(pos.xyz + step, params), pos.w);
    }
}
