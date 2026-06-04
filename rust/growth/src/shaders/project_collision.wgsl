// XPBD collision constraint projection (Jacobi style).
// Uses spatial hash built at frame start to find nearby vertices.
// Pushes apart non-adjacent vertices closer than repulsion_distance.

@group(0) @binding(0) var<storage, read_write> vertex_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sorted_idx: array<u32>;
@group(0) @binding(2) var<storage, read> bin_offset: array<u32>;
@group(0) @binding(3) var<storage, read> he_packed: array<vec4<i32>>;
@group(0) @binding(4) var<storage, read> vertex_he: array<i32>;
@group(1) @binding(0) var<uniform> params: SimParams;

// Check if vertex j is a mesh neighbor of vertex i (connected by an edge).
// Skip collision for direct mesh neighbors — springs handle those.
fn is_mesh_neighbor(i: u32, j: u32) -> bool {
    let start_he = vertex_he[i];
    if start_he < 0 { return false; }

    let start_dest = he_packed[start_he].x;
    var he = start_he;
    var first = true;

    for (var iter = 0u; iter < 20u; iter += 1u) {
        let data = he_packed[he];
        let dest = data.x;
        let twin = data.y;

        if dest == i32(j) { return true; }

        if twin < 0 { break; }
        let twin_next = he_packed[twin].z;
        if twin_next < 0 { break; }
        he = twin_next;
        if he_packed[he].x == start_dest && !first { break; }
        first = false;
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }
    let pos = vertex_pos[id.x];
    if pos.w < 0.0 { return; }

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
        // SOR under-relaxation (see SimParams.relaxation)
        let step = params.relaxation * params.repulsion_strength * correction / f32(num_collisions);
        vertex_pos[id.x] = vec4f(pos.xyz + step, pos.w);
    }
}
