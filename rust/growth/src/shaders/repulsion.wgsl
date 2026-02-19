@group(0) @binding(0) var<storage, read> vertex_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vertex_force: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sorted_idx: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offset: array<u32>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices {
        return;
    }
    let pos = vertex_pos[id.x];
    if pos.w < 0.0 {
        vertex_force[id.x] = vec4f(0.0);
        return;
    }

    let info = get_bin_info(pos.x, pos.y, params);
    var force = vec3f(0.0);

    for (var dy: i32 = -1; dy <= 1; dy += 1) {
        for (var dx: i32 = -1; dx <= 1; dx += 1) {
            let nx = i32(info.bin_x) + dx;
            let ny = i32(info.bin_y) + dy;
            if nx < 0 || ny < 0 || u32(nx) >= params.num_bins_x || u32(ny) >= params.num_bins_y {
                continue;
            }
            let bi = u32(ny) * params.num_bins_x + u32(nx);
            let start = bin_offset[bi];
            let end = bin_offset[bi + 1u];

            for (var k = start; k < end; k += 1u) {
                let j = sorted_idx[k];
                if j == id.x { continue; }

                let other = vertex_pos[j];
                let diff = pos.xyz - other.xyz;
                let dist_sq = dot(diff, diff) + EPSILON;
                let dist = sqrt(dist_sq);

                if dist < params.repulsion_distance {
                    let ratio = (params.repulsion_distance - dist) / params.repulsion_distance;
                    force += ratio * ratio * (diff / dist);
                }
            }
        }
    }

    vertex_force[id.x] = vec4f(params.repulsion_strength * force, 0.0);
}
