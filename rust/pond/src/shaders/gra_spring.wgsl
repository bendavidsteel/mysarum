// ── GRA spring forces between connected nodes ───────────────────────────────
// Concatenated after common.wgsl — all types available.

@group(0) @binding(0) var<storage, read> gra_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> gra_force: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> adj_offset: array<u32>;
@group(0) @binding(3) var<storage, read> adj_list: array<u32>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.num_gra_nodes {
        return;
    }

    let pos = gra_pos[i];
    if pos.w < 0.0 {
        return;
    }

    let start = adj_offset[i];
    let end = adj_offset[i + 1u];

    var spring_force = vec2<f32>(0.0);

    for (var e = start; e < end; e++) {
        let j = adj_list[e];
        let other = gra_pos[j];

        var diff = other.xy - pos.xy;
        diff.x = wrap_delta_1d(diff.x, params.world_half);
        diff.y = wrap_delta_1d(diff.y, params.world_half);

        let d = length(diff);
        if d > EPSILON {
            let f = (d - params.gra_spring_length) * params.gra_spring_stiffness / d;
            spring_force += diff * f;
        }
    }

    let existing = gra_force[i];
    gra_force[i] = vec4(existing.xy + spring_force, 0.0, 0.0);
}
