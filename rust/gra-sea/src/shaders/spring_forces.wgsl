@group(0) @binding(0) var<storage, read> node_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> node_force: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> adj_offset: array<u32>;
@group(0) @binding(3) var<storage, read> adj_list: array<u32>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_nodes { return; }
    let pos = node_pos[id.x];
    if pos.w < 0.0 { return; }

    let start = adj_offset[id.x];
    let end = adj_offset[id.x + 1u];

    var spring_force = vec2f(0.0);

    for (var k = start; k < end; k += 1u) {
        let j = adj_list[k];
        let other = node_pos[j];

        // Spring force (minimum-image for toroidal wrapping)
        var diff = other.xy - pos.xy;
        let size = params.world_half * 2.0;
        if diff.x > params.world_half { diff.x -= size; }
        if diff.x < -params.world_half { diff.x += size; }
        if diff.y > params.world_half { diff.y -= size; }
        if diff.y < -params.world_half { diff.y += size; }
        let d = length(diff);
        if d > EPSILON {
            let f = (d - params.spring_length) * params.spring_stiffness / d;
            spring_force += diff * f;
        }
    }

    // Add spring force to existing repulsion force
    let existing = node_force[id.x];
    node_force[id.x] = vec4f(existing.xy + spring_force, 0.0, 0.0);
}
