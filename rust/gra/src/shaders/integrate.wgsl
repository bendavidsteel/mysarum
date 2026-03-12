@group(0) @binding(0) var<storage, read_write> node_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> node_vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> node_force: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_nodes { return; }
    var p = node_pos[id.x];
    if p.w < 0.0 { return; }

    var vel = node_vel[id.x].xy;
    let force = node_force[id.x].xy;

    // Add force to velocity
    vel += force;

    // Clamp velocity magnitude
    let speed = length(vel);
    if speed > params.max_velocity {
        vel = vel * (params.max_velocity / speed);
    }

    // Update position
    p = vec4f(p.xy + vel, 0.0, p.w);
    node_pos[id.x] = p;

    // Apply damping
    vel *= (1.0 - params.damping);
    node_vel[id.x] = vec4f(vel, 0.0, 0.0);
}
