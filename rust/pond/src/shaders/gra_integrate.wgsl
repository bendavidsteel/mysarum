// ── GRA node velocity & position integration ────────────────────────────────
// Concatenated after common.wgsl — all types available.

@group(0) @binding(0) var<storage, read_write> gra_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> gra_vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gra_force: array<vec4<f32>>;
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

    let force = gra_force[i];
    var vel = gra_vel[i].xy;

    // Damping
    vel *= (1.0 - params.gra_damping);

    // Apply force
    vel += force.xy;

    // Clamp velocity magnitude
    let speed = length(vel);
    if speed > params.gra_max_velocity {
        vel = vel * (params.gra_max_velocity / speed);
    }

    // Integrate position
    let new_pos = wrap_pos(pos.xy + vel, params.world_half);

    gra_pos[i] = vec4(new_pos, pos.z, pos.w);
    gra_vel[i] = vec4(vel, 0.0, 0.0);
}
