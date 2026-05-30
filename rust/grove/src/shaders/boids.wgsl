// 3D boids flocking — port of compute_boids.glsl.
// O(N^2) neighbour loop over all agents.

@group(0) @binding(0) var<storage, read>       p_in:  array<Particle>;
@group(0) @binding(1) var<storage, read_write> p_out: array<Particle>;
@group(1) @binding(0) var<uniform> cp: ComputeParams;

fn avoid_walls(pos: vec3<f32>, vel: ptr<function, vec3<f32>>) {
    let res = cp.world_res.xyz;
    if (pos.x < 0.0)     { (*vel).x += 1.0; } else if (pos.x > res.x) { (*vel).x -= 1.0; }
    if (pos.y < 0.0)     { (*vel).y += 1.0; } else if (pos.y > res.y) { (*vel).y -= 1.0; }
    if (pos.z < 0.0)     { (*vel).z += 1.0; } else if (pos.z > res.z) { (*vel).z -= 1.0; }
}

fn ang(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return dot(a, b) / (length(a) * length(b));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(cp.world_res.w);
    if (i >= n) { return; }

    let pos   = p_in[i].pos.xyz;
    let vel   = p_in[i].vel.xyz;
    let nat_freq = p_in[i].attr.x;
    var phase    = p_in[i].attr.y;

    let time     = cp.timing.x;
    let dt       = cp.timing.y;
    let attraction = cp.boid_a.x;  let attraction_max = cp.boid_a.y;
    let alignment  = cp.boid_a.z;  let alignment_max  = cp.boid_a.w;
    let repulsion  = cp.boid_b.x;  let repulsion_max  = cp.boid_b.y;
    let max_speed  = cp.boid_b.z;  let random_strength = cp.boid_b.w;
    let fov        = cp.boid_c.x;
    let kuramoto_str = cp.boid_c.y; let kuramoto_max = cp.boid_c.z;

    let vel_norm = normalize(vel);

    let seed = vec4<f32>(pos.x, pos.y, pos.z, time);
    var rand_vec = vec3<f32>(
        random4(seed.xyzw), random4(seed.yzwx), random4(seed.zwxy));
    rand_vec = rand_vec * 2.0 - 1.0;

    let rand_perp = normalize(cross(vel_norm, rand_vec));
    let rand_angle = random4(seed.xyzw) * random_strength;
    var new_vel = turn_toward(vel, rand_perp, rand_angle);

    var phase_sum = 0.0;
    var kuramoto_count = 0;

    var repulse_sum = vec3<f32>(0.0); var repulse_count = 0;
    var align_sum   = vec3<f32>(0.0); var align_count   = 0;
    var attract_sum = vec3<f32>(0.0); var attract_count = 0;

    for (var j = 0u; j < n; j = j + 1u) {
        if (j == i) { continue; }
        let their_pos = p_in[j].pos.xyz;
        if (ang(vel, their_pos - pos) < fov) { continue; }
        let their_vel = p_in[j].vel.xyz;
        let rel = their_pos - pos;
        let sqd = dot(rel, rel);

        if (sqd < repulsion_max * repulsion_max) { repulse_sum += rel; repulse_count += 1; }
        if (sqd < alignment_max * alignment_max) { align_sum += their_vel; align_count += 1; }
        if (sqd < attraction_max * attraction_max) { attract_sum += rel; attract_count += 1; }
        if (sqd < kuramoto_max * kuramoto_max) {
            let their_phase = p_in[j].attr.y;
            phase_sum += sin(their_phase - phase);
            kuramoto_count += 1;
        }
    }

    if (repulse_count > 0) {
        let d = repulse_sum / f32(repulse_count);
        new_vel = turn_toward(new_vel, -d, repulsion);
    }
    if (align_count > 0) {
        let d = align_sum / f32(align_count);
        new_vel = turn_toward(new_vel, d, alignment);
    }
    if (attract_count > 0) {
        let d = attract_sum / f32(attract_count);
        new_vel = turn_toward(new_vel, d, attraction);
    }
    if (kuramoto_count > 0) {
        phase += (nat_freq + (kuramoto_str * phase_sum / f32(kuramoto_count))) * dt;
        phase = phase % TAU;
    }

    new_vel = normalize(new_vel) * max_speed;
    avoid_walls(pos, &new_vel);

    // wind on xz
    new_vel.x += cp.wind.x * cp.timing.z;
    new_vel.z += cp.wind.y * cp.timing.z;

    let new_pos = pos + new_vel * dt;

    p_out[i].pos = vec4<f32>(new_pos, 1.0);
    p_out[i].vel = vec4<f32>(new_vel, 1.0);
    p_out[i].attr = vec4<f32>(nat_freq, phase, 0.0, 0.0);

    let amp = sin((time * nat_freq) + phase);
    let brightness = cp.timing.w;
    p_out[i].color = vec4<f32>(1.0, 1.0, 1.0, (0.66 + 0.33 * amp) * brightness);
}
