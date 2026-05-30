// Physarum 3D agents — port of compute_agents.glsl.
// Reads the trail volume (sampled texture), senses 5 directions, steers, moves
// and deposits into the write target volume.

@group(0) @binding(0) var<storage, read>       a_in:  array<Agent>;
@group(0) @binding(1) var<storage, read_write> a_out: array<Agent>;
@group(0) @binding(2) var trail_read:  texture_3d<f32>;
@group(0) @binding(3) var trail_write: texture_storage_3d<rgba8unorm, write>;
@group(1) @binding(0) var<uniform> cp: ComputeParams;

fn load_trail(coord: vec3<i32>) -> vec4<f32> {
    let res = vec3<i32>(cp.vol_res.xyz);
    let c = clamp(coord, vec3<i32>(0), res - vec3<i32>(1));
    return textureLoad(trail_read, c, 0);
}

fn sense_dir(vel_norm: vec3<f32>, oa: vec3<f32>, ob: vec3<f32>,
             offset: f32, oa_off: f32, ob_off: f32) -> vec3<f32> {
    var d = vel_norm * offset;
    if (oa_off > 0.0) { d += oa * offset * tan(oa_off); }
    else if (oa_off < 0.0) { d -= oa * offset * tan(-oa_off); }
    if (ob_off > 0.0) { d += ob * offset * tan(ob_off); }
    else if (ob_off < 0.0) { d -= ob * offset * tan(-ob_off); }
    return d;
}

fn sense(pos: vec3<f32>, sensor_dir: vec3<f32>, mask: vec4<f32>) -> f32 {
    let sp = pos + sensor_dir;
    let sense_weight = (mask * 2.0) - 1.0;
    let t = load_trail(vec3<i32>(sp));
    return dot(sense_weight, t);
}

fn ensure_rebound(pos: ptr<function, vec3<f32>>, vel: ptr<function, vec3<f32>>) {
    let res = cp.vol_res.xyz;
    var normal = vec3<f32>(0.0);
    var rebound = false;
    if ((*pos).x < 0.0)     { normal = vec3<f32>( 1.0, 0.0, 0.0); rebound = true; }
    if ((*pos).x >= res.x)  { normal = vec3<f32>(-1.0, 0.0, 0.0); rebound = true; }
    if ((*pos).y < 0.0)     { normal = vec3<f32>(0.0,  1.0, 0.0); rebound = true; }
    if ((*pos).y >= res.y)  { normal = vec3<f32>(0.0, -1.0, 0.0); rebound = true; }
    if ((*pos).z < 0.0)     { normal = vec3<f32>(0.0, 0.0,  1.0); rebound = true; }
    if ((*pos).z >= res.z)  { normal = vec3<f32>(0.0, 0.0, -1.0); rebound = true; }
    if (rebound) {
        *vel = *vel - (normal * 2.0 * dot(*vel, normal));
        *pos = clamp(*pos, vec3<f32>(0.0), res);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&a_in)) { return; }

    let pos = a_in[i].pos.xyz;
    let vel = a_in[i].vel.xyz;
    // single species → mask = (1,0,0,0)
    let mask = vec4<f32>(1.0, 0.0, 0.0, 0.0);

    let time   = cp.timing.x;
    let dt     = cp.timing.y;
    let sensor_angle  = cp.phys.y;
    let sensor_offset = cp.phys.z;
    let move_speed    = cp.phys.w;
    let turn_speed    = cp.phys2.x;
    let trail_weight  = cp.phys.x;

    let seed = vec4<f32>(pos.x, pos.y, pos.z, time);
    let rand_vec = (2.0 * vec3<f32>(
        random4(seed.xyzw), random4(seed.yzwx), random4(seed.zwxy))) - 1.0;

    let ortho_a = normalize(cross(vel, rand_vec));
    let ortho_b = normalize(cross(vel, ortho_a));
    let vel_norm = normalize(vel);

    let ahead = sense_dir(vel_norm, ortho_a, ortho_b, sensor_offset, 0.0, 0.0);
    let left  = sense_dir(vel_norm, ortho_a, ortho_b, sensor_offset, sensor_angle, 0.0);
    let right = sense_dir(vel_norm, ortho_a, ortho_b, sensor_offset, -sensor_angle, 0.0);
    let up    = sense_dir(vel_norm, ortho_a, ortho_b, sensor_offset, 0.0, sensor_angle);
    let down  = sense_dir(vel_norm, ortho_a, ortho_b, sensor_offset, 0.0, -sensor_angle);

    let w_ahead = sense(pos, ahead, mask);
    let w_left  = sense(pos, left,  mask);
    let w_right = sense(pos, right, mask);
    let w_up    = sense(pos, up,    mask);
    let w_down  = sense(pos, down,  mask);

    let rand_perp = normalize(cross(vel_norm, rand_vec));
    var rand_angle = random4(seed) * turn_speed;
    var new_vel = vel;

    if (w_left > w_ahead && w_left > w_right && w_left > w_up && w_left > w_down) {
        new_vel = turn_toward(vel_norm, normalize(left), rand_angle);
    } else if (w_right > w_ahead && w_right > w_left && w_right > w_up && w_right > w_down) {
        new_vel = turn_toward(vel_norm, normalize(right), rand_angle);
    } else if (w_up > w_ahead && w_up > w_left && w_up > w_right && w_up > w_down) {
        new_vel = turn_toward(vel_norm, normalize(up), rand_angle);
    } else if (w_down > w_ahead && w_down > w_left && w_down > w_right && w_down > w_up) {
        new_vel = turn_toward(vel_norm, normalize(down), rand_angle);
    } else if (w_ahead > w_left && w_ahead > w_right && w_ahead > w_up && w_ahead > w_down) {
        rand_angle /= 4.0;
        new_vel = turn_toward(vel_norm, rand_perp, rand_angle);
    } else if (w_ahead < w_left && w_ahead < w_right && w_ahead < w_up && w_ahead < w_down) {
        new_vel = turn_toward(vel_norm, rand_perp, rand_angle);
    } else {
        rand_angle /= 2.0;
        new_vel = turn_toward(vel_norm, rand_perp, rand_angle);
    }

    let max_sense = max(max(max(max(w_ahead, w_left), w_right), w_up), w_down);
    let avg_sense = (w_ahead + w_left + w_right + w_up + w_down) / 5.0;
    let turn_amount = rand_angle / max(turn_speed, 1e-5);

    let res = cp.vol_res.xyz;
    var force = vec3<f32>(0.0);
    force += vec3<f32>(0.0, 0.1, 0.0) * ((res.y - pos.y) / res.y);     // gravity-ish
    force += vec3<f32>(cp.wind.x, 0.0, cp.wind.y) * cp.timing.z;        // wind

    new_vel = normalize(new_vel + (force * dt));
    var new_pos = pos + (new_vel * dt * move_speed);

    ensure_rebound(&new_pos, &new_vel);

    a_out[i].pos = vec4<f32>(new_pos, 1.0);
    a_out[i].vel = vec4<f32>(new_vel, 1.0);
    a_out[i].attr = a_in[i].attr;
    a_out[i].state = vec4<f32>(max_sense, avg_sense, turn_amount, 0.0);

    let coord = vec3<i32>(new_pos);
    let old_trail = load_trail(coord);
    let new_trail = clamp(old_trail + (mask * trail_weight * dt), vec4<f32>(0.0), vec4<f32>(1.0));
    let c = clamp(coord, vec3<i32>(0), vec3<i32>(res) - vec3<i32>(1));
    textureStore(trail_write, c, new_trail);
}
