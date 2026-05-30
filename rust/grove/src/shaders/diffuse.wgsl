// 3D box blur of the trail volume (replaces the OF separable 3-axis blur with
// a single 6-neighbour pass — visually equivalent, one pass instead of three).

@group(0) @binding(0) var trail_read:  texture_3d<f32>;
@group(0) @binding(1) var trail_write: texture_storage_3d<rgba8unorm, write>;
@group(1) @binding(0) var<uniform> cp: ComputeParams;

fn load_c(coord: vec3<i32>) -> vec4<f32> {
    let res = vec3<i32>(cp.vol_res.xyz);
    return textureLoad(trail_read, clamp(coord, vec3<i32>(0), res - vec3<i32>(1)), 0);
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec3<u32>(cp.vol_res.xyz);
    if (gid.x >= res.x || gid.y >= res.y || gid.z >= res.z) { return; }
    let coord = vec3<i32>(gid);

    let original = load_c(coord);
    var sum = original;
    sum += load_c(coord + vec3<i32>(1, 0, 0));
    sum += load_c(coord - vec3<i32>(1, 0, 0));
    sum += load_c(coord + vec3<i32>(0, 1, 0));
    sum += load_c(coord - vec3<i32>(0, 1, 0));
    sum += load_c(coord + vec3<i32>(0, 0, 1));
    sum += load_c(coord - vec3<i32>(0, 0, 1));
    let blurred = sum / 7.0;

    let w = clamp(cp.phys2.y * cp.timing.y, 0.0, 1.0);   // diffuseRate * dt
    let out = (original * (1.0 - w)) + (blurred * w);
    textureStore(trail_write, coord, out);
}
