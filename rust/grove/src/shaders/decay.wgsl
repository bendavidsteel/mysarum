// Multiplicative decay of the trail volume — port of compute_decay.glsl.

@group(0) @binding(0) var trail_read:  texture_3d<f32>;
@group(0) @binding(1) var trail_write: texture_storage_3d<rgba8unorm, write>;
@group(1) @binding(0) var<uniform> cp: ComputeParams;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec3<u32>(cp.vol_res.xyz);
    if (gid.x >= res.x || gid.y >= res.y || gid.z >= res.z) { return; }
    let coord = vec3<i32>(gid);

    let trail = textureLoad(trail_read, coord, 0);
    let out = clamp(trail * cp.phys2.z * cp.timing.y, vec4<f32>(0.0), vec4<f32>(1.0));
    textureStore(trail_write, coord, out);
}
