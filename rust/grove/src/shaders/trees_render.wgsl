// Instanced camera-facing ribbon per branch segment (replaces the OF
// geometry-shader cylinder). Simplex shading from node.frag.

@group(0) @binding(0) var<storage, read> segments: array<Segment>;
@group(0) @binding(1) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0) world: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32,
           @builtin(instance_index) inst: u32) -> VsOut {
    let s = segments[inst];
    let p0 = s.p0.xyz;
    let p1 = s.p1.xyz;
    let w0 = max(s.p0.w, 0.3);
    let w1 = max(s.p1.w, 0.3);

    let mid = (p0 + p1) * 0.5;
    let seg_dir = normalize(p1 - p0 + vec3<f32>(1e-5, 0.0, 0.0));
    let view_dir = normalize(u.cam_pos.xyz - mid);
    var side = cross(seg_dir, view_dir);
    if (length(side) < 1e-4) { side = vec3<f32>(1.0, 0.0, 0.0); }
    side = normalize(side);

    // 6 verts → quad (p0+side, p0-side, p1+side) (p1+side, p0-side, p1-side)
    var idx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let k = idx[vid];
    var world: vec3<f32>;
    if (k == 0u)      { world = p0 + side * w0; }
    else if (k == 1u) { world = p0 - side * w0; }
    else if (k == 2u) { world = p1 + side * w1; }
    else              { world = p1 - side * w1; }

    var out: VsOut;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var p = in.world / u.world_dim.xyz;
    p.z += u.misc.x * 0.1;
    let n = simplex3d_fractal(p * 8.0 + 8.0);
    let grey = n * 0.25 + 0.75;
    return vec4<f32>(vec3<f32>(grey), 1.0);
}
