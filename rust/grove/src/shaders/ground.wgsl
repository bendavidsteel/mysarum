// Noise-displaced ground grid. Vertices are generated procedurally from
// @builtin(vertex_index) so no vertex buffer is needed. Ports the spirit of
// shader.vert (height displacement) + node.frag (simplex shading).

@group(0) @binding(0) var<uniform> u: Uniforms;

const CELLS: u32 = 96u;          // grid cells per side

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) grey:  f32,
}

fn ground_height(world_xz: vec2<f32>) -> f32 {
    let res = u.world_dim.xyz;
    let p = vec3<f32>(world_xz.x / res.x, world_xz.y / res.z, u.misc.x * 0.1);
    let n = simplex3d_fractal(p * 8.0 + 8.0);
    return (n * 0.5 + 0.5) * res.y * 0.06;
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // 6 verts per cell (two triangles)
    let cell = vid / 6u;
    let corner = vid % 6u;
    let cx = cell % CELLS;
    let cz = cell / CELLS;

    // corner offsets for the two triangles of a quad
    var off = array<vec2<u32>, 6>(
        vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(1u, 1u),
        vec2<u32>(0u, 0u), vec2<u32>(1u, 1u), vec2<u32>(0u, 1u),
    );
    let o = off[corner];
    let gx = f32(cx + o.x) / f32(CELLS);
    let gz = f32(cz + o.y) / f32(CELLS);

    let res = u.world_dim.xyz;
    let wx = gx * res.x;
    let wz = gz * res.z;
    let wy = ground_height(vec2<f32>(wx, wz));

    let world = vec3<f32>(wx, wy, wz);

    var out: VsOut;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.grey = 0.0;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let res = u.world_dim.xyz;
    var p = in.world / res;
    p.z += u.misc.x * 0.1;
    let n = simplex3d_fractal(p * 8.0 + 8.0);
    let g = n * 0.25 + 0.30;          // dim ground
    return vec4<f32>(vec3<f32>(g) * vec3<f32>(0.5, 0.7, 0.7), 1.0);
}
