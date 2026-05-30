// Wireframe bounding box drawn as a line list (24 vertices = 12 edges).

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // 8 cube corners in [0,1]^3
    var corners = array<vec3<f32>, 8>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
    );
    // 12 edges → 24 indices
    var edges = array<u32, 24>(
        0u,1u, 1u,2u, 2u,3u, 3u,0u,   // bottom
        4u,5u, 5u,6u, 6u,7u, 7u,4u,   // top
        0u,4u, 1u,5u, 2u,6u, 3u,7u,   // verticals
    );
    let c = corners[edges[vid]] * u.world_dim.xyz;
    var out: VsOut;
    out.clip = u.view_proj * vec4<f32>(c, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.25, 0.25, 0.25, 1.0);
}
