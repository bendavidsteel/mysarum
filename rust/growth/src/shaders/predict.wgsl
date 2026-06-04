// Overdamped position prediction from accumulated external forces (repulsion + bending + bulge).
// No velocity — first-order integration: p += (dt / damping) * f

@group(0) @binding(0) var<storage, read_write> vertex_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vertex_force: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }
    var p = vertex_pos[id.x];
    if p.w < 0.0 { return; }

    let f = vertex_force[id.x].xyz;
    var displacement = (params.dt / params.damping) * f;

    // Clamp max displacement to prevent overshooting
    let max_disp = params.spring_len * 0.25;
    let disp_len = length(displacement);
    if disp_len > max_disp {
        displacement = displacement * (max_disp / disp_len);
    }

    p = vec4f(p.xyz + displacement, p.w);
    vertex_pos[id.x] = p;
}
