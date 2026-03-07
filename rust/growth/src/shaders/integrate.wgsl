@group(0) @binding(0) var<storage, read_write> vertex_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vertex_force: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }
    var p = vertex_pos[id.x];
    if p.w < 0.0 { return; }
    let f = vertex_force[id.x].xyz;
    var displacement = params.dt * f;
    // Clamp max displacement to half a spring length to prevent overshooting
    let max_step = params.spring_len * 0.5;
    let disp_len = length(displacement);
    if disp_len > max_step {
        displacement = displacement * (max_step / disp_len);
    }
    p = vec4f(p.xyz + params.damping * displacement, p.w);
    vertex_pos[id.x] = p;
}
