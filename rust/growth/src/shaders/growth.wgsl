// Apply Gaussian growth function: state += dt * gaussian(result, mu, sigma)

@group(0) @binding(0) var<storage, read_write> vertex_state: array<f32>;
@group(0) @binding(1) var<storage, read_write> vertex_u: array<f32>;
@group(0) @binding(2) var<storage, read> result: array<f32>;
@group(0) @binding(3) var<storage, read> vertex_pos: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }

    let pos = vertex_pos[id.x];
    if pos.w < 0.0 { return; }

    let r = result[id.x];
    vertex_u[id.x] = r;

    let x = r - params.growth_mu;
    let sigma = params.growth_sigma + EPSILON;
    let g = exp(-0.5 * (x * x) / (sigma * sigma));

    var s = vertex_state[id.x] + params.dt * 0.1 * g;
    s = clamp(s, 0.0, 1.0);
    vertex_state[id.x] = s;
}
