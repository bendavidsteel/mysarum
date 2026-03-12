// Apply Gaussian growth function: state += state_dt * gaussian(result, growth_mu, growth_sigma)

@group(0) @binding(0) var<storage, read_write> node_state: array<f32>;
@group(0) @binding(1) var<storage, read_write> node_u: array<f32>;
@group(0) @binding(2) var<storage, read> result: array<f32>;
@group(0) @binding(3) var<storage, read> node_pos: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_nodes { return; }
    let pos = node_pos[id.x];
    if pos.w < 0.0 { return; }

    let r = result[id.x];
    node_u[id.x] = r;

    let x = r - params.growth_mu;
    let sigma = params.growth_sigma + EPSILON;
    let g = 2.0 * exp(-0.5 * (x * x) / (sigma * sigma)) - 1.0;

    var s = node_state[id.x] + params.state_dt * g;
    s = clamp(s, 0.0, 1.0);
    node_state[id.x] = s;
}
