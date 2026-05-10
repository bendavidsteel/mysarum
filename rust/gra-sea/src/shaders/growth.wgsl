// Apply Gaussian growth with cross-channel coupling.
// Per-node params allow each trial to have different growth/coupling.
// conv = chebyshev convolution result (per-channel)
// u = coupling_matrix * conv  (cross-channel mixing)
// growth_i = gaussian(u_i, growth_mu_i, growth_sigma_i)
// state_i += state_dt * growth_i

@group(0) @binding(0) var<storage, read_write> node_state: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> node_u: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> result: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> node_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> node_growth: array<NodeGrowthParams>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_nodes { return; }
    let pos = node_pos[id.x];
    if pos.w < 0.0 { return; }

    let ngp = node_growth[id.x];
    let r = result[id.x];

    if ngp.num_channels == 1u {
        // Single channel mode: no coupling, just use .x
        let u_val = r.x;
        node_u[id.x] = vec4(u_val, 0.0, 0.0, 0.0);

        let x = u_val - ngp.growth_mu.x;
        let sigma = ngp.growth_sigma.x + EPSILON;
        let g = 2.0 * exp(-0.5 * (x * x) / (sigma * sigma)) - 1.0;

        var s = node_state[id.x].x + ngp.state_dt * g;
        s = clamp(s, 0.0, 1.0);
        node_state[id.x] = vec4(s, 0.0, 0.0, 0.0);
    } else {
        // 3-channel mode with cross-channel coupling
        let conv = r.xyz;
        let u = vec3<f32>(
            dot(ngp.coupling_row0.xyz, conv),
            dot(ngp.coupling_row1.xyz, conv),
            dot(ngp.coupling_row2.xyz, conv),
        );
        node_u[id.x] = vec4(u, 0.0);

        // Per-channel growth function
        let x = u - ngp.growth_mu.xyz;
        let sigma = ngp.growth_sigma.xyz + vec3(EPSILON);
        let g = 2.0 * exp(-0.5 * (x * x) / (sigma * sigma)) - vec3(1.0);

        var s = node_state[id.x].xyz + ngp.state_dt * g;
        s = clamp(s, vec3(0.0), vec3(1.0));
        node_state[id.x] = vec4(s, 0.0);
    }
}
