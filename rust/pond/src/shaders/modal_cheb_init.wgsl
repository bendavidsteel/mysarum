// Modal analysis: Initialize Chebyshev T_0 and T_1 from GRA state luminance.
// Accumulate bandpass coefficients for k=0 and k=1 into modal_amp (8 bands).
// modal_amp layout: 2 vec4 per node (bands 0-3, bands 4-7).
// Concatenated after common.wgsl — SimParams is available.

@group(0) @binding(0) var<storage, read> gra_state: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> t_a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> t_b: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> modal_amp: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> adj_offset: array<u32>;
@group(0) @binding(5) var<storage, read> adj_list: array<u32>;

@group(1) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(1) var<uniform> c0_lo: vec4<f32>;  // bands 0-3 coeff for T_0
@group(1) @binding(2) var<uniform> c0_hi: vec4<f32>;  // bands 4-7 coeff for T_0
@group(1) @binding(3) var<uniform> c1_lo: vec4<f32>;  // bands 0-3 coeff for T_1
@group(1) @binding(4) var<uniform> c1_hi: vec4<f32>;  // bands 4-7 coeff for T_1

fn state_to_scalar(s: vec4<f32>) -> f32 {
    return 0.3 * s.x + 0.6 * s.y + 0.1 * s.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_gra_nodes { return; }

    let signal = state_to_scalar(gra_state[id.x]);
    t_a[id.x] = vec4(signal, 0.0, 0.0, 0.0);

    // W(signal) = average of neighbor signals via adjacency list
    let start = adj_offset[id.x];
    let end = adj_offset[id.x + 1u];
    let count = end - start;
    var avg = signal;
    if count > 0u {
        var sum = 0.0;
        for (var k = start; k < end; k += 1u) {
            sum += state_to_scalar(gra_state[adj_list[k]]);
        }
        avg = sum / f32(count);
    }
    t_b[id.x] = vec4(avg, 0.0, 0.0, 0.0);

    // Initialize modal amplitudes: c0 * T_0 + c1 * T_1
    modal_amp[id.x * 2u]      = c0_lo * signal + c1_lo * avg;
    modal_amp[id.x * 2u + 1u] = c0_hi * signal + c1_hi * avg;
}
