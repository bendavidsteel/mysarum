// Modal analysis: Chebyshev step k >= 2.
// Three-term recurrence: t_next = 2*W(t_curr) - t_prev
// Accumulate per-band: modal_amp += ck * t_next
// Concatenated after common.wgsl — SimParams is available.

@group(0) @binding(0) var<storage, read> t_curr: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> t_prev: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> t_next: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> modal_amp: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> adj_offset: array<u32>;
@group(0) @binding(5) var<storage, read> adj_list: array<u32>;

@group(1) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(1) var<uniform> ck_lo: vec4<f32>;  // bands 0-3 coeff for this order
@group(1) @binding(2) var<uniform> ck_hi: vec4<f32>;  // bands 4-7 coeff for this order

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_gra_nodes { return; }

    let curr_val = t_curr[id.x].x;

    // W(t_curr) = average of neighbor values
    let start = adj_offset[id.x];
    let end = adj_offset[id.x + 1u];
    let count = end - start;
    var avg = curr_val;
    if count > 0u {
        var sum = 0.0;
        for (var k = start; k < end; k += 1u) {
            sum += t_curr[adj_list[k]].x;
        }
        avg = sum / f32(count);
    }

    let t_next_val = 2.0 * avg - t_prev[id.x].x;
    t_next[id.x] = vec4(t_next_val, 0.0, 0.0, 0.0);

    // Accumulate bandpass contribution for this Chebyshev order
    modal_amp[id.x * 2u]      += ck_lo * t_next_val;
    modal_amp[id.x * 2u + 1u] += ck_hi * t_next_val;
}
