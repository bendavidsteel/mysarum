// Chebyshev step k >= 2: t_next = 2*W(t_curr) - t_prev, result += coeff * t_next
// Uses adjacency list for averaging operator W

@group(0) @binding(0) var<storage, read> t_curr: array<f32>;
@group(0) @binding(1) var<storage, read> t_prev: array<f32>;
@group(0) @binding(2) var<storage, read_write> t_next: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<storage, read> adj_offset: array<u32>;
@group(0) @binding(5) var<storage, read> adj_list: array<u32>;
@group(1) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(1) var<uniform> coeff: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_nodes { return; }

    let curr_v = t_curr[id.x];

    // Compute W(t_curr) = avg_neighbors(t_curr) via adjacency list
    let start = adj_offset[id.x];
    let end = adj_offset[id.x + 1u];
    var avg = curr_v;

    let count = end - start;
    if count > 0u {
        var sum = 0.0;
        for (var k = start; k < end; k += 1u) {
            let j = adj_list[k];
            sum += t_curr[j];
        }
        avg = sum / f32(count);
    }

    let t_next_val = 2.0 * avg - t_prev[id.x];
    t_next[id.x] = t_next_val;
    result[id.x] += coeff * t_next_val;
}
