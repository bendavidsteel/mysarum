// Initialize Chebyshev: T_0 = state, T_1 = W(state), result = c0*T_0 + c1*T_1
// Uses adjacency list for averaging operator W

@group(0) @binding(0) var<storage, read> node_state: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_a: array<f32>;   // will hold T_0 (=state)
@group(0) @binding(2) var<storage, read_write> t_b: array<f32>;   // will hold T_1 (=W(state))
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<storage, read> adj_offset: array<u32>;
@group(0) @binding(5) var<storage, read> adj_list: array<u32>;
@group(1) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(1) var<uniform> c0: f32;
@group(1) @binding(2) var<uniform> c1: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_nodes { return; }

    let state_v = node_state[id.x];
    t_a[id.x] = state_v;

    // Compute W(state) = avg_neighbors(state) via adjacency list
    let start = adj_offset[id.x];
    let end = adj_offset[id.x + 1u];
    var avg = state_v;

    let count = end - start;
    if count > 0u {
        var sum = 0.0;
        for (var k = start; k < end; k += 1u) {
            let j = adj_list[k];
            sum += node_state[j];
        }
        avg = sum / f32(count);
    }

    t_b[id.x] = avg;
    result[id.x] = c0 * state_v + c1 * avg;
}
