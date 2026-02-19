// Initialize Chebyshev: T_0 = state, T_1 = L(state), result = c0*T_0 + c1*T_1
// he_packed[i] = vec4<i32>(dest, twin, next, face)

@group(0) @binding(0) var<storage, read> vertex_state: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_a: array<f32>;   // will hold T_0 (=state)
@group(0) @binding(2) var<storage, read_write> t_b: array<f32>;   // will hold T_1 (=L(state))
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<storage, read> he_packed: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read> vertex_he: array<i32>;
@group(1) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(1) var<uniform> c0: f32;
@group(1) @binding(2) var<uniform> c1: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }

    let state_v = vertex_state[id.x];
    t_a[id.x] = state_v;

    // Compute L(state) via fan walk: (state[v] - neighbor_avg) * 0.5
    let start_he = vertex_he[id.x];
    var avg = state_v;

    if start_he >= 0 {
        let start_dest = he_packed[start_he].x;
        var he = start_he;
        var sum = 0.0;
        var count = 0u;
        var first = true;

        for (var iter = 0u; iter < 20u; iter += 1u) {
            let data = he_packed[he];
            let dest = data.x;
            let twin = data.y;
            if dest >= 0 {
                sum += vertex_state[u32(dest)];
                count += 1u;
            }
            if twin < 0 { break; }
            let next = he_packed[twin].z;
            if next < 0 { break; }
            he = next;
            if he_packed[he].x == start_dest && !first { break; }
            first = false;
        }

        if count > 0u {
            avg = sum / f32(count);
        }
    }

    let l_state = (state_v - avg) * 0.5;
    t_b[id.x] = l_state;
    result[id.x] = c0 * state_v + c1 * l_state;
}
