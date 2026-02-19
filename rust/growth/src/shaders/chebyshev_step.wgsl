// Chebyshev step k >= 2: t_next = 2*L(t_curr) - t_prev, result += coeff * t_next
// he_packed[i] = vec4<i32>(dest, twin, next, face)

@group(0) @binding(0) var<storage, read> t_curr: array<f32>;
@group(0) @binding(1) var<storage, read> t_prev: array<f32>;
@group(0) @binding(2) var<storage, read_write> t_next: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<storage, read> he_packed: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read> vertex_he: array<i32>;
@group(1) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(1) var<uniform> coeff: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }

    let curr_v = t_curr[id.x];

    // Compute L(t_curr) via fan walk
    let start_he = vertex_he[id.x];
    var avg = curr_v;

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
                sum += t_curr[u32(dest)];
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

    let l_curr = (curr_v - avg) * 0.5;
    let t_next_val = 2.0 * l_curr - t_prev[id.x];
    t_next[id.x] = t_next_val;
    result[id.x] += coeff * t_next_val;
}
