// Apply Gaussian growth function: state += dt * gaussian(result, mu, sigma)
// + stochastic intrinsic edge length growth driven by vertex state

// Integer hash — maps vertex index to a stable pseudo-random float in [0,1)
fn hash_u32(n: u32) -> f32 {
    var x = n;
    x ^= x >> 16u;
    x *= 0x45d9f3bu;
    x ^= x >> 16u;
    x *= 0x45d9f3bu;
    x ^= x >> 16u;
    return f32(x) / 4294967296.0;
}

@group(0) @binding(0) var<storage, read_write> vertex_state: array<f32>;
@group(0) @binding(1) var<storage, read_write> vertex_u: array<f32>;
@group(0) @binding(2) var<storage, read> result: array<f32>;
@group(0) @binding(3) var<storage, read> vertex_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> he_intrinsic_len: array<f32>;
@group(0) @binding(5) var<storage, read> he_packed: array<vec4<i32>>;
@group(0) @binding(6) var<storage, read> vertex_he: array<i32>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }

    let pos = vertex_pos[id.x];
    if pos.w < 0.0 { return; }

    let r = result[id.x];
    vertex_u[id.x] = r;

    var s = vertex_state[id.x];
    // Lenia mode: evolve state via the Gaussian growth function.
    // Dot mode (growth_mode == 1): leave the seeded state untouched — the
    // static dot keeps driving differential edge growth below.
    if params.growth_mode == 0u {
        let x = r - params.growth_mu;
        let sigma = params.growth_sigma + EPSILON;
        let g = 2.0 * exp(-0.5 * (x * x) / (sigma * sigma)) - 1.0;
        s = clamp(s + params.state_dt * g, 0.0, 1.0);
        vertex_state[id.x] = s;
    }

    // Intrinsic edge length growth: walk half-edge fan, grow each outgoing edge.
    // Each vertex only writes to its own outgoing half-edges (no race condition).
    // The spring shader averages he and twin to get the effective rest length.
    let grow = max(2.0 * s - 1.0, 0.0);
    if grow > 0.0 {
        // Resample noise each step by mixing the per-frame seed into the hash,
        // matching the Python reference's per-step uniform(0,1) draw.
        let noise = 0.5 + hash_u32(id.x * 0x9e3779b9u + params.frame_seed);
        let delta = grow * grow * noise * params.state_dt * params.growth_rate;

        let start_he = vertex_he[id.x];
        if start_he >= 0 {
            let start_dest = he_packed[start_he].x;
            var he = start_he;
            var first = true;

            for (var iter = 0u; iter < 20u; iter += 1u) {
                let data = he_packed[he];
                let dest = data.x;
                let twin = data.y;

                if dest >= 0 {
                    let new_len = he_intrinsic_len[he] + delta;
                    he_intrinsic_len[he] = min(new_len, params.spring_len * 3.0);
                }

                // Walk fan: he -> twin -> next
                if twin < 0 { break; }
                let twin_next = he_packed[twin].z;
                if twin_next < 0 { break; }
                he = twin_next;
                if he_packed[he].x == start_dest && !first { break; }
                first = false;
            }
        }
    }
}
