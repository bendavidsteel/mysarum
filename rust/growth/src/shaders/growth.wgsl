// Apply Gaussian growth function: state += dt * gaussian(result, mu, sigma)
// + stochastic intrinsic edge length growth driven by vertex state

// Phototropism mode: direction toward the virtual overhead light source.
// Matches the mesh convention that +Z is "up" (apex = max Z).
const LIGHT_DIR: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

// Unit vector for the preferred anisotropy axis (0 = X, 1 = Y, 2 = Z).
fn anisotropy_axis_vec(axis: u32) -> vec3<f32> {
    if axis == 0u { return vec3<f32>(1.0, 0.0, 0.0); }
    if axis == 1u { return vec3<f32>(0.0, 1.0, 0.0); }
    return vec3<f32>(0.0, 0.0, 1.0);
}

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
@group(0) @binding(1) var<storage, read> result: array<f32>;
@group(0) @binding(2) var<storage, read> vertex_pos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> he_intrinsic_len: array<f32>;
@group(0) @binding(4) var<storage, read> he_packed: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read> vertex_he: array<i32>;
@group(0) @binding(6) var<storage, read> vertex_source: array<f32>;
@group(1) @binding(0) var<uniform> params: SimParams;

// Area-weighted vertex normal: walk the half-edge fan and accumulate the face
// normal of each interior wedge. Each wedge face is (dest, v, next_dest) in
// the mesh's own winding order, so the cross product is consistently oriented
// (outward for the closed sphere) without any per-mesh sign fixup.
fn vertex_normal(v: u32, p: vec3<f32>) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    let start_he = vertex_he[v];
    if start_he < 0 { return acc; }
    let start_dest = he_packed[start_he].x;
    var he = start_he;
    var first = true;

    for (var iter = 0u; iter < 20u; iter += 1u) {
        let data = he_packed[he];
        let dest = data.x;
        let twin = data.y;

        if twin < 0 { break; }
        let twin_data = he_packed[twin];
        let twin_next = twin_data.z;
        if twin_next < 0 { break; }

        // Wedge face containing the incoming edge dest -> v; skip the open
        // boundary wedge (face == -1) on circle meshes.
        if twin_data.w >= 0 && dest >= 0 {
            let next_dest = he_packed[twin_next].x;
            if next_dest >= 0 {
                let p_d = vertex_pos[dest].xyz;
                let p_n = vertex_pos[next_dest].xyz;
                acc += cross(p - p_d, p_n - p_d);
            }
        }

        // Walk fan: he -> twin -> next
        he = twin_next;
        if he_packed[he].x == start_dest && !first { break; }
        first = false;
    }

    let len = length(acc);
    if len > EPSILON { return acc / len; }
    return vec3<f32>(0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }

    let pos = vertex_pos[id.x];
    if pos.w < 0.0 { return; }

    let r = result[id.x];

    var s = vertex_state[id.x];
    if params.growth_mode == 0u {
        // Lenia mode: evolve state via the Gaussian growth function.
        let x = r - params.growth_mu;
        let sigma = params.growth_sigma + EPSILON;
        let g = 2.0 * exp(-0.5 * (x * x) / (sigma * sigma)) - 1.0;
        s = clamp(s + params.state_dt * g, 0.0, 1.0);
        vertex_state[id.x] = s;
    } else if params.growth_mode == 2u {
        // Grow-at-dot mode: the growth potential diffuses outward from a fixed
        // set of source vertices via an explicit heat equation. `result` holds
        // the 1-ring neighbour average of the (read-only) previous state — the
        // Chebyshev pass is configured to compute exactly W(state) for this mode
        // — so the graph Laplacian is just (avg - s), evaluated race-free.
        // dot_diffusion / dot_decay are direct per-step rates (not scaled by
        // state_dt) so the potential spreads in a useful number of frames; the
        // steady-state falloff radius is ~sqrt(dot_diffusion / dot_decay) edges,
        // so a small decay gives a large growth zone (decay 0 fills the mesh).
        let lap = r - s;
        s = s + params.dot_diffusion * lap - params.dot_decay * s;
        // Source vertices are held at full potential (Dirichlet boundary).
        if vertex_source[id.x] > 0.5 {
            s = 1.0;
        }
        s = clamp(s, 0.0, 1.0);
        vertex_state[id.x] = s;
    } else {
        // Phototropism mode (growth_mode == 1): state is how directly this
        // vertex faces the overhead light — the dot product of the vertex
        // normal with the light direction, clamped to [0, 1]. The most
        // light-aligned regions get the highest state and so grow the most
        // via the differential edge growth below.
        let n = vertex_normal(id.x, pos.xyz);
        s = clamp(dot(n, LIGHT_DIR), 0.0, 1.0);
        vertex_state[id.x] = s;
    }

    // Pinned vertices (w == 0): state still evolves above, but their outgoing
    // edges never grow, so a pinned region stays static (no subdivision either).
    if pos.w < 0.5 { return; }

    // Intrinsic edge length growth: walk half-edge fan, grow each outgoing edge.
    // Each vertex only writes to its own outgoing half-edges (no race condition).
    // The spring shader averages he and twin to get the effective rest length.
    let grow = max(2.0 * s - 1.0, 0.0);
    if grow > 0.0 {
        // Resample noise each step by mixing the per-frame seed into the hash,
        // matching the Python reference's per-step uniform(0,1) draw.
        let noise = 0.5 + hash_u32(id.x * 0x9e3779b9u + params.frame_seed);
        let delta = grow * grow * noise * params.state_dt * params.growth_rate;

        let axis = anisotropy_axis_vec(params.anisotropy_axis);
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
                    // Anisotropy: scale this edge's growth by how well its 3D
                    // direction aligns with the preferred axis. |dot| treats the
                    // two edge orientations symmetrically. mix(1, align, strength)
                    // ramps from isotropic (strength 0) to axis-only (strength 1).
                    let edge = vertex_pos[dest].xyz - pos.xyz;
                    let elen = length(edge);
                    var aniso = 1.0;
                    if elen > EPSILON {
                        let align = abs(dot(edge / elen, axis));
                        aniso = mix(1.0, align, params.anisotropy_strength);
                    }
                    let new_len = he_intrinsic_len[he] + delta * aniso;
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
