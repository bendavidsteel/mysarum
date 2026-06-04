// XPBD dihedral angle bending constraint projection (Jacobi style).
// Each vertex gathers bending corrections from all edges it participates in:
// - As ENDPOINT of each outgoing edge (Part 1: fan walk)
// - As OPPOSITE VERTEX of the next edge in each face (Part 2: gather trick)
// Rest dihedral angle = 0 (prefer flat). Dispatch within the XPBD iteration loop.

@group(0) @binding(0) var<storage, read_write> vertex_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> he_packed: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> vertex_he: array<i32>;
@group(1) @binding(0) var<uniform> params: SimParams;

// Compute XPBD bending correction for a vertex given the full edge geometry.
// Returns: correction vector for the vertex, and the constraint violation magnitude.
// grad_v: gradient of dihedral angle w.r.t. this vertex
// sum_grad_sq: sum of squared gradient norms for all 4 vertices
// theta: current dihedral angle (constraint = theta - 0 = theta)
fn xpbd_bend_correction(
    grad_v: vec3f,
    sum_grad_sq: f32,
    theta: f32,
    alpha_tilde: f32,
) -> vec3f {
    let denom = sum_grad_sq + alpha_tilde;
    if denom < EPSILON { return vec3f(0.0); }
    let dlambda = -theta / denom;
    return dlambda * grad_v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }
    let pos = vertex_pos[id.x];
    if pos.w < 0.0 { return; }

    // Compliance is pre-scaled (α̃ in XPBD notation) so behavior is dt-invariant.
    let alpha_tilde = params.bending_compliance;

    var correction = vec3f(0.0);
    var num_constraints = 0u;

    let start_he = vertex_he[id.x];
    if start_he < 0 { return; }

    let start_data = he_packed[start_he];
    let start_dest = start_data.x;
    var he = start_he;
    var first = true;

    for (var iter = 0u; iter < 20u; iter += 1u) {
        let data = he_packed[he];
        let dest = data.x;
        let twin = data.y;
        let next_he = data.z;
        let face = data.w;

        if dest < 0 { break; }
        let np = vertex_pos[dest].xyz;

        // ── Part 1: Vertex as ENDPOINT of edge v→dest ──────────────────
        if twin >= 0 && next_he >= 0 {
            let c_idx = he_packed[next_he].x;
            let twin_data = he_packed[twin];
            let twin_next = twin_data.z;
            if c_idx >= 0 && twin_next >= 0 && twin_data.w >= 0 {
                let d_idx = he_packed[twin_next].x;
                if d_idx >= 0 {
                    let c = vertex_pos[c_idx].xyz;
                    let d = vertex_pos[d_idx].xyz;

                    let e = np - pos.xyz;
                    let fn1 = cross(e, c - pos.xyz);
                    let fn2 = cross(pos.xyz - np, d - np);

                    let a1 = length(fn1);
                    let a2 = length(fn2);
                    let len = max(length(e), EPSILON);

                    if a1 > EPSILON && a2 > EPSILON {
                        let n1 = fn1 / a1;
                        let n2 = fn2 / a2;

                        let e_hat = e / len;
                        let cos_theta = dot(n1, n2);
                        let sin_theta = dot(cross(n1, n2), e_hat);
                        let theta = atan2(sin_theta, cos_theta);

                        // Cotangent weights
                        let vc = pos.xyz - c;
                        let jc = np - c;
                        let area_c = length(cross(vc, jc));
                        var cot_c = 0.0;
                        if area_c > EPSILON { cot_c = clamp(dot(vc, jc) / area_c, -5.0, 5.0); }

                        let vd = pos.xyz - d;
                        let jd = np - d;
                        let area_d = length(cross(vd, jd));
                        var cot_d = 0.0;
                        if area_d > EPSILON { cot_d = clamp(dot(vd, jd) / area_d, -5.0, 5.0); }

                        // Gradients for all 4 vertices
                        let g_a = (cot_c * n1 + cot_d * n2) / len;
                        let g_c = len * n1 / a1;
                        let g_d = len * n2 / a2;
                        let g_b = -(g_a + g_c + g_d);

                        let sum_sq = dot(g_a, g_a) + dot(g_b, g_b) + dot(g_c, g_c) + dot(g_d, g_d);

                        correction += xpbd_bend_correction(g_a, sum_sq, theta, alpha_tilde);
                        num_constraints += 1u;
                    }
                }
            }
        }

        // ── Part 2: Vertex as OPPOSITE VERTEX of next edge in face ─────
        // In triangle (v, dest, w), v is opposite to edge dest→w (= he.next).
        if next_he >= 0 && face >= 0 {
            let opp_he_data = he_packed[next_he];
            let w_idx = opp_he_data.x;
            let opp_twin = opp_he_data.y;
            if w_idx >= 0 && opp_twin >= 0 {
                let opp_twin_data = he_packed[opp_twin];
                let opp_twin_next = opp_twin_data.z;
                if opp_twin_next >= 0 && opp_twin_data.w >= 0 {
                    let d_opp_idx = he_packed[opp_twin_next].x;
                    if d_opp_idx >= 0 {
                        let a = np;
                        let b = vertex_pos[w_idx].xyz;
                        let c_opp = pos.xyz;
                        let d_opp = vertex_pos[d_opp_idx].xyz;

                        let e_opp = b - a;
                        let fn1_opp = cross(e_opp, c_opp - a);
                        let fn2_opp = cross(a - b, d_opp - b);

                        let a1 = length(fn1_opp);
                        let a2 = length(fn2_opp);
                        let len = max(length(e_opp), EPSILON);

                        if a1 > EPSILON && a2 > EPSILON {
                            let n1 = fn1_opp / a1;
                            let n2 = fn2_opp / a2;

                            let e_hat = e_opp / len;
                            let cos_theta = dot(n1, n2);
                            let sin_theta = dot(cross(n1, n2), e_hat);
                            let theta = atan2(sin_theta, cos_theta);

                            // Gradients: v is the C vertex (opposite in face 1)
                            let g_c = len * n1 / a1;

                            // Recompute all gradients for the denominator
                            let ac = a - c_opp;
                            let bc = b - c_opp;
                            let area_ac = length(cross(ac, bc));
                            var cot_c_at_a = 0.0;
                            if area_ac > EPSILON { cot_c_at_a = clamp(dot(ac, bc) / area_ac, -5.0, 5.0); }

                            let ad = a - d_opp;
                            let bd = b - d_opp;
                            let area_ad = length(cross(ad, bd));
                            var cot_d_at_a = 0.0;
                            if area_ad > EPSILON { cot_d_at_a = clamp(dot(ad, bd) / area_ad, -5.0, 5.0); }

                            let g_a = (cot_c_at_a * n1 + cot_d_at_a * n2) / len;
                            let g_d = len * n2 / a2;
                            let g_b = -(g_a + g_c + g_d);

                            let sum_sq = dot(g_a, g_a) + dot(g_b, g_b) + dot(g_c, g_c) + dot(g_d, g_d);

                            correction += xpbd_bend_correction(g_c, sum_sq, theta, alpha_tilde);
                            num_constraints += 1u;
                        }
                    }
                }
            }
        }

        // Walk fan: he -> twin -> next
        if twin < 0 { break; }
        let twin_next = he_packed[twin].z;
        if twin_next < 0 { break; }
        he = twin_next;
        if he_packed[he].x == start_dest && !first { break; }
        first = false;
    }

    if num_constraints > 0u {
        var avg_corr = correction / f32(num_constraints);
        // Clamp correction to prevent instability
        let corr_mag = length(avg_corr);
        let max_corr = params.spring_len * 0.1;
        if corr_mag > max_corr {
            avg_corr = avg_corr * (max_corr / corr_mag);
        }
        // SOR under-relaxation (see SimParams.relaxation)
        vertex_pos[id.x] = vec4f(pos.xyz + params.relaxation * avg_corr, pos.w);
    }
}
