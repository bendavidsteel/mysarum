// Bulge + Laplacian smoothing forces via half-edge fan walk per vertex.
// Bending is handled by XPBD constraint projection (project_bending.wgsl).
// Spring forces are handled by XPBD spring projection (project_springs.wgsl).
// he_packed[i] = vec4<i32>(dest, twin, next, face)

@group(0) @binding(0) var<storage, read> vertex_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vertex_force: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> he_packed: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read> vertex_he: array<i32>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices { return; }
    let pos = vertex_pos[id.x];
    if pos.w < 0.0 { return; }

    let start_he = vertex_he[id.x];
    if start_he < 0 { return; }

    let start_data = he_packed[start_he];
    let start_dest = start_data.x;
    var he = start_he;
    var bulge_acc = vec3f(0.0);
    var centroid = vec3f(0.0);
    var neighbor_count = 0u;
    var first = true;

    for (var iter = 0u; iter < 20u; iter += 1u) {
        let data = he_packed[he];
        let dest = data.x;
        let twin = data.y;
        let next_he = data.z;
        let face = data.w;

        if dest >= 0 {
            let np = vertex_pos[dest].xyz;

            // Accumulate for Laplacian smoothing
            centroid += np;
            neighbor_count += 1u;
        }

        // Bulge check 1: outgoing boundary edge (this he has face == -1)
        if face == -1 && twin >= 0 && dest >= 0 {
            let ev = vertex_pos[dest].xyz - pos.xyz;
            let twin_data = he_packed[twin];
            let tn = twin_data.z;  // twin.next
            if tn >= 0 {
                let nd = he_packed[tn].x;  // dest of twin.next
                if nd >= 0 {
                    let ne = vertex_pos[nd].xyz - pos.xyz;
                    let sn = cross(ev, ne);
                    let en = cross(ev, sn);
                    let n = length(en);
                    if n > EPSILON { bulge_acc += en / n; }
                }
            }
        }

        // Bulge check 2: incoming boundary edge (twin has face == -1)
        if twin >= 0 {
            let twin_data = he_packed[twin];
            if twin_data.w == -1 && dest >= 0 {
                let sp = vertex_pos[dest].xyz;
                let ev = pos.xyz - sp;
                if next_he >= 0 {
                    let nd = he_packed[next_he].x;
                    if nd >= 0 {
                        let ne = vertex_pos[nd].xyz - sp;
                        let sn = cross(ev, ne);
                        let en = cross(ev, sn);
                        let n = length(en);
                        if n > EPSILON { bulge_acc += en / n; }
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

    // Normalize bulge
    let bn = length(bulge_acc);
    if bn > EPSILON { bulge_acc = bulge_acc / bn; }

    // Laplacian smoothing: push vertex toward neighbor centroid
    var laplacian = vec3f(0.0);
    if neighbor_count > 0u {
        laplacian = centroid / f32(neighbor_count) - pos.xyz;
    }

    let total = params.bulge_strength * bulge_acc
              + params.smoothing_strength * 0.5 * laplacian;
    vertex_force[id.x] = vec4f(vertex_force[id.x].xyz + total, 0.0);
}
