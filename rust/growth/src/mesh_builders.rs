use std::collections::HashMap;
use nannou::prelude::*;

use crate::mesh::{HalfEdgeMesh, N_RINGS, SEGMENTS_INNER};

// ── Shared half-edge builder from face list ─────────────────────────────────

struct HalfEdgeData {
    dest: usize,
    face: i32,
    twin: i32,
    next: i32,
    prev: i32,
}

/// Build a complete half-edge structure from a face list into an existing mesh.
/// Assumes vertices are already allocated in `mesh`.
pub fn build_halfedge_from_faces(mesh: &mut HalfEdgeMesh, faces_list: &[[usize; 3]]) {
    let mut edge_dict: HashMap<(usize, usize), usize> = HashMap::new();
    let mut half_edge_list: Vec<HalfEdgeData> = Vec::new();

    for (face_i, face) in faces_list.iter().enumerate() {
        let [v0, v1, v2] = *face;
        let face_edges = [(v0, v1), (v1, v2), (v2, v0)];
        let mut face_half_edges = [0usize; 3];

        for (j, &(v_from, v_to)) in face_edges.iter().enumerate() {
            let he_idx = half_edge_list.len();
            half_edge_list.push(HalfEdgeData {
                dest: v_to,
                face: face_i as i32,
                twin: -1,
                next: -1,
                prev: -1,
            });
            edge_dict.insert((v_from, v_to), he_idx);
            face_half_edges[j] = he_idx;
        }

        // Set next/prev within face
        for j in 0..3 {
            half_edge_list[face_half_edges[j]].next = face_half_edges[(j + 1) % 3] as i32;
            half_edge_list[face_half_edges[j]].prev = face_half_edges[(j + 2) % 3] as i32;
        }
    }

    // Create boundary half-edges and set twins
    let edge_dict_snapshot: Vec<((usize, usize), usize)> =
        edge_dict.iter().map(|(&k, &v)| (k, v)).collect();
    let mut boundary_half_edges: Vec<(usize, usize, usize)> = Vec::new();

    for ((v_from, v_to), he_idx) in &edge_dict_snapshot {
        let twin_key = (*v_to, *v_from);
        if let Some(&twin_he) = edge_dict.get(&twin_key) {
            half_edge_list[*he_idx].twin = twin_he as i32;
        } else {
            let boundary_he_idx = half_edge_list.len();
            half_edge_list.push(HalfEdgeData {
                dest: *v_from,
                face: -1,
                twin: *he_idx as i32,
                next: -1,
                prev: -1,
            });
            half_edge_list[*he_idx].twin = boundary_he_idx as i32;
            boundary_half_edges.push((*v_to, *v_from, boundary_he_idx));
        }
    }

    // Connect boundary half-edges
    let boundary_dict: HashMap<(usize, usize), usize> = boundary_half_edges
        .iter()
        .map(|&(vf, vt, he)| ((vf, vt), he))
        .collect();

    for &(_v_from, _v_to, he_idx) in &boundary_half_edges {
        let dest = half_edge_list[he_idx].dest;
        for (&(bnf, _bnt), &bhe) in boundary_dict.iter() {
            if bnf == dest {
                half_edge_list[he_idx].next = bhe as i32;
                half_edge_list[bhe].prev = he_idx as i32;
                break;
            }
        }
    }

    // Copy into mesh arrays
    let n_half_edges = half_edge_list.len();
    mesh.next_half_edge = n_half_edges;
    for (i, he) in half_edge_list.iter().enumerate() {
        mesh.half_edge_idx[i] = i as i32;
        mesh.half_edge_dest[i] = he.dest as i32;
        mesh.half_edge_face[i] = he.face;
        mesh.half_edge_twin[i] = he.twin;
        mesh.half_edge_next[i] = he.next;
        mesh.half_edge_prev[i] = he.prev;
    }

    // Set vertex half-edges
    for v in 0..mesh.next_vertex {
        for (&(vf, _vt), &he_idx) in &edge_dict {
            if vf == v {
                mesh.vertex_half_edge[v] = he_idx as i32;
                break;
            }
        }
    }

    // Fix vertex_half_edge for boundary vertices: walk backward to find the start of the open fan.
    // The forward fan walk (he → twin → next) breaks when twin < 0. For boundary vertices, if
    // vertex_half_edge points to the middle of an open fan, the walk misses neighbors on one side.
    // Walking backward (he → prev(he).twin) finds the outgoing edge at the START of the open fan,
    // so the forward walk traverses ALL neighbors before hitting the boundary at the other end.
    for v in 0..mesh.next_vertex {
        let start = mesh.vertex_half_edge[v];
        if start < 0 { continue; }
        let mut he = start as usize;
        loop {
            let prev = mesh.half_edge_prev[he];
            if prev < 0 { break; }
            let prev_twin = mesh.half_edge_twin[prev as usize];
            if prev_twin < 0 { break; }
            let prev_twin = prev_twin as usize;
            // Back to where we started means closed fan; no fix needed
            if prev_twin == start as usize { break; }
            he = prev_twin;
        }
        mesh.vertex_half_edge[v] = he as i32;
    }

    // Set face data
    let n_faces = faces_list.len();
    mesh.next_face = n_faces;
    for (f, face) in faces_list.iter().enumerate() {
        mesh.face_idx[f] = f as i32;
        let he_idx = edge_dict[&(face[0], face[1])];
        mesh.face_half_edge[f] = he_idx as i32;
    }
}

// ── Disc face triangulation (center fan + ring-to-ring strips) ──────────────

/// Append faces for a disc triangulation: a fan from the center vertex to the
/// first ring, then strips between consecutive rings. `get_vertex_idx` maps
/// (ring, segment) to a vertex index (ring < 0 means the center vertex).
/// `flip` reverses the winding order (for caps whose outward normal points
/// opposite the usual orientation).
fn push_disc_faces(
    faces_list: &mut Vec<[usize; 3]>,
    get_vertex_idx: &dyn Fn(i32, usize) -> usize,
    ring_vertex_counts: &[usize],
    flip: bool,
) {
    let n_rings = ring_vertex_counts.len();
    let mut push = |f: [usize; 3]| {
        faces_list.push(if flip { [f[0], f[2], f[1]] } else { f });
    };

    // Center to first ring
    let n_seg0 = ring_vertex_counts[0];
    for i in 0..n_seg0 {
        let v0 = get_vertex_idx(-1, 0);
        let v1 = get_vertex_idx(0, i);
        let v2 = get_vertex_idx(0, i + 1);
        push([v0, v1, v2]);
    }

    // Ring-to-ring
    for ring_idx in 0..n_rings - 1 {
        let n_inner = ring_vertex_counts[ring_idx];
        let n_outer = ring_vertex_counts[ring_idx + 1];

        let mut inner_idx = 0usize;
        let mut outer_idx = 0usize;

        while inner_idx < n_inner || outer_idx < n_outer {
            let v_inner_curr = get_vertex_idx(ring_idx as i32, inner_idx);
            let v_inner_next = get_vertex_idx(ring_idx as i32, inner_idx + 1);
            let v_outer_curr = get_vertex_idx(ring_idx as i32 + 1, outer_idx);
            let v_outer_next = get_vertex_idx(ring_idx as i32 + 1, outer_idx + 1);

            let inner_angle_next = (inner_idx + 1) as f32 / n_inner as f32;
            let outer_angle_curr = outer_idx as f32 / n_outer as f32;
            let outer_angle_next = (outer_idx + 1) as f32 / n_outer as f32;

            if outer_angle_next < inner_angle_next {
                push([v_inner_curr, v_outer_curr, v_outer_next]);
                outer_idx += 1;
            } else {
                push([v_inner_curr, v_outer_curr, v_inner_next]);
                if outer_angle_curr < inner_angle_next {
                    push([v_inner_next, v_outer_curr, v_outer_next]);
                    outer_idx += 1;
                }
                inner_idx += 1;
            }
        }
    }
}

// ── make_circle: build initial circular disc mesh ───────────────────────────

pub fn make_circle(spring_len: f32, n_rings: usize, segments_inner: usize) -> Box<HalfEdgeMesh> {
    let mut mesh = HalfEdgeMesh::new();

    // Center vertex
    let center = mesh.alloc_vertex().unwrap();
    mesh.vertex_pos[center] = Vec3::ZERO;

    // Ring vertices
    let mut ring_vertex_counts = Vec::new();
    let mut ring_starts = Vec::new();

    for ring_idx in 0..n_rings {
        let radius = spring_len * (ring_idx + 1) as f32;
        let n_segments = segments_inner * (ring_idx + 1);
        ring_vertex_counts.push(n_segments);
        let ring_start = mesh.next_vertex;
        ring_starts.push(ring_start);

        for seg in 0..n_segments {
            let angle = 2.0 * std::f32::consts::PI * seg as f32 / n_segments as f32;
            let v = mesh.alloc_vertex().unwrap();
            mesh.vertex_pos[v] = Vec3::new(radius * angle.cos(), radius * angle.sin(), 0.0);
        }
    }

    let get_vertex_idx = |ring: i32, segment: usize| -> usize {
        if ring < 0 {
            return 0; // center
        }
        let ring = ring as usize;
        ring_starts[ring] + (segment % ring_vertex_counts[ring])
    };

    // Build faces
    let mut faces_list: Vec<[usize; 3]> = Vec::new();
    push_disc_faces(&mut faces_list, &get_vertex_idx, &ring_vertex_counts, false);

    build_halfedge_from_faces(&mut mesh, &faces_list);
    mesh
}

// ── make_hemisphere: dome with a flat bottom cap (closed surface) ────────────

/// Build an initial hemisphere mesh: a dome (center pole vertex plus
/// concentric rings, as in `make_circle`) whose vertices are placed on a
/// hemisphere surface — the center sits at the pole and the outermost ring
/// lies on the equator (z = 0) — closed by a flat disc at z = 0 that shares
/// the equator ring, so the mesh has no boundary. The sphere radius is chosen
/// so the arc spacing between rings along a meridian is roughly `spring_len`.
pub fn make_hemisphere(spring_len: f32, n_rings: usize, segments_inner: usize) -> Box<HalfEdgeMesh> {
    let mut mesh = HalfEdgeMesh::new();

    // Sphere radius: n_rings meridian steps of arc length spring_len span a
    // quarter circle (pole → equator), so R * (PI/2) = spring_len * n_rings.
    let radius = spring_len * n_rings as f32 * 2.0 / std::f32::consts::PI;

    // Map a planar ring index (0..n_rings) to a polar angle from the pole.
    // ring_idx + 1 == n_rings lands exactly on the equator (theta = PI/2).
    let polar_angle = |ring_idx: usize| -> f32 {
        (ring_idx + 1) as f32 / n_rings as f32 * std::f32::consts::FRAC_PI_2
    };

    // Center vertex (pole)
    let center = mesh.alloc_vertex().unwrap();
    mesh.vertex_pos[center] = Vec3::new(0.0, 0.0, radius);

    // Ring vertices
    let mut ring_vertex_counts = Vec::new();
    let mut ring_starts = Vec::new();

    for ring_idx in 0..n_rings {
        let theta = polar_angle(ring_idx);
        let ring_radius = radius * theta.sin();
        let z = radius * theta.cos();
        let n_segments = segments_inner * (ring_idx + 1);
        ring_vertex_counts.push(n_segments);
        let ring_start = mesh.next_vertex;
        ring_starts.push(ring_start);

        for seg in 0..n_segments {
            let angle = 2.0 * std::f32::consts::PI * seg as f32 / n_segments as f32;
            let v = mesh.alloc_vertex().unwrap();
            mesh.vertex_pos[v] = Vec3::new(ring_radius * angle.cos(), ring_radius * angle.sin(), z);
        }
    }

    let get_vertex_idx = |ring: i32, segment: usize| -> usize {
        if ring < 0 {
            return 0; // center
        }
        let ring = ring as usize;
        ring_starts[ring] + (segment % ring_vertex_counts[ring])
    };

    // Bottom cap: flat disc at z = 0 sharing the equator ring. Inner rings are
    // evenly spaced (radius * (ring_idx+1) / n_rings) with the same per-ring
    // segment counts as the dome, plus a center vertex at the origin.
    // The whole cap (equator included) is pinned: physics and growth skip
    // these vertices, so the dome grows from a static base.
    let equator_start = ring_starts[n_rings - 1];
    for v in equator_start..mesh.next_vertex {
        mesh.vertex_pinned[v] = true;
    }
    let mut bottom_ring_starts = Vec::new();

    for ring_idx in 0..n_rings - 1 {
        let ring_radius = radius * (ring_idx + 1) as f32 / n_rings as f32;
        let n_segments = ring_vertex_counts[ring_idx];
        let ring_start = mesh.next_vertex;
        bottom_ring_starts.push(ring_start);

        for seg in 0..n_segments {
            let angle = 2.0 * std::f32::consts::PI * seg as f32 / n_segments as f32;
            let v = mesh.alloc_vertex().unwrap();
            mesh.vertex_pos[v] = Vec3::new(ring_radius * angle.cos(), ring_radius * angle.sin(), 0.0);
            mesh.vertex_pinned[v] = true;
        }
    }

    let bottom_center = mesh.alloc_vertex().unwrap();
    mesh.vertex_pos[bottom_center] = Vec3::ZERO;
    mesh.vertex_pinned[bottom_center] = true;

    let get_bottom_vertex_idx = |ring: i32, segment: usize| -> usize {
        if ring < 0 {
            return bottom_center;
        }
        let ring = ring as usize;
        if ring == n_rings - 1 {
            // Equator ring is shared with the dome
            return ring_starts[ring] + (segment % ring_vertex_counts[ring]);
        }
        bottom_ring_starts[ring] + (segment % ring_vertex_counts[ring])
    };

    // Build faces: dome with the usual winding, bottom cap flipped so its
    // outward normal points down and equator edges get opposite-direction twins.
    let mut faces_list: Vec<[usize; 3]> = Vec::new();
    push_disc_faces(&mut faces_list, &get_vertex_idx, &ring_vertex_counts, false);
    push_disc_faces(&mut faces_list, &get_bottom_vertex_idx, &ring_vertex_counts, true);

    build_halfedge_from_faces(&mut mesh, &faces_list);
    mesh
}

// ── make_icosphere: geodesic icosahedron ────────────────────────────────────

pub fn make_icosphere(radius: f32, nu: usize) -> Box<HalfEdgeMesh> {
    let mut mesh = HalfEdgeMesh::new();

    // Base icosahedron: 12 vertices, 20 faces
    let phi: f32 = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();

    let base_verts: [[f32; 3]; 12] = [
        [0.0, 1.0 / norm, phi / norm],
        [0.0, -1.0 / norm, phi / norm],
        [1.0 / norm, phi / norm, 0.0],
        [-1.0 / norm, phi / norm, 0.0],
        [phi / norm, 0.0, 1.0 / norm],
        [-phi / norm, 0.0, 1.0 / norm],
        // negated
        [0.0, -1.0 / norm, -phi / norm],
        [0.0, 1.0 / norm, -phi / norm],
        [-1.0 / norm, -phi / norm, 0.0],
        [1.0 / norm, -phi / norm, 0.0],
        [-phi / norm, 0.0, -1.0 / norm],
        [phi / norm, 0.0, -1.0 / norm],
    ];

    let base_faces: [[usize; 3]; 20] = [
        [0, 5, 1], [0, 3, 5], [0, 2, 3], [0, 4, 2], [0, 1, 4],
        [1, 5, 8], [5, 3, 10], [3, 2, 7], [2, 4, 11], [4, 1, 9],
        [7, 11, 6], [11, 9, 6], [9, 8, 6], [8, 10, 6], [10, 7, 6],
        [2, 11, 7], [4, 9, 11], [1, 8, 9], [5, 10, 8], [3, 7, 10],
    ];

    if nu <= 1 {
        // No subdivision — just use the base icosahedron
        for bv in &base_verts {
            let v = mesh.alloc_vertex().unwrap();
            mesh.vertex_pos[v] = Vec3::new(bv[0] * radius, bv[1] * radius, bv[2] * radius);
        }
        let faces_list: Vec<[usize; 3]> = base_faces.to_vec();
        build_halfedge_from_faces(&mut mesh, &faces_list);
        return mesh;
    }

    // Subdivision with frequency nu
    // Step 1: collect vertices and edges from the base icosahedron
    let v_count = base_verts.len(); // 12
    let mut vertices: Vec<[f32; 3]> = base_verts.to_vec();

    // Unique edges
    let mut edge_set: Vec<[usize; 2]> = Vec::new();
    let mut edge_lookup: HashMap<(usize, usize), i64> = HashMap::new();

    for face in &base_faces {
        let pairs = [[face[0], face[1]], [face[1], face[2]], [face[0], face[2]]];
        for p in &pairs {
            let (a, b) = if p[0] < p[1] { (p[0], p[1]) } else { (p[1], p[0]) };
            if !edge_lookup.contains_key(&(a, b)) {
                let idx = edge_set.len();
                // Use 1-based signed indices so edge 0 is distinguishable from -0
                edge_lookup.insert((a, b), (idx as i64) + 1);
                edge_lookup.insert((b, a), -((idx as i64) + 1));
                edge_set.push([a, b]);
            }
        }
    }

    let e_count = edge_set.len(); // 30

    // Step 2: add on-edge vertices (nu-1 per edge)
    let w: Vec<f32> = (1..nu).map(|k| k as f32 / nu as f32).collect();

    for e in &edge_set {
        for k in 0..(nu - 1) {
            let w1 = w[nu - 2 - k];
            let w2 = w[k];
            vertices.push([
                w1 * vertices[e[0]][0] + w2 * vertices[e[1]][0],
                w1 * vertices[e[0]][1] + w2 * vertices[e[1]][1],
                w1 * vertices[e[0]][2] + w2 * vertices[e[1]][2],
            ]);
        }
    }

    // Step 3: build face template and vertex ordering
    let template = faces_template(nu);
    let ordering = vertex_ordering(nu);
    let reordered_template: Vec<[usize; 3]> = template
        .iter()
        .map(|f| [ordering[f[0]], ordering[f[1]], ordering[f[2]]])
        .collect();

    // Step 4: process each face — add on-face vertices and build sub-faces
    let mut all_faces: Vec<[usize; 3]> = Vec::new();
    let r: Vec<usize> = (0..(nu - 1)).collect();

    for (fi, face) in base_faces.iter().enumerate() {
        // On-face vertices for this face
        let n_face_interior = (nu as i64 - 1) * (nu as i64 - 2) / 2;
        let t_start = v_count + e_count * (nu - 1) + fi * n_face_interior as usize;
        let t_end = t_start + n_face_interior as usize;
        let t_indices: Vec<usize> = (t_start..t_end).collect();

        // Edge indices (signed convention: positive = forward, negative = reversed)
        let e_ab = edge_lookup[&(face[0], face[1])];
        let e_ac = edge_lookup[&(face[0], face[2])];
        let e_bc = edge_lookup[&(face[1], face[2])];

        let ab: Vec<usize> = resolve_edge_verts(e_ab, v_count, nu, &r);
        let ac: Vec<usize> = resolve_edge_verts(e_ac, v_count, nu, &r);
        let bc: Vec<usize> = resolve_edge_verts(e_bc, v_count, nu, &r);

        // VEF: corners, then on-edge vertices (AB, AC, BC), then on-face vertices
        let mut vef: Vec<usize> = Vec::new();
        vef.push(face[0]);
        vef.push(face[1]);
        vef.push(face[2]);
        vef.extend_from_slice(&ab);
        vef.extend_from_slice(&ac);
        vef.extend_from_slice(&bc);
        vef.extend_from_slice(&t_indices);

        // Build sub-faces for this face
        for sf in &reordered_template {
            all_faces.push([vef[sf[0]], vef[sf[1]], vef[sf[2]]]);
        }

        // Compute on-face vertex positions by interpolation
        compute_inside_points(&mut vertices, &ab, &ac);
    }

    // Normalize all vertices to unit sphere and scale by radius
    for v_pos in vertices.iter_mut() {
        let len = (v_pos[0] * v_pos[0] + v_pos[1] * v_pos[1] + v_pos[2] * v_pos[2]).sqrt();
        if len > 1e-10 {
            v_pos[0] = v_pos[0] / len * radius;
            v_pos[1] = v_pos[1] / len * radius;
            v_pos[2] = v_pos[2] / len * radius;
        }
    }

    // Allocate vertices in mesh
    for v_pos in &vertices {
        let v = mesh.alloc_vertex().unwrap();
        mesh.vertex_pos[v] = Vec3::new(v_pos[0], v_pos[1], v_pos[2]);
    }

    build_halfedge_from_faces(&mut mesh, &all_faces);
    mesh
}

// ── Icosphere helper functions ──────────────────────────────────────────────

/// Resolve on-edge vertex indices, reversing if the edge is stored in opposite direction.
fn resolve_edge_verts(edge_signed: i64, v_count: usize, nu: usize, r: &[usize]) -> Vec<usize> {
    let reversed = edge_signed < 0;
    let edge_idx = (edge_signed.unsigned_abs() - 1) as usize;
    let mut indices: Vec<usize> = r.iter().map(|&k| v_count + edge_idx * (nu - 1) + k).collect();
    if reversed {
        indices.reverse();
    }
    indices
}

/// Faces template for subdivision — generates the triangulation pattern for a
/// subdivided triangle with frequency `nu`.
fn faces_template(nu: usize) -> Vec<[usize; 3]> {
    let mut faces = Vec::new();
    for i in 0..nu {
        let vertex0 = i * (i + 1) / 2;
        let skip = i + 1;
        for j in 0..i {
            faces.push([j + vertex0, j + vertex0 + skip, j + vertex0 + skip + 1]);
            faces.push([j + vertex0, j + vertex0 + skip + 1, j + vertex0 + 1]);
        }
        faces.push([i + vertex0, i + vertex0 + skip, i + vertex0 + skip + 1]);
    }
    faces
}

/// Permutation that reorders reading-order vertex indices into the
/// corners-first, then on-edges, then on-face ordering.
fn vertex_ordering(nu: usize) -> Vec<usize> {
    let left: Vec<usize> = (3..nu + 2).collect();
    let right: Vec<usize> = (nu + 2..2 * nu + 1).collect();
    let bottom: Vec<usize> = (2 * nu + 1..3 * nu).collect();
    let inside: Vec<usize> = (3 * nu..(nu + 1) * (nu + 2) / 2).collect();

    let mut o: Vec<usize> = vec![0]; // topmost corner
    for i in 0..(nu - 1) {
        o.push(left[i]);
        // inside[i*(i-1)/2 .. i*(i+1)/2]
        let inside_start = if i == 0 { 0 } else { i * (i - 1) / 2 };
        let inside_end = i * (i + 1) / 2;
        for k in inside_start..inside_end {
            if k < inside.len() {
                o.push(inside[k]);
            }
        }
        o.push(right[i]);
    }
    o.push(1); // bottom-left corner
    o.extend_from_slice(&bottom);
    o.push(2); // bottom-right corner

    o
}

/// Compute on-face (interior) vertex positions by interpolating between
/// on-edge vertices AB and AC, appending results to the vertices array.
fn compute_inside_points(vertices: &mut Vec<[f32; 3]>, ab: &[usize], ac: &[usize]) {
    for i in 1..ab.len() {
        let w: Vec<f32> = (1..=i).map(|k| k as f32 / (i + 1) as f32).collect();
        for k in 0..i {
            let w1 = w[i - 1 - k];
            let w2 = w[k];
            vertices.push([
                w1 * vertices[ab[i]][0] + w2 * vertices[ac[i]][0],
                w1 * vertices[ab[i]][1] + w2 * vertices[ac[i]][1],
                w1 * vertices[ab[i]][2] + w2 * vertices[ac[i]][2],
            ]);
        }
    }
}

// ── Convenience dispatcher ──────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub enum StartShape {
    Circle,
    Sphere,
    Hemisphere,
}

pub fn make_start_mesh(start_shape: StartShape, spring_len: f32, ico_nu: usize) -> Box<HalfEdgeMesh> {
    let mut mesh = match start_shape {
        StartShape::Circle => make_circle(spring_len, N_RINGS, SEGMENTS_INNER),
        StartShape::Sphere => {
            let radius = spring_len * ico_nu as f32;
            make_icosphere(radius, ico_nu)
        }
        StartShape::Hemisphere => make_hemisphere(spring_len, N_RINGS, SEGMENTS_INNER),
    };
    // Initialize intrinsic edge lengths from extrinsic geometry
    mesh.compute_intrinsic_lengths();
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::validate_mesh;

    #[test]
    fn hemisphere_is_closed() {
        let mesh = make_hemisphere(1.0, N_RINGS, SEGMENTS_INNER);
        validate_mesh(&mesh);

        // No boundary half-edges
        let n_boundary = (0..mesh.next_half_edge)
            .filter(|&he| mesh.half_edge_idx[he] >= 0 && mesh.half_edge_face[he] < 0)
            .count();
        assert_eq!(n_boundary, 0, "closed hemisphere should have no boundary half-edges");

        // Euler characteristic V - E + F = 2 for a closed genus-0 surface
        let v = mesh.next_vertex as i64;
        let e = (mesh.next_half_edge / 2) as i64;
        let f = mesh.next_face as i64;
        assert_eq!(v - e + f, 2, "Euler characteristic should be 2 (V={v}, E={e}, F={f})");

        // Bottom cap (everything at z == 0, equator included) is pinned;
        // dome vertices (z > 0) are free.
        for v in 0..mesh.next_vertex {
            let at_bottom = mesh.vertex_pos[v].z.abs() < 1e-5;
            assert_eq!(
                mesh.vertex_pinned[v], at_bottom,
                "vertex {v} at z={} has pinned={}",
                mesh.vertex_pos[v].z, mesh.vertex_pinned[v]
            );
        }
        let n_pinned = (0..mesh.next_vertex).filter(|&v| mesh.vertex_pinned[v]).count();
        assert!(n_pinned > 0 && n_pinned < mesh.next_vertex);
    }

    #[test]
    fn circle_has_boundary() {
        let mesh = make_circle(1.0, N_RINGS, SEGMENTS_INNER);
        validate_mesh(&mesh);

        let n_boundary = (0..mesh.next_half_edge)
            .filter(|&he| mesh.half_edge_idx[he] >= 0 && mesh.half_edge_face[he] < 0)
            .count();
        assert_eq!(
            n_boundary,
            SEGMENTS_INNER * N_RINGS,
            "circle should have one boundary half-edge per outer-ring segment"
        );
    }
}
