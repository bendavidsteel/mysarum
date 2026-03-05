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

    // Set face data
    let n_faces = faces_list.len();
    mesh.next_face = n_faces;
    for (f, face) in faces_list.iter().enumerate() {
        mesh.face_idx[f] = f as i32;
        let he_idx = edge_dict[&(face[0], face[1])];
        mesh.face_half_edge[f] = he_idx as i32;
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

    // Center to first ring
    let n_seg0 = ring_vertex_counts[0];
    for i in 0..n_seg0 {
        let v0 = 0;
        let v1 = get_vertex_idx(0, i);
        let v2 = get_vertex_idx(0, i + 1);
        faces_list.push([v0, v1, v2]);
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
                faces_list.push([v_inner_curr, v_outer_curr, v_outer_next]);
                outer_idx += 1;
            } else {
                faces_list.push([v_inner_curr, v_outer_curr, v_inner_next]);
                if outer_angle_curr < inner_angle_next {
                    faces_list.push([v_inner_next, v_outer_curr, v_outer_next]);
                    outer_idx += 1;
                }
                inner_idx += 1;
            }
        }
    }

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
                edge_lookup.insert((a, b), idx as i64);
                edge_lookup.insert((b, a), -(idx as i64));
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
    let edge_idx = edge_signed.unsigned_abs() as usize;
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
}

pub fn make_start_mesh(start_shape: StartShape, spring_len: f32, ico_nu: usize) -> Box<HalfEdgeMesh> {
    match start_shape {
        StartShape::Circle => make_circle(spring_len, N_RINGS, SEGMENTS_INNER),
        StartShape::Sphere => {
            let radius = spring_len * N_RINGS as f32 * 0.5;
            make_icosphere(radius, ico_nu)
        }
    }
}
