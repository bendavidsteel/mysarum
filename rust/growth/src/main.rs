#![allow(dead_code)]

use std::collections::HashMap;

use nannou::prelude::*;
use bevy::camera::ScalingMode;

// ── Constants ──────────────────────────────────────────────────────────────────

const MAX_VERTICES: usize = 2000;
const MAX_HALF_EDGES: usize = MAX_VERTICES * 6;
const MAX_FACES: usize = MAX_VERTICES * 2;
const EPSILON: f32 = 1e-6;
const N_RINGS: usize = 6;
const SEGMENTS_INNER: usize = 6;

// ── Gaussian helper ────────────────────────────────────────────────────────────

fn gaussian(x: f32, mu: f32, sigma: f32) -> f32 {
    (-0.5 * ((x - mu) / (sigma + EPSILON)).powi(2)).exp()
}

// ── Half-edge mesh (structure-of-arrays, heap-allocated) ───────────────────────

struct HalfEdgeMesh {
    // Face arrays
    face_idx: [i32; MAX_FACES],
    face_half_edge: [i32; MAX_FACES],

    // Vertex arrays
    vertex_idx: [i32; MAX_VERTICES],
    vertex_half_edge: [i32; MAX_VERTICES],
    vertex_pos: [Vec3; MAX_VERTICES],
    vertex_state: [f32; MAX_VERTICES],
    vertex_u: [f32; MAX_VERTICES],

    // Half-edge arrays
    half_edge_idx: [i32; MAX_HALF_EDGES],
    half_edge_twin: [i32; MAX_HALF_EDGES],
    half_edge_dest: [i32; MAX_HALF_EDGES],
    half_edge_face: [i32; MAX_HALF_EDGES],
    half_edge_next: [i32; MAX_HALF_EDGES],
    half_edge_prev: [i32; MAX_HALF_EDGES],

    // Allocation watermarks
    next_vertex: usize,
    next_face: usize,
    next_half_edge: usize,
}

impl HalfEdgeMesh {
    fn new() -> Box<Self> {
        let mut mesh = Box::new(HalfEdgeMesh {
            face_idx: [-1; MAX_FACES],
            face_half_edge: [-1; MAX_FACES],
            vertex_idx: [-1; MAX_VERTICES],
            vertex_half_edge: [-1; MAX_VERTICES],
            vertex_pos: [Vec3::ZERO; MAX_VERTICES],
            vertex_state: [0.0; MAX_VERTICES],
            vertex_u: [0.0; MAX_VERTICES],
            half_edge_idx: [-1; MAX_HALF_EDGES],
            half_edge_twin: [-1; MAX_HALF_EDGES],
            half_edge_dest: [-1; MAX_HALF_EDGES],
            half_edge_face: [-1; MAX_HALF_EDGES],
            half_edge_next: [-1; MAX_HALF_EDGES],
            half_edge_prev: [-1; MAX_HALF_EDGES],
            next_vertex: 0,
            next_face: 0,
            next_half_edge: 0,
        });
        // initialise vertex_state with random values
        for s in mesh.vertex_state.iter_mut() {
            *s = random_f32();
        }
        mesh
    }

    fn alloc_vertex(&mut self) -> Option<usize> {
        if self.next_vertex >= MAX_VERTICES {
            return None;
        }
        let idx = self.next_vertex;
        self.next_vertex += 1;
        self.vertex_idx[idx] = idx as i32;
        Some(idx)
    }

    fn alloc_face(&mut self) -> Option<usize> {
        if self.next_face >= MAX_FACES {
            return None;
        }
        let idx = self.next_face;
        self.next_face += 1;
        self.face_idx[idx] = idx as i32;
        Some(idx)
    }

    fn alloc_half_edge(&mut self) -> Option<usize> {
        if self.next_half_edge >= MAX_HALF_EDGES {
            return None;
        }
        let idx = self.next_half_edge;
        self.next_half_edge += 1;
        self.half_edge_idx[idx] = idx as i32;
        Some(idx)
    }

    fn is_vertex_active(&self, i: usize) -> bool {
        i < self.next_vertex && self.vertex_idx[i] >= 0
    }

    fn is_face_active(&self, i: usize) -> bool {
        i < self.next_face && self.face_idx[i] >= 0
    }

    fn is_half_edge_active(&self, i: usize) -> bool {
        i < self.next_half_edge && self.half_edge_idx[i] >= 0
    }

    fn active_vertex_count(&self) -> usize {
        (0..self.next_vertex)
            .filter(|&i| self.vertex_idx[i] >= 0)
            .count()
    }

    // ── Get half-edge from vertex a to vertex b ────────────────────────────

    fn get_vertex_half_edge(&self, vertex_a: usize, vertex_b: usize) -> Option<usize> {
        let start_he = self.vertex_half_edge[vertex_a];
        if start_he < 0 {
            return None;
        }
        let start = start_he as usize;
        let mut he = start;
        let mut first = true;
        loop {
            if self.half_edge_dest[he] == vertex_b as i32 {
                return Some(he);
            }
            let twin = self.half_edge_twin[he];
            if twin < 0 {
                return None;
            }
            let next = self.half_edge_next[twin as usize];
            if next < 0 {
                return None;
            }
            he = next as usize;
            if he == start && !first {
                return None;
            }
            first = false;
        }
    }

    // ── Add external triangle at boundary edge a→b ─────────────────────────

    fn add_external_triangle(&mut self, vertex_a: usize, vertex_b: usize) -> bool {
        let half_edge_ab = match self.get_vertex_half_edge(vertex_a, vertex_b) {
            Some(he) => he,
            None => return false,
        };
        // must be a boundary half-edge (face == -1)
        if self.half_edge_face[half_edge_ab] != -1 {
            return false;
        }

        let new_face = match self.alloc_face() {
            Some(f) => f,
            None => return false,
        };
        let vertex_c = match self.alloc_vertex() {
            Some(v) => v,
            None => return false,
        };
        let half_edge_bc = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_ca = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_cb = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_ac = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };

        // New vertex at midpoint with z-perturbation to allow 3D buckling
        let mut new_pos = (self.vertex_pos[vertex_a] + self.vertex_pos[vertex_b]) * 0.5;
        let edge_len = (self.vertex_pos[vertex_b] - self.vertex_pos[vertex_a]).length();
        new_pos.z += (random_f32() - 0.5) * edge_len * 0.05;
        self.vertex_pos[vertex_c] = new_pos;
        self.vertex_state[vertex_c] = (self.vertex_state[vertex_a] + self.vertex_state[vertex_b]) * 0.5;
        self.vertex_half_edge[vertex_c] = half_edge_ca as i32;

        // Face
        self.face_half_edge[new_face] = half_edge_ab as i32;

        // Existing connectivity
        let half_edge_b_next = self.half_edge_next[half_edge_ab] as usize;
        let half_edge_a_prev = self.half_edge_prev[half_edge_ab] as usize;

        // half_edge_ab now belongs to the new face
        self.half_edge_face[half_edge_ab] = new_face as i32;

        // Set half-edge faces
        self.half_edge_face[half_edge_bc] = new_face as i32;
        self.half_edge_face[half_edge_ca] = new_face as i32;
        self.half_edge_face[half_edge_cb] = -1;
        self.half_edge_face[half_edge_ac] = -1;

        // Destinations
        self.half_edge_dest[half_edge_bc] = vertex_c as i32;
        self.half_edge_dest[half_edge_ca] = vertex_a as i32;
        self.half_edge_dest[half_edge_cb] = vertex_b as i32;
        self.half_edge_dest[half_edge_ac] = vertex_c as i32;

        // Twins
        self.half_edge_twin[half_edge_bc] = half_edge_cb as i32;
        self.half_edge_twin[half_edge_cb] = half_edge_bc as i32;
        self.half_edge_twin[half_edge_ca] = half_edge_ac as i32;
        self.half_edge_twin[half_edge_ac] = half_edge_ca as i32;

        // Next pointers
        self.half_edge_next[half_edge_ab] = half_edge_bc as i32;
        self.half_edge_next[half_edge_bc] = half_edge_ca as i32;
        self.half_edge_next[half_edge_ca] = half_edge_ab as i32;
        self.half_edge_next[half_edge_cb] = half_edge_b_next as i32;
        self.half_edge_next[half_edge_ac] = half_edge_cb as i32;
        self.half_edge_next[half_edge_a_prev] = half_edge_ac as i32;

        // Prev pointers
        self.half_edge_prev[half_edge_ab] = half_edge_ca as i32;
        self.half_edge_prev[half_edge_bc] = half_edge_ab as i32;
        self.half_edge_prev[half_edge_ca] = half_edge_bc as i32;
        self.half_edge_prev[half_edge_cb] = half_edge_ac as i32;
        self.half_edge_prev[half_edge_ac] = half_edge_a_prev as i32;
        self.half_edge_prev[half_edge_b_next] = half_edge_cb as i32;

        true
    }

    // ── Add internal edge triangle (edge a→b internal, twin on boundary) ───

    fn add_internal_edge_triangle(&mut self, vertex_a: usize, vertex_b: usize) -> bool {
        let half_edge_ab = match self.get_vertex_half_edge(vertex_a, vertex_b) {
            Some(he) => he,
            None => return false,
        };
        let half_edge_ba = self.half_edge_twin[half_edge_ab];
        if half_edge_ba < 0 {
            return false;
        }
        let half_edge_ba = half_edge_ba as usize;

        // ab must be internal, ba must be boundary
        if self.half_edge_face[half_edge_ab] == -1 || self.half_edge_face[half_edge_ba] != -1 {
            return false;
        }

        let half_edge_bc = self.half_edge_next[half_edge_ab] as usize;
        let half_edge_ca = self.half_edge_next[half_edge_bc] as usize;
        let vertex_c = self.half_edge_dest[half_edge_bc] as usize;
        let face_abc = self.half_edge_face[half_edge_ab];

        // Allocate
        let vertex_d = match self.alloc_vertex() {
            Some(v) => v,
            None => return false,
        };
        let face_dbc = match self.alloc_face() {
            Some(f) => f,
            None => return false,
        };
        let half_edge_db = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_cd = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_dc = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_da = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };

        // Reuse names from Python
        let half_edge_ad = half_edge_ab; // repurposed
        let half_edge_bd = half_edge_ba; // repurposed
        let face_adc = face_abc; // repurposed

        let half_edge_ba_next = self.half_edge_next[half_edge_ba] as usize;
        let half_edge_ba_prev = self.half_edge_prev[half_edge_ba] as usize;

        // New vertex with z-perturbation to allow 3D buckling
        let mut new_pos = (self.vertex_pos[vertex_a] + self.vertex_pos[vertex_b]) * 0.5;
        let edge_len = (self.vertex_pos[vertex_b] - self.vertex_pos[vertex_a]).length();
        new_pos.z += (random_f32() - 0.5) * edge_len * 0.05;
        self.vertex_pos[vertex_d] = new_pos;
        self.vertex_state[vertex_d] = (self.vertex_state[vertex_a] + self.vertex_state[vertex_b]) * 0.5;
        self.vertex_half_edge[vertex_d] = half_edge_db as i32;

        // Faces
        self.face_half_edge[face_dbc as usize] = half_edge_bc as i32;
        self.face_half_edge[face_adc as usize] = half_edge_ca as i32;

        // Destinations
        self.half_edge_dest[half_edge_ad] = vertex_d as i32;
        self.half_edge_dest[half_edge_db] = vertex_b as i32;
        self.half_edge_dest[half_edge_bd] = vertex_d as i32;
        self.half_edge_dest[half_edge_da] = vertex_a as i32;
        self.half_edge_dest[half_edge_dc] = vertex_c as i32;
        self.half_edge_dest[half_edge_cd] = vertex_d as i32;

        // Twins
        self.half_edge_twin[half_edge_ad] = half_edge_da as i32;
        self.half_edge_twin[half_edge_da] = half_edge_ad as i32;
        self.half_edge_twin[half_edge_db] = half_edge_bd as i32;
        self.half_edge_twin[half_edge_bd] = half_edge_db as i32;
        self.half_edge_twin[half_edge_dc] = half_edge_cd as i32;
        self.half_edge_twin[half_edge_cd] = half_edge_dc as i32;

        // Faces
        self.half_edge_face[half_edge_ad] = face_adc;
        self.half_edge_face[half_edge_dc] = face_adc;
        self.half_edge_face[half_edge_ca] = face_adc;
        self.half_edge_face[half_edge_db] = face_dbc as i32;
        self.half_edge_face[half_edge_bc] = face_dbc as i32;
        self.half_edge_face[half_edge_cd] = face_dbc as i32;
        self.half_edge_face[half_edge_bd] = -1;
        self.half_edge_face[half_edge_da] = -1;

        // Next
        self.half_edge_next[half_edge_ad] = half_edge_dc as i32;
        self.half_edge_next[half_edge_dc] = half_edge_ca as i32;
        self.half_edge_next[half_edge_ca] = half_edge_ad as i32;
        self.half_edge_next[half_edge_db] = half_edge_bc as i32;
        self.half_edge_next[half_edge_bc] = half_edge_cd as i32;
        self.half_edge_next[half_edge_cd] = half_edge_db as i32;
        self.half_edge_next[half_edge_bd] = half_edge_da as i32;
        self.half_edge_next[half_edge_da] = half_edge_ba_next as i32;
        self.half_edge_next[half_edge_ba_prev] = half_edge_bd as i32;

        // Prev
        self.half_edge_prev[half_edge_ad] = half_edge_ca as i32;
        self.half_edge_prev[half_edge_dc] = half_edge_ad as i32;
        self.half_edge_prev[half_edge_ca] = half_edge_dc as i32;
        self.half_edge_prev[half_edge_db] = half_edge_cd as i32;
        self.half_edge_prev[half_edge_bc] = half_edge_db as i32;
        self.half_edge_prev[half_edge_cd] = half_edge_bc as i32;
        self.half_edge_prev[half_edge_da] = half_edge_bd as i32;
        self.half_edge_prev[half_edge_bd] = half_edge_ba_prev as i32;
        self.half_edge_prev[half_edge_ba_next] = half_edge_da as i32;

        true
    }

    // ── Add internal triangles (split fully internal edge) ─────────────────

    fn add_internal_triangles(&mut self, vertex_a: usize, vertex_b: usize) -> bool {
        let half_edge_ab = match self.get_vertex_half_edge(vertex_a, vertex_b) {
            Some(he) => he,
            None => return false,
        };
        let half_edge_ba = self.half_edge_twin[half_edge_ab];
        if half_edge_ba < 0 {
            return false;
        }
        let half_edge_ba = half_edge_ba as usize;

        // Both sides must be internal
        if self.half_edge_face[half_edge_ab] == -1 || self.half_edge_face[half_edge_ba] == -1 {
            return false;
        }

        let half_edge_bc_old = self.half_edge_next[half_edge_ab] as usize;
        let half_edge_ca = self.half_edge_next[half_edge_bc_old] as usize;
        let half_edge_ad = self.half_edge_next[half_edge_ba] as usize;
        let half_edge_db = self.half_edge_next[half_edge_ad] as usize;

        let vertex_c = self.half_edge_dest[half_edge_bc_old] as usize;
        let vertex_d = self.half_edge_dest[half_edge_ad] as usize;

        let face_abc = self.half_edge_face[half_edge_ab];
        let face_bad = self.half_edge_face[half_edge_ba];

        // Allocate new vertex
        let vertex_e = match self.alloc_vertex() {
            Some(v) => v,
            None => return false,
        };
        let face_ebc = match self.alloc_face() {
            Some(f) => f,
            None => return false,
        };
        let face_ead = match self.alloc_face() {
            Some(f) => f,
            None => return false,
        };

        let half_edge_ec = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_ea = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_ce = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_eb = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_de = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };
        let half_edge_ed = match self.alloc_half_edge() {
            Some(h) => h,
            None => return false,
        };

        // Repurposed
        let half_edge_ae = half_edge_ab;
        let half_edge_be = half_edge_ba;
        let face_aec = face_abc;
        let face_bed = face_bad;

        // New vertex position with z-perturbation to allow 3D buckling
        let mut new_pos = (self.vertex_pos[vertex_a] + self.vertex_pos[vertex_b]) * 0.5;
        let edge_len = (self.vertex_pos[vertex_b] - self.vertex_pos[vertex_a]).length();
        new_pos.z += (random_f32() - 0.5) * edge_len * 0.05;
        self.vertex_pos[vertex_e] = new_pos;
        self.vertex_state[vertex_e] = (self.vertex_state[vertex_a] + self.vertex_state[vertex_b]) * 0.5;
        self.vertex_half_edge[vertex_e] = half_edge_eb as i32;

        // Faces (must update repurposed faces too, not just new ones)
        self.face_half_edge[face_aec as usize] = half_edge_ae as i32;
        self.face_half_edge[face_bed as usize] = half_edge_be as i32;
        self.face_half_edge[face_ebc as usize] = half_edge_eb as i32;
        self.face_half_edge[face_ead as usize] = half_edge_ea as i32;

        // Destinations
        self.half_edge_dest[half_edge_ae] = vertex_e as i32;
        self.half_edge_dest[half_edge_ec] = vertex_c as i32;
        self.half_edge_dest[half_edge_eb] = vertex_b as i32;
        self.half_edge_dest[half_edge_ce] = vertex_e as i32;
        self.half_edge_dest[half_edge_ea] = vertex_a as i32;
        self.half_edge_dest[half_edge_de] = vertex_e as i32;
        self.half_edge_dest[half_edge_be] = vertex_e as i32;
        self.half_edge_dest[half_edge_ed] = vertex_d as i32;

        // Twins
        self.half_edge_twin[half_edge_ae] = half_edge_ea as i32;
        self.half_edge_twin[half_edge_ea] = half_edge_ae as i32;
        self.half_edge_twin[half_edge_ec] = half_edge_ce as i32;
        self.half_edge_twin[half_edge_ce] = half_edge_ec as i32;
        self.half_edge_twin[half_edge_eb] = half_edge_be as i32;
        self.half_edge_twin[half_edge_be] = half_edge_eb as i32;
        self.half_edge_twin[half_edge_de] = half_edge_ed as i32;
        self.half_edge_twin[half_edge_ed] = half_edge_de as i32;

        // Faces
        self.half_edge_face[half_edge_ae] = face_aec as i32;
        self.half_edge_face[half_edge_ec] = face_aec as i32;
        self.half_edge_face[half_edge_ca] = face_aec as i32;
        self.half_edge_face[half_edge_eb] = face_ebc as i32;
        self.half_edge_face[half_edge_bc_old] = face_ebc as i32;
        self.half_edge_face[half_edge_ce] = face_ebc as i32;
        self.half_edge_face[half_edge_ea] = face_ead as i32;
        self.half_edge_face[half_edge_ad] = face_ead as i32;
        self.half_edge_face[half_edge_de] = face_ead as i32;
        self.half_edge_face[half_edge_be] = face_bed as i32;
        self.half_edge_face[half_edge_ed] = face_bed as i32;
        self.half_edge_face[half_edge_db] = face_bed as i32;

        // Next
        self.half_edge_next[half_edge_ae] = half_edge_ec as i32;
        self.half_edge_next[half_edge_ec] = half_edge_ca as i32;
        self.half_edge_next[half_edge_ca] = half_edge_ae as i32;
        self.half_edge_next[half_edge_eb] = half_edge_bc_old as i32;
        self.half_edge_next[half_edge_bc_old] = half_edge_ce as i32;
        self.half_edge_next[half_edge_ce] = half_edge_eb as i32;
        self.half_edge_next[half_edge_ea] = half_edge_ad as i32;
        self.half_edge_next[half_edge_ad] = half_edge_de as i32;
        self.half_edge_next[half_edge_de] = half_edge_ea as i32;
        self.half_edge_next[half_edge_be] = half_edge_ed as i32;
        self.half_edge_next[half_edge_ed] = half_edge_db as i32;
        self.half_edge_next[half_edge_db] = half_edge_be as i32;

        // Prev
        self.half_edge_prev[half_edge_ae] = half_edge_ca as i32;
        self.half_edge_prev[half_edge_ec] = half_edge_ae as i32;
        self.half_edge_prev[half_edge_ca] = half_edge_ec as i32;
        self.half_edge_prev[half_edge_eb] = half_edge_ce as i32;
        self.half_edge_prev[half_edge_bc_old] = half_edge_eb as i32;
        self.half_edge_prev[half_edge_ce] = half_edge_bc_old as i32;
        self.half_edge_prev[half_edge_ea] = half_edge_de as i32;
        self.half_edge_prev[half_edge_ad] = half_edge_ea as i32;
        self.half_edge_prev[half_edge_de] = half_edge_ad as i32;
        self.half_edge_prev[half_edge_be] = half_edge_db as i32;
        self.half_edge_prev[half_edge_ed] = half_edge_be as i32;
        self.half_edge_prev[half_edge_db] = half_edge_ed as i32;

        true
    }

    // ── Flip edge ──────────────────────────────────────────────────────────

    fn flip_edge(&mut self, half_edge_ab: usize) -> bool {
        let half_edge_ba_i = self.half_edge_twin[half_edge_ab];
        if half_edge_ba_i < 0 {
            return false;
        }
        let half_edge_ba = half_edge_ba_i as usize;

        // Both sides must be internal
        if self.half_edge_face[half_edge_ab] == -1 || self.half_edge_face[half_edge_ba] == -1 {
            return false;
        }

        let vertex_a = self.half_edge_dest[half_edge_ba] as usize;
        let vertex_b = self.half_edge_dest[half_edge_ab] as usize;

        let half_edge_bc = self.half_edge_next[half_edge_ab] as usize;
        let vertex_c = self.half_edge_dest[half_edge_bc] as usize;
        let half_edge_ca = self.half_edge_next[half_edge_bc] as usize;

        let half_edge_ad = self.half_edge_next[half_edge_ba] as usize;
        let vertex_d = self.half_edge_dest[half_edge_ad] as usize;
        let half_edge_db = self.half_edge_next[half_edge_ad] as usize;

        let face_abc = self.half_edge_face[half_edge_ab];
        let face_bad = self.half_edge_face[half_edge_ba];

        let face_adc = face_abc;
        let face_bcd = face_bad;

        let half_edge_dc = half_edge_ab;
        let half_edge_cd = half_edge_ba;

        // Face half-edges
        self.face_half_edge[face_adc as usize] = half_edge_ca as i32;
        self.face_half_edge[face_bcd as usize] = half_edge_db as i32;

        // Vertex half-edges (ensure they don't point to flipped edge)
        self.vertex_half_edge[vertex_a] = half_edge_ad as i32;
        self.vertex_half_edge[vertex_b] = half_edge_bc as i32;

        // Destinations
        self.half_edge_dest[half_edge_dc] = vertex_c as i32;
        self.half_edge_dest[half_edge_cd] = vertex_d as i32;

        // Next
        self.half_edge_next[half_edge_dc] = half_edge_ca as i32;
        self.half_edge_next[half_edge_ca] = half_edge_ad as i32;
        self.half_edge_next[half_edge_ad] = half_edge_dc as i32;
        self.half_edge_next[half_edge_cd] = half_edge_db as i32;
        self.half_edge_next[half_edge_db] = half_edge_bc as i32;
        self.half_edge_next[half_edge_bc] = half_edge_cd as i32;

        // Prev
        self.half_edge_prev[half_edge_dc] = half_edge_ad as i32;
        self.half_edge_prev[half_edge_ca] = half_edge_dc as i32;
        self.half_edge_prev[half_edge_ad] = half_edge_ca as i32;
        self.half_edge_prev[half_edge_cd] = half_edge_bc as i32;
        self.half_edge_prev[half_edge_db] = half_edge_cd as i32;
        self.half_edge_prev[half_edge_bc] = half_edge_db as i32;

        // Faces
        self.half_edge_face[half_edge_dc] = face_adc;
        self.half_edge_face[half_edge_ca] = face_adc;
        self.half_edge_face[half_edge_ad] = face_adc;
        self.half_edge_face[half_edge_cd] = face_bcd;
        self.half_edge_face[half_edge_db] = face_bcd;
        self.half_edge_face[half_edge_bc] = face_bcd;

        true
    }

    // ── Edge count per vertex (walk half-edge fan) ─────────────────────────

    fn get_edge_count(&self, vertex: usize) -> usize {
        let start_he = self.vertex_half_edge[vertex];
        if start_he < 0 {
            return 0;
        }
        let start = start_he as usize;
        let start_dest = self.half_edge_dest[start];
        let mut he = start;
        let mut count = 1;
        loop {
            let twin = self.half_edge_twin[he];
            if twin < 0 {
                break;
            }
            let next = self.half_edge_next[twin as usize];
            if next < 0 {
                break;
            }
            he = next as usize;
            if self.half_edge_dest[he] == start_dest {
                break;
            }
            count += 1;
        }
        count
    }

    // ── Mesh refinement (flip edges to optimize valences) ──────────────────

    fn refine_mesh(&mut self) {
        let mut edge_counts = vec![0usize; self.next_vertex];
        for v in 0..self.next_vertex {
            if self.vertex_idx[v] >= 0 {
                edge_counts[v] = self.get_edge_count(v);
            }
        }

        // Collect edges to potentially flip (only consider he where dest < dest_twin to avoid duplicates)
        let mut edges_to_check: Vec<usize> = Vec::new();
        for he in 0..self.next_half_edge {
            if self.half_edge_idx[he] < 0 {
                continue;
            }
            if self.half_edge_face[he] == -1 {
                continue;
            }
            let twin = self.half_edge_twin[he];
            if twin < 0 || self.half_edge_face[twin as usize] == -1 {
                continue;
            }
            let dest_a = self.half_edge_dest[he];
            let dest_b = self.half_edge_dest[twin as usize];
            if dest_a < dest_b {
                edges_to_check.push(he);
            }
        }

        for he in edges_to_check {
            // Re-check validity since flips may have changed things
            if self.half_edge_idx[he] < 0 {
                continue;
            }
            let twin_i = self.half_edge_twin[he];
            if twin_i < 0 {
                continue;
            }
            let twin = twin_i as usize;
            if self.half_edge_face[he] == -1 || self.half_edge_face[twin] == -1 {
                continue;
            }

            let va = self.half_edge_dest[twin] as usize;
            let vb = self.half_edge_dest[he] as usize;
            let vc = self.half_edge_dest[self.half_edge_next[he] as usize] as usize;
            let vd = self.half_edge_dest[self.half_edge_next[twin] as usize] as usize;

            let ec_a = edge_counts[va] as i32;
            let ec_b = edge_counts[vb] as i32;
            let ec_c = edge_counts[vc] as i32;
            let ec_d = edge_counts[vd] as i32;

            let no_flip_cost = (ec_a - 6).pow(2) + (ec_b - 6).pow(2) + (ec_c - 6).pow(2) + (ec_d - 6).pow(2);
            let flip_cost = (ec_a - 7).pow(2) + (ec_b - 7).pow(2) + (ec_c - 5).pow(2) + (ec_d - 5).pow(2);

            if flip_cost < no_flip_cost {
                if self.flip_edge(he) {
                    edge_counts[va] -= 1;
                    edge_counts[vb] -= 1;
                    edge_counts[vc] += 1;
                    edge_counts[vd] += 1;
                }
            }
        }
    }
}

// ── make_circle: build initial circular disc mesh ──────────────────────────────

fn make_circle(spring_len: f32, n_rings: usize, segments_inner: usize) -> Box<HalfEdgeMesh> {
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

    // Build half-edge structure from face list
    let mut edge_dict: HashMap<(usize, usize), usize> = HashMap::new();

    struct HalfEdgeData {
        dest: usize,
        face: i32,
        twin: i32,
        next: i32,
        prev: i32,
    }

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
    let edge_dict_snapshot: Vec<((usize, usize), usize)> = edge_dict.iter().map(|(&k, &v)| (k, v)).collect();
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
        // Find next boundary half-edge (the one starting at this one's destination = v_from's dest)
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

    mesh
}

// ── Force calculations ─────────────────────────────────────────────────────────

fn calculate_spring_force(mesh: &HalfEdgeMesh, spring_len: f32, elastic_constant: f32) -> Vec<Vec3> {
    let mut forces = vec![Vec3::ZERO; mesh.next_vertex];

    for he in 0..mesh.next_half_edge {
        if mesh.half_edge_idx[he] < 0 {
            continue;
        }
        let dest = mesh.half_edge_dest[he];
        if dest < 0 {
            continue;
        }
        let prev = mesh.half_edge_prev[he];
        if prev < 0 {
            continue;
        }
        let src = mesh.half_edge_dest[prev as usize];
        if src < 0 {
            continue;
        }

        let dest = dest as usize;
        let src = src as usize;

        let edge_vec = mesh.vertex_pos[dest] - mesh.vertex_pos[src];
        let length = edge_vec.length();
        let safe_len = length.max(EPSILON);
        let force = -1.0 * (length - spring_len) * elastic_constant * (edge_vec / safe_len);

        forces[dest] += force;
        forces[src] -= force;
    }

    forces
}

fn calculate_repulsion_force(mesh: &HalfEdgeMesh, repulsion_distance: f32) -> Vec<Vec3> {
    let mut forces = vec![Vec3::ZERO; mesh.next_vertex];

    for i in 0..mesh.next_vertex {
        if mesh.vertex_idx[i] < 0 {
            continue;
        }
        for j in (i + 1)..mesh.next_vertex {
            if mesh.vertex_idx[j] < 0 {
                continue;
            }
            let diff = mesh.vertex_pos[i] - mesh.vertex_pos[j];
            let dist = (diff.length_squared() + EPSILON).sqrt();

            if dist < repulsion_distance {
                let ratio = ((repulsion_distance - dist) / repulsion_distance).max(0.0);
                let force = ratio * ratio * (diff / dist);
                forces[i] += force;
                forces[j] -= force;
            }
        }
    }

    forces
}

fn calculate_planar_force(mesh: &HalfEdgeMesh) -> Vec<Vec3> {
    let mut forces = vec![Vec3::ZERO; mesh.next_vertex];

    for v in 0..mesh.next_vertex {
        if mesh.vertex_idx[v] < 0 {
            continue;
        }
        let start_he = mesh.vertex_half_edge[v];
        if start_he < 0 {
            continue;
        }
        let start = start_he as usize;
        let start_dest = mesh.half_edge_dest[start];

        let mut sum = Vec3::ZERO;
        let mut count = 0;
        let mut he = start;
        let mut first = true;

        loop {
            let dest = mesh.half_edge_dest[he];
            if dest >= 0 {
                sum += mesh.vertex_pos[dest as usize];
                count += 1;
            }

            let twin = mesh.half_edge_twin[he];
            if twin < 0 {
                break;
            }
            let next = mesh.half_edge_next[twin as usize];
            if next < 0 {
                break;
            }
            he = next as usize;
            if mesh.half_edge_dest[he] == start_dest && !first {
                break;
            }
            first = false;
        }

        if count > 0 {
            let avg = sum / count as f32;
            forces[v] = avg - mesh.vertex_pos[v];
        }
    }

    forces
}

fn calculate_bulge_force(mesh: &HalfEdgeMesh) -> Vec<Vec3> {
    let mut forces = vec![Vec3::ZERO; mesh.next_vertex];

    for he in 0..mesh.next_half_edge {
        if mesh.half_edge_idx[he] < 0 {
            continue;
        }
        // Only boundary half-edges (face == -1)
        if mesh.half_edge_face[he] != -1 {
            continue;
        }

        let twin = mesh.half_edge_twin[he];
        if twin < 0 {
            continue;
        }
        let twin = twin as usize;

        // Edge vertices: src → dest of the boundary he
        let src_i = mesh.half_edge_dest[mesh.half_edge_twin[he] as usize];
        let dest_i = mesh.half_edge_dest[he];
        if src_i < 0 || dest_i < 0 {
            continue;
        }
        let src = src_i as usize;
        let dest = dest_i as usize;

        let edge_vec = mesh.vertex_pos[dest] - mesh.vertex_pos[src];

        // Next edge in the internal twin's face
        let twin_next = mesh.half_edge_next[twin];
        if twin_next < 0 {
            continue;
        }
        let next_dest_i = mesh.half_edge_dest[twin_next as usize];
        if next_dest_i < 0 {
            continue;
        }
        let next_dest = next_dest_i as usize;
        let next_edge_vec = mesh.vertex_pos[next_dest] - mesh.vertex_pos[src];

        // Cross products for outward normal
        let surface_normal = edge_vec.cross(next_edge_vec);
        let edge_normal = edge_vec.cross(surface_normal);
        let norm = edge_normal.length();
        if norm < EPSILON {
            continue;
        }
        let edge_normal = edge_normal / norm;

        forces[src] += edge_normal;
        forces[dest] += edge_normal;
    }

    // Normalize per-vertex
    for v in 0..mesh.next_vertex {
        let norm = forces[v].length();
        if norm > EPSILON {
            forces[v] /= norm;
        }
    }

    forces
}

// ── Position update ────────────────────────────────────────────────────────────

fn update_positions(
    mesh: &mut HalfEdgeMesh,
    spring_len: f32,
    elastic_constant: f32,
    repulsion_distance: f32,
    repulsion_strength: f32,
    bulge_strength: f32,
    planar_strength: f32,
    dt: f32,
    damping: f32,
) {
    let spring = calculate_spring_force(mesh, spring_len, elastic_constant);
    let repulsion = calculate_repulsion_force(mesh, repulsion_distance);
    let bulge = calculate_bulge_force(mesh);
    let planar = calculate_planar_force(mesh);

    for v in 0..mesh.next_vertex {
        if mesh.vertex_idx[v] < 0 {
            continue;
        }
        let total = spring[v]
            + repulsion_strength * repulsion[v]
            + bulge_strength * bulge[v]
            + planar_strength * planar[v];

        mesh.vertex_pos[v] += dt * total;
    }

    // Damping (applied to implicit velocity = position change)
    // Since we have no explicit velocity, we just move with dt*force each frame.
    // The damping factor is embedded in the dt.
    let _ = damping; // damping is implicit in dt scaling
}

// ── State evolution ────────────────────────────────────────────────────────────

fn laplacian_pass(mesh: &HalfEdgeMesh, x: &[f32]) -> Vec<f32> {
    let n = mesh.next_vertex;
    let mut avg = vec![0.0f32; n];
    let mut deg = vec![0usize; n];

    for he in 0..mesh.next_half_edge {
        if mesh.half_edge_idx[he] < 0 {
            continue;
        }
        let dest = mesh.half_edge_dest[he];
        let prev = mesh.half_edge_prev[he];
        if dest < 0 || prev < 0 {
            continue;
        }
        let src = mesh.half_edge_dest[prev as usize];
        if src < 0 {
            continue;
        }
        avg[dest as usize] += x[src as usize];
        deg[dest as usize] += 1;
    }

    // L_scaled = (I - D^{-1}A) / 2, eigenvalues in [-1, 1] for Chebyshev stability
    let mut out = vec![0.0f32; n];
    for v in 0..n {
        if mesh.vertex_idx[v] < 0 {
            continue;
        }
        let neighbor_avg = if deg[v] > 0 { avg[v] / deg[v] as f32 } else { x[v] };
        out[v] = (x[v] - neighbor_avg) * 0.5;
    }
    out
}

fn update_vertex_state(
    mesh: &mut HalfEdgeMesh,
    cheb_order: usize,
    cheb_coeffs: &[f32; 10],
    growth_mu: f32,
    growth_sigma: f32,
    dt: f32,
) {
    let n = mesh.next_vertex;
    let state: Vec<f32> = mesh.vertex_state[..n].to_vec();

    // T_0 = state
    let t_prev_init = state.clone();
    // T_1 = L_scaled * state
    let t_curr_init = laplacian_pass(mesh, &state);

    // Accumulate: result = c[0]*T_0 + c[1]*T_1
    let mut result = vec![0.0f32; n];
    for v in 0..n {
        result[v] = cheb_coeffs[0] * t_prev_init[v] + cheb_coeffs[1] * t_curr_init[v];
    }

    let mut t_prev = t_prev_init;
    let mut t_curr = t_curr_init;

    for k in 2..cheb_order {
        let l_curr = laplacian_pass(mesh, &t_curr);
        let mut t_next = vec![0.0f32; n];
        for v in 0..n {
            t_next[v] = 2.0 * l_curr[v] - t_prev[v];
        }
        for v in 0..n {
            result[v] += cheb_coeffs[k] * t_next[v];
        }
        t_prev = t_curr;
        t_curr = t_next;
    }

    // Apply growth function
    for v in 0..n {
        if mesh.vertex_idx[v] < 0 {
            continue;
        }
        mesh.vertex_u[v] = result[v];
        mesh.vertex_state[v] += dt * gaussian(result[v], growth_mu, growth_sigma);
        mesh.vertex_state[v] = mesh.vertex_state[v].clamp(0.0, 1.0);
    }
}

// ── Growth: generate new triangles ─────────────────────────────────────────────

fn generate_new_triangles(mesh: &mut HalfEdgeMesh, split_threshold: f32) {
    // Collect vertices that want to split
    let mut splits: Vec<(usize, usize, bool, bool)> = Vec::new();

    for v in 0..mesh.next_vertex {
        if mesh.vertex_idx[v] < 0 {
            continue;
        }
        if mesh.vertex_state[v] < split_threshold || random_f32() > 0.05 {
            continue;
        }

        let he_start = mesh.vertex_half_edge[v];
        if he_start < 0 {
            continue;
        }
        let he = he_start as usize;
        let dest = mesh.half_edge_dest[he];
        if dest < 0 {
            continue;
        }
        let twin = mesh.half_edge_twin[he];
        if twin < 0 {
            continue;
        }

        let src = mesh.half_edge_dest[twin as usize];
        if src < 0 {
            continue;
        }
        let src = src as usize;
        let dest = dest as usize;

        let he_on_boundary = mesh.half_edge_face[he] == -1;
        let twin_on_boundary = mesh.half_edge_face[twin as usize] == -1;

        splits.push((src, dest, he_on_boundary, twin_on_boundary));
    }

    for (src, dest, he_on_boundary, twin_on_boundary) in splits {
        if mesh.active_vertex_count() >= MAX_VERTICES - 5 {
            break;
        }

        if he_on_boundary || twin_on_boundary {
            // Boundary split
            if twin_on_boundary {
                mesh.add_internal_edge_triangle(src, dest);
            } else {
                mesh.add_internal_edge_triangle(dest, src);
            }
        } else {
            mesh.add_internal_triangles(src, dest);
        }
    }
}

// ── Model ──────────────────────────────────────────────────────────────────────

struct Model {
    window: Entity,
    mesh: Box<HalfEdgeMesh>,
    spring_len: f32,
    elastic_constant: f32,
    repulsion_distance: f32,
    repulsion_strength: f32,
    bulge_strength: f32,
    planar_strength: f32,
    dt: f32,
    damping: f32,
    // Lenia-style params
    kernel_mu: f32,
    kernel_sigma: f32,
    growth_mu: f32,
    growth_sigma: f32,
    split_threshold: f32,
    cheb_order: usize,
    cheb_coeffs: [f32; 10],
    frame: u64,
    // Camera rotation
    camera_yaw: f32,
    camera_pitch: f32,
    dragging: bool,
    last_mouse: Vec2,
}

fn recompute_cheb_coeffs(model: &mut Model) {
    let mut sum = 0.0f32;
    for k in 0..model.cheb_order {
        let c = (-((k as f32 - model.kernel_mu).powi(2)) / (2.0 * model.kernel_sigma * model.kernel_sigma)).exp();
        model.cheb_coeffs[k] = c;
        sum += c;
    }
    for k in 0..model.cheb_order {
        model.cheb_coeffs[k] /= sum;
    }
    for k in model.cheb_order..10 {
        model.cheb_coeffs[k] = 0.0;
    }
}

fn randomize_params(model: &mut Model) {
    model.spring_len = 20.0 + random_f32() * 30.0;
    model.elastic_constant = 0.05 + random_f32() * 0.15;
    model.repulsion_distance = model.spring_len * (3.0 + random_f32() * 4.0);
    model.repulsion_strength = 1.0 + random_f32() * 4.0;
    model.bulge_strength = 5.0 + random_f32() * 15.0;
    model.planar_strength = 0.05 + random_f32() * 0.15;
    model.dt = 0.05 + random_f32() * 0.15;
    model.damping = 0.5;
    // Lenia-style params
    model.kernel_mu = 1.0 + random_f32() * 4.0;
    model.kernel_sigma = 0.5 + random_f32() * 2.0;
    model.growth_mu = random_f32();
    model.growth_sigma = 0.1 + random_f32() * 0.5;
    model.split_threshold = 0.5 + random_f32() * 0.4;

    model.cheb_order = 3 + (random_f32() * 6.0) as usize; // 3–8
    recompute_cheb_coeffs(model);
}

// ── Lit mesh shader model ───────────────────────────────────────────────────────

#[shader_model(fragment = "lit_mesh.wgsl")]
struct LitMesh {
    #[uniform(0)]
    light: Vec4, // xyz = light direction, w = ambient
}

// ── App ────────────────────────────────────────────────────────────────────────

fn main() {
    nannou::app(model)
        .update(update)
        .shader_model::<LitMesh>()
        .run();
}

fn model(app: &App) -> Model {
    let cam = app.new_camera()
        .projection(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical { viewport_height: 20.0 },
            ..OrthographicProjection::default_3d()
        })
        .build();

    let w_id = app.new_window()
        .size(1000, 1000)
        .camera(cam)
        .view(view)
        .key_pressed(key_pressed)
        .mouse_pressed(mouse_pressed)
        .mouse_released(mouse_released)
        .mouse_moved(mouse_moved)
        .build();

    let mut m = Model {
        window: w_id,
        mesh: HalfEdgeMesh::new(),
        spring_len: 30.0,
        elastic_constant: 0.1,
        repulsion_distance: 150.0,
        repulsion_strength: 2.0,
        bulge_strength: 10.0,
        planar_strength: 0.1,
        dt: 0.1,
        damping: 0.5,
        kernel_mu: 2.0,
        kernel_sigma: 1.0,
        growth_mu: 0.5,
        growth_sigma: 0.3,
        split_threshold: 0.7,
        cheb_order: 5,
        cheb_coeffs: [0.2; 10],
        frame: 0,
        camera_yaw: 0.0,
        camera_pitch: 0.0,
        dragging: false,
        last_mouse: Vec2::ZERO,
    };
    randomize_params(&mut m);
    m.mesh = make_circle(m.spring_len, N_RINGS, SEGMENTS_INNER);
    m
}

fn update(app: &App, model: &mut Model) {
    // egui settings panel
    let mut egui_ctx = app.egui_for_window(model.window);
    let ctx = egui_ctx.get_mut();

    let mut kernel_changed = false;
    egui::Window::new("Settings").show(&ctx, |ui| {
        ui.label("Kernel (ring)");
        kernel_changed |= ui.add(egui::Slider::new(&mut model.kernel_mu, 0.0..=8.0).text("mu")).changed();
        kernel_changed |= ui.add(egui::Slider::new(&mut model.kernel_sigma, 0.1..=4.0).text("sigma")).changed();
        kernel_changed |= ui.add(egui::Slider::new(&mut model.cheb_order, 2..=10).text("order")).changed();
        ui.separator();
        ui.label("Growth function");
        ui.add(egui::Slider::new(&mut model.growth_mu, 0.0..=1.0).text("mu"));
        ui.add(egui::Slider::new(&mut model.growth_sigma, 0.01..=1.0).text("sigma"));
        ui.separator();
        ui.label("Split");
        ui.add(egui::Slider::new(&mut model.split_threshold, 0.0..=1.0).text("threshold"));
        ui.separator();
        ui.label("Physics");
        ui.add(egui::Slider::new(&mut model.spring_len, 5.0..=80.0).text("spring len"));
        ui.add(egui::Slider::new(&mut model.elastic_constant, 0.01..=0.5).text("elastic"));
        ui.add(egui::Slider::new(&mut model.repulsion_strength, 0.0..=10.0).text("repulsion"));
        ui.add(egui::Slider::new(&mut model.bulge_strength, 0.0..=30.0).text("bulge"));
        ui.add(egui::Slider::new(&mut model.planar_strength, 0.0..=0.5).text("planar"));
        ui.add(egui::Slider::new(&mut model.dt, 0.01..=0.3).text("dt"));
    });
    drop(egui_ctx);

    if kernel_changed {
        recompute_cheb_coeffs(model);
    }

    model.frame += 1;

    // Physics
    update_positions(
        &mut model.mesh,
        model.spring_len,
        model.elastic_constant,
        model.repulsion_distance,
        model.repulsion_strength,
        model.bulge_strength,
        model.planar_strength,
        model.dt,
        model.damping,
    );

    // State evolution
    update_vertex_state(
        &mut model.mesh,
        model.cheb_order,
        &model.cheb_coeffs,
        model.growth_mu,
        model.growth_sigma,
        model.dt * 0.1,
    );

    // Growth (every 10 frames)
    if model.frame % 10 == 0 {
        generate_new_triangles(
            &mut model.mesh,
            model.split_threshold,
        );
    }

    // Mesh refinement (every 20 frames)
    if model.frame % 20 == 0 {
        model.mesh.refine_mesh();
    }

    #[cfg(debug_assertions)]
    if model.frame % 100 == 0 {
        validate_mesh(&model.mesh);
    }
}

fn view(app: &App, model: &Model) {
    let draw = app.draw();
    draw.background().color(BLACK);

    let mesh = &model.mesh;

    // Compute bounding box of active vertices
    let mut min_pos = Vec3::splat(f32::INFINITY);
    let mut max_pos = Vec3::splat(f32::NEG_INFINITY);
    let mut any_active = false;

    for v in 0..mesh.next_vertex {
        if mesh.vertex_idx[v] < 0 {
            continue;
        }
        any_active = true;
        let p = mesh.vertex_pos[v];
        min_pos = min_pos.min(p);
        max_pos = max_pos.max(p);
    }

    if !any_active {
        return;
    }

    let center = (min_pos + max_pos) * 0.5;

    // Scale to fit bounding sphere within camera's clip range (camera at z=10, far=1000)
    let mut max_radius = 0.0f32;
    for v in 0..mesh.next_vertex {
        if mesh.vertex_idx[v] < 0 {
            continue;
        }
        let r = (mesh.vertex_pos[v] - center).length();
        max_radius = max_radius.max(r);
    }
    max_radius *= 1.15; // padding
    let scale = 8.0 / max_radius.max(1.0);

    // Barycentric coords for each vertex in a triangle
    let bary = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    // Build non-indexed triangle list with barycentric coords in color RGB, state in alpha
    let mut tris: Vec<(Vec3, Color)> = Vec::new();

    for f in 0..mesh.next_face {
        if mesh.face_idx[f] < 0 {
            continue;
        }
        let he0 = mesh.face_half_edge[f];
        if he0 < 0 {
            continue;
        }
        let he0 = he0 as usize;
        let he1 = mesh.half_edge_next[he0];
        if he1 < 0 {
            continue;
        }
        let he1 = he1 as usize;
        let he2 = mesh.half_edge_next[he1];
        if he2 < 0 {
            continue;
        }
        let he2 = he2 as usize;

        let verts = [
            mesh.half_edge_dest[he0] as usize,
            mesh.half_edge_dest[he1] as usize,
            mesh.half_edge_dest[he2] as usize,
        ];

        if verts.iter().any(|&v| mesh.vertex_idx[v] < 0) {
            continue;
        }

        for (i, &vi) in verts.iter().enumerate() {
            let p = mesh.vertex_pos[vi];
            let sp = Vec3::new(
                (p.x - center.x) * scale,
                (p.y - center.y) * scale,
                p.z * scale,
            );
            let b = bary[i];
            let color = Color::linear_rgba(b.x, b.y, b.z, mesh.vertex_state[vi]);
            tris.push((sp, color));
        }

        // Back face (reversed winding)
        for &i in &[0, 2, 1] {
            let vi = verts[i];
            let p = mesh.vertex_pos[vi];
            let sp = Vec3::new(
                (p.x - center.x) * scale,
                (p.y - center.y) * scale,
                p.z * scale,
            );
            let b = bary[i];
            let color = Color::linear_rgba(b.x, b.y, b.z, mesh.vertex_state[vi]);
            tris.push((sp, color));
        }
    }

    if !tris.is_empty() {
        let lit_draw = draw
            .y_radians(model.camera_yaw)
            .x_radians(model.camera_pitch)
            .shader_model(LitMesh {
                light: Vec4::new(0.3, 0.4, 1.0, 0.2),
            });
        lit_draw.mesh().points_colored(tris);
    }
}

fn key_pressed(app: &App, model: &mut Model, key: KeyCode) {
    if key == KeyCode::KeyS {
        app.main_window()
            .save_screenshot(app.exe_name().unwrap() + ".png");
    }
    if key == KeyCode::KeyR {
        randomize_params(model);
        model.mesh = make_circle(model.spring_len, N_RINGS, SEGMENTS_INNER);
        model.frame = 0;
        model.camera_yaw = 0.0;
        model.camera_pitch = 0.0;
    }
}

fn mouse_pressed(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Left {
        model.dragging = true;
    }
}

fn mouse_released(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Left {
        model.dragging = false;
    }
}

fn mouse_moved(_app: &App, model: &mut Model, pos: Vec2) {
    if model.dragging {
        let delta = pos - model.last_mouse;
        model.camera_yaw += delta.x * 0.005;
        model.camera_pitch += delta.y * 0.005;
    }
    model.last_mouse = pos;
}

// ── Debug validation ───────────────────────────────────────────────────────────

#[cfg(debug_assertions)]
fn validate_mesh(mesh: &HalfEdgeMesh) {
    for he in 0..mesh.next_half_edge {
        if mesh.half_edge_idx[he] < 0 {
            continue;
        }

        // next(prev(he)) == he
        let prev = mesh.half_edge_prev[he];
        if prev >= 0 {
            let next_of_prev = mesh.half_edge_next[prev as usize];
            assert_eq!(
                next_of_prev, he as i32,
                "next(prev({he})) = {next_of_prev} != {he}"
            );
        }

        // prev(next(he)) == he
        let next = mesh.half_edge_next[he];
        if next >= 0 {
            let prev_of_next = mesh.half_edge_prev[next as usize];
            assert_eq!(
                prev_of_next, he as i32,
                "prev(next({he})) = {prev_of_next} != {he}"
            );
        }

        // twin(twin(he)) == he
        let twin = mesh.half_edge_twin[he];
        if twin >= 0 {
            let twin_twin = mesh.half_edge_twin[twin as usize];
            assert_eq!(
                twin_twin, he as i32,
                "twin(twin({he})) = {twin_twin} != {he}"
            );
        }

        // Face consistency: all half-edges in a face loop should have the same face
        if next >= 0 {
            let face_he = mesh.half_edge_face[he];
            let face_next = mesh.half_edge_face[next as usize];
            assert_eq!(
                face_he, face_next,
                "face({he}) = {face_he} != face({next}) = {face_next}"
            );
        }
    }

    // face_half_edge must point to a half-edge that belongs to that face
    for f in 0..mesh.next_face {
        if mesh.face_idx[f] < 0 {
            continue;
        }
        let fhe = mesh.face_half_edge[f];
        assert!(
            fhe >= 0,
            "face {f} has face_half_edge = {fhe} (invalid)"
        );
        let fhe = fhe as usize;
        assert!(
            mesh.half_edge_idx[fhe] >= 0,
            "face {f} face_half_edge {fhe} points to inactive half-edge"
        );
        assert_eq!(
            mesh.half_edge_face[fhe], f as i32,
            "face {f} face_half_edge {fhe} belongs to face {} instead",
            mesh.half_edge_face[fhe]
        );

        // Walk the face loop and verify it has exactly 3 edges (triangle)
        let mut he = fhe;
        for step in 0..4 {
            let next = mesh.half_edge_next[he];
            assert!(
                next >= 0,
                "face {f} has broken next chain at half-edge {he} (step {step})"
            );
            he = next as usize;
            if he == fhe {
                assert_eq!(
                    step, 2,
                    "face {f} loop has {} edges instead of 3",
                    step + 1
                );
                break;
            }
            assert!(
                step < 3,
                "face {f} loop did not close after 3 steps"
            );
        }
    }
}
