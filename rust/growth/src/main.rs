#![allow(dead_code)]

mod mesh_builders;

use std::sync::{Arc, Mutex};

use nannou::prelude::*;
use bevy::input::mouse::MouseWheel;
use bytemuck::{Pod, Zeroable};

use mesh_builders::{StartShape, make_start_mesh};

// ── Constants ──────────────────────────────────────────────────────────────────

pub(crate) const MAX_VERTICES: usize = 8000;
pub(crate) const MAX_HALF_EDGES: usize = MAX_VERTICES * 6;
pub(crate) const MAX_FACES: usize = MAX_VERTICES * 2;
const EPSILON: f32 = 1e-6;
pub(crate) const N_RINGS: usize = 24;
pub(crate) const SEGMENTS_INNER: usize = 6;
const MAX_BINS_PER_DIM: u32 = 256;
const WORKGROUP_SIZE: u32 = 64;

// ── Half-edge mesh (structure-of-arrays, heap-allocated) ───────────────────────

#[derive(Clone)]
pub(crate) struct HalfEdgeMesh {
    // Face arrays
    pub(crate) face_idx: [i32; MAX_FACES],
    pub(crate) face_half_edge: [i32; MAX_FACES],

    // Vertex arrays
    pub(crate) vertex_idx: [i32; MAX_VERTICES],
    pub(crate) vertex_half_edge: [i32; MAX_VERTICES],
    pub(crate) vertex_pos: [Vec3; MAX_VERTICES],
    pub(crate) vertex_state: [f32; MAX_VERTICES],
    pub(crate) vertex_u: [f32; MAX_VERTICES],

    // Half-edge arrays
    pub(crate) half_edge_idx: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_twin: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_dest: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_face: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_next: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_prev: [i32; MAX_HALF_EDGES],

    // Allocation watermarks
    pub(crate) next_vertex: usize,
    pub(crate) next_face: usize,
    pub(crate) next_half_edge: usize,
}

impl HalfEdgeMesh {
    pub(crate) fn new() -> Box<Self> {
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

    pub(crate) fn alloc_vertex(&mut self) -> Option<usize> {
        if self.next_vertex >= MAX_VERTICES {
            return None;
        }
        let idx = self.next_vertex;
        self.next_vertex += 1;
        self.vertex_idx[idx] = idx as i32;
        Some(idx)
    }

    pub(crate) fn alloc_face(&mut self) -> Option<usize> {
        if self.next_face >= MAX_FACES {
            return None;
        }
        let idx = self.next_face;
        self.next_face += 1;
        self.face_idx[idx] = idx as i32;
        Some(idx)
    }

    pub(crate) fn alloc_half_edge(&mut self) -> Option<usize> {
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

// ── Growth: generate new triangles ─────────────────────────────────────────────

fn generate_new_triangles(mesh: &mut HalfEdgeMesh, split_threshold: f32, split_chance: f32) {
    // Collect vertices that want to split
    let mut splits: Vec<(usize, usize, bool, bool)> = Vec::new();

    for v in 0..mesh.next_vertex {
        if mesh.vertex_idx[v] < 0 {
            continue;
        }
        if mesh.vertex_state[v] < split_threshold || random_f32() > split_chance {
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

// ── GPU Compute ─────────────────────────────────────────────────────────────────

// Shader sources (loaded at compile time, common prepended)
const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");
const FILL_BINS_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/fill_bins.wgsl"));
const PREFIX_SUM_WGSL: &str = include_str!("shaders/prefix_sum.wgsl");
const SORT_VERTICES_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/sort_vertices.wgsl"));
const REPULSION_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/repulsion.wgsl"));
const TOPO_FORCES_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/topo_forces.wgsl"));
const INTEGRATE_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/integrate.wgsl"));
const CHEBYSHEV_INIT_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/chebyshev_init.wgsl"));
const CHEBYSHEV_STEP_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/chebyshev_step.wgsl"));
const GROWTH_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/growth.wgsl"));
const BBOX_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/bbox.wgsl"));
const MESH_RENDER_WGSL: &str = include_str!("shaders/mesh_render.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuSimParams {
    num_vertices: u32,
    num_half_edges: u32,
    repulsion_distance: f32,
    spring_len: f32,
    elastic_constant: f32,
    bulge_strength: f32,
    planar_strength: f32,
    dt: f32,
    origin_x: f32,
    origin_y: f32,
    bin_size: f32,
    num_bins_x: u32,
    num_bins_y: u32,
    growth_mu: f32,
    growth_sigma: f32,
    cheb_order: u32,
    repulsion_strength: f32,
    state_dt: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RenderUniforms {
    view_proj: [[f32; 4]; 4],
    center: [f32; 4],       // xyz = mesh center, w = scale
    light: [f32; 4],        // xyz = direction, w = ambient
    render_mode: [f32; 4],  // x = mode, yzw unused
}

fn dispatch_count(n: u32) -> u32 {
    (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

#[derive(Clone)]
struct GpuCompute {
    // Vertex buffers
    vertex_pos_buf: wgpu::Buffer,
    vertex_force_buf: wgpu::Buffer,

    // Spatial hash buffers
    bin_size_buf: wgpu::Buffer,
    bin_offset_buf: wgpu::Buffer,
    bin_offset_tmp_buf: wgpu::Buffer,
    sorted_idx_buf: wgpu::Buffer,
    prefix_sum_step_buf: wgpu::Buffer,

    // Topology buffers (packed: vec4<i32> = dest, twin, next, face)
    he_packed_buf: wgpu::Buffer,
    vertex_he_buf: wgpu::Buffer,

    // State evolution buffers
    vertex_state_buf: wgpu::Buffer,
    vertex_u_buf: wgpu::Buffer,
    cheb_a_buf: wgpu::Buffer,
    cheb_b_buf: wgpu::Buffer,
    cheb_c_buf: wgpu::Buffer,
    cheb_result_buf: wgpu::Buffer,
    cheb_coeff_buf: wgpu::Buffer,  // single f32 uniform
    cheb_c0_buf: wgpu::Buffer,    // uniform for init
    cheb_c1_buf: wgpu::Buffer,    // uniform for init

    // Params
    sim_params_buf: wgpu::Buffer,

    // Readback
    pos_readback_buf: wgpu::Buffer,
    state_readback_buf: wgpu::Buffer,

    // Bounding box reduction
    bbox_atomic_buf: wgpu::Buffer,
    bbox_readback_buf: wgpu::Buffer,
    bbox_clear_pipeline: wgpu::ComputePipeline,
    bbox_reduce_pipeline: wgpu::ComputePipeline,
    bbox_data_bg: wgpu::BindGroup,
    bbox_params_bg: wgpu::BindGroup,

    // Pipelines
    clear_bins_pipeline: wgpu::ComputePipeline,
    fill_bins_pipeline: wgpu::ComputePipeline,
    prefix_sum_pipeline: wgpu::ComputePipeline,
    sort_clear_pipeline: wgpu::ComputePipeline,
    sort_vertices_pipeline: wgpu::ComputePipeline,
    repulsion_pipeline: wgpu::ComputePipeline,
    topo_forces_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,
    chebyshev_init_pipeline: wgpu::ComputePipeline,
    chebyshev_step_pipeline: wgpu::ComputePipeline,
    growth_pipeline: wgpu::ComputePipeline,

    // Bind groups - spatial hash
    fill_bins_bg: [wgpu::BindGroup; 3],  // [pos, params, bins]
    prefix_sum_bgs: [wgpu::BindGroup; 3], // ping-pong: [0]bin_size→offset, [1]offset→tmp, [2]tmp→offset
    sort_data_bg: wgpu::BindGroup,
    sort_params_bg: wgpu::BindGroup,

    // Bind groups - forces
    repulsion_data_bg: wgpu::BindGroup,
    repulsion_params_bg: wgpu::BindGroup,
    topo_data_bg: wgpu::BindGroup,
    topo_params_bg: wgpu::BindGroup,
    integrate_data_bg: wgpu::BindGroup,
    integrate_params_bg: wgpu::BindGroup,

    // Bind groups - chebyshev
    cheb_init_data_bg: wgpu::BindGroup,
    cheb_init_params_bg: wgpu::BindGroup,
    cheb_step_data_bgs: [wgpu::BindGroup; 3], // rotating A→B→C
    cheb_step_params_bg: wgpu::BindGroup,
    growth_data_bg: wgpu::BindGroup,
    growth_params_bg: wgpu::BindGroup,

    // Render buffers (read by vertex shader directly)
    render_index_buf: wgpu::Buffer,
    render_uniform_buf: wgpu::Buffer,
    num_render_tris: u32,

    // Config
    max_bins: u32,
    topology_dirty: bool,
}

fn create_gpu_compute(device: &wgpu::Device) -> GpuCompute {
    let pos_buf_size = (MAX_VERTICES * 16) as u64;   // vec4<f32>
    let force_buf_size = pos_buf_size;
    let max_bins = MAX_BINS_PER_DIM * MAX_BINS_PER_DIM;
    let bin_buf_size = ((max_bins + 1) * 4) as u64;   // u32 per bin + 1
    let idx_buf_size = (MAX_VERTICES * 4) as u64;      // u32 per vertex
    let he_packed_size = (MAX_HALF_EDGES * 16) as u64; // vec4<i32>
    let state_buf_size = (MAX_VERTICES * 4) as u64;    // f32 per vertex

    let storage_rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let storage_r = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;

    // Create buffers
    let vertex_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_pos"), size: pos_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let vertex_force_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_force"), size: force_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let bin_size_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bin_size"), size: bin_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let bin_offset_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bin_offset"), size: bin_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let bin_offset_tmp_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bin_offset_tmp"), size: bin_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let sorted_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sorted_idx"), size: idx_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let prefix_sum_step_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prefix_sum_step"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let he_packed_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("he_packed"), size: he_packed_size, usage: storage_r, mapped_at_creation: false,
    });
    let vertex_he_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_he"), size: idx_buf_size, usage: storage_r, mapped_at_creation: false,
    });
    let vertex_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_state"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let vertex_u_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_u"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let cheb_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_a"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let cheb_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_b"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let cheb_c_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_c"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let cheb_result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_result"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let cheb_coeff_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_coeff"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let cheb_c0_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_c0"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let cheb_c1_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_c1"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim_params"), size: std::mem::size_of::<GpuSimParams>() as u64, usage: uniform, mapped_at_creation: false,
    });
    // Render buffers: index buffer (3 u32 per face) + uniform buffer
    let render_index_buf_size = (MAX_FACES * 3 * 4) as u64;
    let render_index_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_index"), size: render_index_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let render_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_uniform"), size: std::mem::size_of::<RenderUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pos_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pos_readback"), size: pos_buf_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let state_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("state_readback"), size: state_buf_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let bbox_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bbox_atomic"), size: 16, usage: storage_rw, mapped_at_creation: false,
    });
    let bbox_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bbox_readback"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });

    // Create shader modules
    let fill_bins_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fill_bins"), source: wgpu::ShaderSource::Wgsl(FILL_BINS_WGSL.into()),
    });
    let prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("prefix_sum"), source: wgpu::ShaderSource::Wgsl(PREFIX_SUM_WGSL.into()),
    });
    let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sort_vertices"), source: wgpu::ShaderSource::Wgsl(SORT_VERTICES_WGSL.into()),
    });
    let repulsion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("repulsion"), source: wgpu::ShaderSource::Wgsl(REPULSION_WGSL.into()),
    });
    let topo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("topo_forces"), source: wgpu::ShaderSource::Wgsl(TOPO_FORCES_WGSL.into()),
    });
    let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("integrate"), source: wgpu::ShaderSource::Wgsl(INTEGRATE_WGSL.into()),
    });
    let cheb_init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cheb_init"), source: wgpu::ShaderSource::Wgsl(CHEBYSHEV_INIT_WGSL.into()),
    });
    let cheb_step_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cheb_step"), source: wgpu::ShaderSource::Wgsl(CHEBYSHEV_STEP_WGSL.into()),
    });
    let growth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("growth"), source: wgpu::ShaderSource::Wgsl(GROWTH_WGSL.into()),
    });
    let bbox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bbox"), source: wgpu::ShaderSource::Wgsl(BBOX_WGSL.into()),
    });

    let cs = wgpu::ShaderStages::COMPUTE;

    // ── Bind group layouts ──────────────────────────────────────────────────

    // fill_bins: group0=pos(R), group1=params(U), group2=bins(RW)
    let fill_bins_pos_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)
        .build(device);
    let params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)
        .build(device);
    let bins_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)
        .build(device);

    // prefix_sum: source(R), dest(RW), step(U)
    let prefix_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)
        .storage_buffer(cs, false, false)
        .uniform_buffer(cs, false)
        .build(device);

    // sort: group0=pos(R)+sorted(RW)+offset(R)+counter(RW), group1=params(U)
    let sort_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_pos
        .storage_buffer(cs, false, false)  // sorted_idx
        .storage_buffer(cs, false, true)   // bin_offset
        .storage_buffer(cs, false, false)  // bin_counter (reuse bin_size)
        .build(device);

    // repulsion: group0=pos(R)+force(RW)+sorted(R)+offset(R), group1=params(U)
    let repulsion_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_pos
        .storage_buffer(cs, false, false)  // vertex_force
        .storage_buffer(cs, false, true)   // sorted_idx
        .storage_buffer(cs, false, true)   // bin_offset
        .build(device);

    // topo_forces: group0=pos(R)+force(RW)+he_packed(R)+vertex_he(R), group1=params(U)
    let topo_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_pos
        .storage_buffer(cs, false, false)  // vertex_force
        .storage_buffer(cs, false, true)   // he_packed
        .storage_buffer(cs, false, true)   // vertex_he
        .build(device);

    // integrate: group0=pos(RW)+force(R), group1=params(U)
    let integrate_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // vertex_pos (RW)
        .storage_buffer(cs, false, true)   // vertex_force (R)
        .build(device);

    // chebyshev_init: group0=state(R)+t_a(RW)+t_b(RW)+result(RW)+he_packed(R)+vertex_he(R)
    //                 group1=params(U)+c0(U)+c1(U)
    let cheb_init_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_state
        .storage_buffer(cs, false, false)  // t_a
        .storage_buffer(cs, false, false)  // t_b
        .storage_buffer(cs, false, false)  // result
        .storage_buffer(cs, false, true)   // he_packed
        .storage_buffer(cs, false, true)   // vertex_he
        .build(device);
    let cheb_init_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)  // params
        .uniform_buffer(cs, false)  // c0
        .uniform_buffer(cs, false)  // c1
        .build(device);

    // chebyshev_step: group0=t_curr(R)+t_prev(R)+t_next(RW)+result(RW)+he_packed(R)+vertex_he(R)
    //                 group1=params(U)+coeff(U)
    let cheb_step_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // t_curr
        .storage_buffer(cs, false, true)   // t_prev
        .storage_buffer(cs, false, false)  // t_next
        .storage_buffer(cs, false, false)  // result
        .storage_buffer(cs, false, true)   // he_packed
        .storage_buffer(cs, false, true)   // vertex_he
        .build(device);
    let cheb_step_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)  // params
        .uniform_buffer(cs, false)  // coeff
        .build(device);

    // growth: group0=state(RW)+u(RW)+result(R)+pos(R), group1=params(U)
    let growth_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // vertex_state
        .storage_buffer(cs, false, false)  // vertex_u
        .storage_buffer(cs, false, true)   // result
        .storage_buffer(cs, false, true)   // vertex_pos (active check)
        .build(device);

    // bbox: group0=pos(R)+bbox_atomic(RW), group1=params(U)
    let bbox_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_pos
        .storage_buffer(cs, false, false)  // bbox_atomic
        .build(device);

    // ── Pipeline layouts ────────────────────────────────────────────────────

    let fill_bins_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fill_bins_pl"),
        bind_group_layouts: &[&fill_bins_pos_layout, &params_layout, &bins_layout],
        push_constant_ranges: &[],
    });
    let prefix_sum_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("prefix_sum_pl"),
        bind_group_layouts: &[&prefix_layout],
        push_constant_ranges: &[],
    });
    let sort_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sort_pl"),
        bind_group_layouts: &[&sort_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let repulsion_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("repulsion_pl"),
        bind_group_layouts: &[&repulsion_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let topo_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("topo_pl"),
        bind_group_layouts: &[&topo_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let integrate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("integrate_pl"),
        bind_group_layouts: &[&integrate_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let cheb_init_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cheb_init_pl"),
        bind_group_layouts: &[&cheb_init_data_layout, &cheb_init_params_layout],
        push_constant_ranges: &[],
    });
    let cheb_step_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cheb_step_pl"),
        bind_group_layouts: &[&cheb_step_data_layout, &cheb_step_params_layout],
        push_constant_ranges: &[],
    });
    let growth_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("growth_pl"),
        bind_group_layouts: &[&growth_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let bbox_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bbox_pl"),
        bind_group_layouts: &[&bbox_data_layout, &params_layout],
        push_constant_ranges: &[],
    });

    // ── Compute pipelines ───────────────────────────────────────────────────

    let make_pipeline = |label, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    let clear_bins_pipeline = make_pipeline("clear_bins", &fill_bins_pl, &fill_bins_shader, "clear_bins");
    let fill_bins_pipeline = make_pipeline("fill_bins", &fill_bins_pl, &fill_bins_shader, "fill_bins");
    let prefix_sum_pipeline = make_pipeline("prefix_sum", &prefix_sum_pl, &prefix_sum_shader, "main");
    let sort_clear_pipeline = make_pipeline("sort_clear", &sort_pl, &sort_shader, "clear_counters");
    let sort_vertices_pipeline = make_pipeline("sort_vertices", &sort_pl, &sort_shader, "sort_vertices");
    let repulsion_pipeline = make_pipeline("repulsion", &repulsion_pl, &repulsion_shader, "main");
    let topo_forces_pipeline = make_pipeline("topo_forces", &topo_pl, &topo_shader, "main");
    let integrate_pipeline = make_pipeline("integrate", &integrate_pl, &integrate_shader, "main");
    let chebyshev_init_pipeline = make_pipeline("cheb_init", &cheb_init_pl, &cheb_init_shader, "main");
    let chebyshev_step_pipeline = make_pipeline("cheb_step", &cheb_step_pl, &cheb_step_shader, "main");
    let growth_pipeline = make_pipeline("growth", &growth_pl, &growth_shader, "main");
    let bbox_clear_pipeline = make_pipeline("bbox_clear", &bbox_pl, &bbox_shader, "bbox_clear");
    let bbox_reduce_pipeline = make_pipeline("bbox_reduce", &bbox_pl, &bbox_shader, "bbox_reduce");

    // ── Bind groups ─────────────────────────────────────────────────────────

    // fill_bins bind groups
    let fill_bins_bg = [
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&vertex_pos_buf, 0, None)
            .build(device, &fill_bins_pos_layout),
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&sim_params_buf, 0, None)
            .build(device, &params_layout),
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&bin_size_buf, 0, None)
            .build(device, &bins_layout),
    ];

    // prefix sum bind groups (ping-pong)
    let prefix_sum_bgs = [
        // [0] bin_size → bin_offset
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&bin_size_buf, 0, None)
            .buffer_bytes(&bin_offset_buf, 0, None)
            .buffer_bytes(&prefix_sum_step_buf, 0, None)
            .build(device, &prefix_layout),
        // [1] bin_offset → bin_offset_tmp
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&bin_offset_buf, 0, None)
            .buffer_bytes(&bin_offset_tmp_buf, 0, None)
            .buffer_bytes(&prefix_sum_step_buf, 0, None)
            .build(device, &prefix_layout),
        // [2] bin_offset_tmp → bin_offset
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&bin_offset_tmp_buf, 0, None)
            .buffer_bytes(&bin_offset_buf, 0, None)
            .buffer_bytes(&prefix_sum_step_buf, 0, None)
            .build(device, &prefix_layout),
    ];

    // sort bind groups
    let sort_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&sorted_idx_buf, 0, None)
        .buffer_bytes(&bin_offset_buf, 0, None)
        .buffer_bytes(&bin_size_buf, 0, None) // reuse as counter
        .build(device, &sort_data_layout);
    let sort_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // repulsion bind groups
    let repulsion_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&vertex_force_buf, 0, None)
        .buffer_bytes(&sorted_idx_buf, 0, None)
        .buffer_bytes(&bin_offset_buf, 0, None)
        .build(device, &repulsion_data_layout);
    let repulsion_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // topo forces bind groups
    let topo_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&vertex_force_buf, 0, None)
        .buffer_bytes(&he_packed_buf, 0, None)
        .buffer_bytes(&vertex_he_buf, 0, None)
        .build(device, &topo_data_layout);
    let topo_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // integrate bind groups
    let integrate_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&vertex_force_buf, 0, None)
        .build(device, &integrate_data_layout);
    let integrate_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // chebyshev init bind groups
    let cheb_init_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_state_buf, 0, None)
        .buffer_bytes(&cheb_a_buf, 0, None)
        .buffer_bytes(&cheb_b_buf, 0, None)
        .buffer_bytes(&cheb_result_buf, 0, None)
        .buffer_bytes(&he_packed_buf, 0, None)
        .buffer_bytes(&vertex_he_buf, 0, None)
        .build(device, &cheb_init_data_layout);
    let cheb_init_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .buffer_bytes(&cheb_c0_buf, 0, None)
        .buffer_bytes(&cheb_c1_buf, 0, None)
        .build(device, &cheb_init_params_layout);

    // chebyshev step bind groups (3 rotations: curr→prev→next)
    // After init: t_prev=A (T_0), t_curr=B (T_1)
    // k=2: curr=B, prev=A, next=C → after: prev=B, curr=C
    // k=3: curr=C, prev=B, next=A → after: prev=C, curr=A
    // k=4: curr=A, prev=C, next=B → after: prev=A, curr=B
    let cheb_step_data_bgs = [
        // [0]: curr=B, prev=A, next=C
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_result_buf, 0, None)
            .buffer_bytes(&he_packed_buf, 0, None)
            .buffer_bytes(&vertex_he_buf, 0, None)
            .build(device, &cheb_step_data_layout),
        // [1]: curr=C, prev=B, next=A
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_result_buf, 0, None)
            .buffer_bytes(&he_packed_buf, 0, None)
            .buffer_bytes(&vertex_he_buf, 0, None)
            .build(device, &cheb_step_data_layout),
        // [2]: curr=A, prev=C, next=B
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_result_buf, 0, None)
            .buffer_bytes(&he_packed_buf, 0, None)
            .buffer_bytes(&vertex_he_buf, 0, None)
            .build(device, &cheb_step_data_layout),
    ];
    let cheb_step_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .buffer_bytes(&cheb_coeff_buf, 0, None)
        .build(device, &cheb_step_params_layout);

    // growth bind groups
    let growth_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_state_buf, 0, None)
        .buffer_bytes(&vertex_u_buf, 0, None)
        .buffer_bytes(&cheb_result_buf, 0, None)
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .build(device, &growth_data_layout);
    let growth_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // bbox bind groups
    let bbox_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&bbox_atomic_buf, 0, None)
        .build(device, &bbox_data_layout);
    let bbox_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    GpuCompute {
        vertex_pos_buf, vertex_force_buf,
        bin_size_buf, bin_offset_buf, bin_offset_tmp_buf, sorted_idx_buf, prefix_sum_step_buf,
        he_packed_buf, vertex_he_buf,
        vertex_state_buf, vertex_u_buf,
        cheb_a_buf, cheb_b_buf, cheb_c_buf, cheb_result_buf,
        cheb_coeff_buf, cheb_c0_buf, cheb_c1_buf,
        sim_params_buf,
        pos_readback_buf, state_readback_buf,
        bbox_atomic_buf, bbox_readback_buf,
        bbox_clear_pipeline, bbox_reduce_pipeline,
        bbox_data_bg, bbox_params_bg,
        clear_bins_pipeline, fill_bins_pipeline, prefix_sum_pipeline,
        sort_clear_pipeline, sort_vertices_pipeline,
        repulsion_pipeline, topo_forces_pipeline, integrate_pipeline,
        chebyshev_init_pipeline, chebyshev_step_pipeline, growth_pipeline,
        fill_bins_bg, prefix_sum_bgs,
        sort_data_bg, sort_params_bg,
        repulsion_data_bg, repulsion_params_bg,
        topo_data_bg, topo_params_bg,
        integrate_data_bg, integrate_params_bg,
        cheb_init_data_bg, cheb_init_params_bg,
        cheb_step_data_bgs, cheb_step_params_bg,
        growth_data_bg, growth_params_bg,
        render_index_buf, render_uniform_buf,
        num_render_tris: 0,
        max_bins,
        topology_dirty: true,
    }
}

fn upload_mesh_to_gpu(queue: &wgpu::Queue, gpu: &GpuCompute, mesh: &HalfEdgeMesh) {
    // Upload vertex positions (vec4: xyz + active flag in w)
    let mut pos_data = vec![[0.0f32; 4]; MAX_VERTICES];
    for v in 0..mesh.next_vertex {
        let active = if mesh.vertex_idx[v] >= 0 { 1.0f32 } else { -1.0 };
        pos_data[v] = [mesh.vertex_pos[v].x, mesh.vertex_pos[v].y, mesh.vertex_pos[v].z, active];
    }
    queue.write_buffer(&gpu.vertex_pos_buf, 0, bytemuck::cast_slice(&pos_data));

    // Upload vertex states
    queue.write_buffer(&gpu.vertex_state_buf, 0, bytemuck::cast_slice(&mesh.vertex_state[..MAX_VERTICES]));

    // Upload packed half-edge topology (vec4<i32>: dest, twin, next, face)
    let mut he_packed = vec![[0i32; 4]; MAX_HALF_EDGES];
    for he in 0..mesh.next_half_edge {
        he_packed[he] = [
            mesh.half_edge_dest[he],
            mesh.half_edge_twin[he],
            mesh.half_edge_next[he],
            mesh.half_edge_face[he],
        ];
    }
    queue.write_buffer(&gpu.he_packed_buf, 0, bytemuck::cast_slice(&he_packed));

    // Upload vertex half-edge indices
    queue.write_buffer(&gpu.vertex_he_buf, 0, bytemuck::cast_slice(&mesh.vertex_half_edge[..MAX_VERTICES]));
}

fn sortable_to_float(s: u32) -> f32 {
    let mask = if (s & 0x80000000) == 0 { 0xFFFFFFFFu32 } else { 0x80000000u32 };
    f32::from_bits(s ^ mask)
}

/// Read back positions, states, and GPU-computed bounding box.
/// Returns [min_x, min_y, max_x, max_y] from the GPU bbox reduction.
fn readback_from_gpu(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuCompute, mesh: &mut HalfEdgeMesh) -> [f32; 4] {
    let n = mesh.next_vertex;
    let pos_size = (MAX_VERTICES * 16) as u64;
    let state_size = (MAX_VERTICES * 4) as u64;

    // Copy GPU buffers to staging (bbox was already copied in gpu_dispatch_frame)
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(&gpu.vertex_pos_buf, 0, &gpu.pos_readback_buf, 0, pos_size);
    encoder.copy_buffer_to_buffer(&gpu.vertex_state_buf, 0, &gpu.state_readback_buf, 0, state_size);
    queue.submit(Some(encoder.finish()));

    // Map all readback buffers, then poll once
    let pos_slice = gpu.pos_readback_buf.slice(..);
    pos_slice.map_async(wgpu::MapMode::Read, |_| {});
    let state_slice = gpu.state_readback_buf.slice(..);
    state_slice.map_async(wgpu::MapMode::Read, |_| {});
    let bbox_slice = gpu.bbox_readback_buf.slice(..);
    bbox_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait).unwrap();

    // Read positions
    {
        let data = pos_slice.get_mapped_range();
        let floats: &[[f32; 4]] = bytemuck::cast_slice(&data);
        for v in 0..n {
            mesh.vertex_pos[v] = Vec3::new(floats[v][0], floats[v][1], floats[v][2]);
        }
    }
    gpu.pos_readback_buf.unmap();

    // Read states
    {
        let data = state_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        mesh.vertex_state[..n].copy_from_slice(&floats[..n]);
    }
    gpu.state_readback_buf.unmap();

    // Read bounding box (sortable uint → float conversion)
    let bbox = {
        let data = bbox_slice.get_mapped_range();
        let uints: &[u32] = bytemuck::cast_slice(&data);
        [
            sortable_to_float(uints[0]),
            sortable_to_float(uints[1]),
            sortable_to_float(uints[2]),
            sortable_to_float(uints[3]),
        ]
    };
    gpu.bbox_readback_buf.unmap();

    bbox
}

fn rebuild_render_indices(queue: &wgpu::Queue, gpu: &mut GpuCompute, mesh: &HalfEdgeMesh) {
    let mut indices: Vec<u32> = Vec::with_capacity(mesh.next_face * 3);
    for f in 0..mesh.next_face {
        if mesh.face_idx[f] < 0 { continue; }
        let he0 = mesh.face_half_edge[f];
        if he0 < 0 { continue; }
        let he0 = he0 as usize;
        let he1 = mesh.half_edge_next[he0];
        if he1 < 0 { continue; }
        let he1 = he1 as usize;
        let he2 = mesh.half_edge_next[he1];
        if he2 < 0 { continue; }
        let he2 = he2 as usize;

        let v0 = mesh.half_edge_dest[he0];
        let v1 = mesh.half_edge_dest[he1];
        let v2 = mesh.half_edge_dest[he2];
        if v0 < 0 || v1 < 0 || v2 < 0 { continue; }
        if mesh.vertex_idx[v0 as usize] < 0
            || mesh.vertex_idx[v1 as usize] < 0
            || mesh.vertex_idx[v2 as usize] < 0
        {
            continue;
        }
        indices.push(v0 as u32);
        indices.push(v1 as u32);
        indices.push(v2 as u32);
    }
    gpu.num_render_tris = (indices.len() / 3) as u32;
    queue.write_buffer(&gpu.render_index_buf, 0, bytemuck::cast_slice(&indices));
}

fn update_render_uniforms(
    queue: &wgpu::Queue,
    gpu: &GpuCompute,
    center: Vec3,
    scale: f32,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    render_mode: u32,
    aspect: f32,
) {
    let rot = Mat4::from_rotation_x(pitch) * Mat4::from_rotation_y(yaw);
    let half_h = 10.0 / zoom;
    let half_w = half_h * aspect;
    let proj = Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, -1000.0, 1000.0);
    let view_proj = proj * rot;

    let uniforms = RenderUniforms {
        view_proj: view_proj.to_cols_array_2d(),
        center: [center.x, center.y, center.z, scale],
        light: [0.3, 0.4, 1.0, 0.2],
        render_mode: [render_mode as f32, 0.0, 0.0, 0.0],
    };
    queue.write_buffer(&gpu.render_uniform_buf, 0, bytemuck::bytes_of(&uniforms));
}

fn gpu_dispatch_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &mut GpuCompute,
    mesh: &HalfEdgeMesh,
    params: &GpuSimParams,
    cheb_order: usize,
    cheb_coeffs: &[f32; 20],
) {
    let n = params.num_vertices;
    let num_bins_total = params.num_bins_x * params.num_bins_y + 1;

    // Upload sim params
    queue.write_buffer(&gpu.sim_params_buf, 0, bytemuck::bytes_of(params));

    // Upload positions (positions may have changed from CPU topology ops)
    if gpu.topology_dirty {
        upload_mesh_to_gpu(queue, gpu, mesh);
        gpu.topology_dirty = false;
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute_encoder"),
    });

    // ── Spatial hash (5 passes) ─────────────────────────────────────────

    // 1. Clear bin sizes
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clear_bins"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.clear_bins_pipeline);
        pass.set_bind_group(0, &gpu.fill_bins_bg[0], &[]);
        pass.set_bind_group(1, &gpu.fill_bins_bg[1], &[]);
        pass.set_bind_group(2, &gpu.fill_bins_bg[2], &[]);
        pass.dispatch_workgroups(dispatch_count(num_bins_total), 1, 1);
    }

    // 2. Fill bin sizes
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fill_bins"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.fill_bins_pipeline);
        pass.set_bind_group(0, &gpu.fill_bins_bg[0], &[]);
        pass.set_bind_group(1, &gpu.fill_bins_bg[1], &[]);
        pass.set_bind_group(2, &gpu.fill_bins_bg[2], &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // 3. Prefix sum
    let num_prefix_steps = (num_bins_total as f32).log2().ceil() as u32;
    for i in 0..num_prefix_steps {
        let step_size = 1u32 << i;
        queue.write_buffer(&gpu.prefix_sum_step_buf, 0, bytemuck::bytes_of(&step_size));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum"), timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.prefix_sum_pipeline);
            let bg_idx = if i == 0 { 0 } else if i % 2 == 1 { 1 } else { 2 };
            pass.set_bind_group(0, &gpu.prefix_sum_bgs[bg_idx], &[]);
            pass.dispatch_workgroups(dispatch_count(num_bins_total), 1, 1);
        }
    }
    // If even number of steps > 1, result is in tmp; copy back
    if num_prefix_steps > 1 && num_prefix_steps % 2 == 0 {
        encoder.copy_buffer_to_buffer(
            &gpu.bin_offset_tmp_buf, 0,
            &gpu.bin_offset_buf, 0,
            (num_bins_total * 4) as u64,
        );
    }

    // 4. Clear bin counters for sort
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_clear"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.sort_clear_pipeline);
        pass.set_bind_group(0, &gpu.sort_data_bg, &[]);
        pass.set_bind_group(1, &gpu.sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(num_bins_total), 1, 1);
    }

    // 5. Sort vertices by bin
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_vertices"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.sort_vertices_pipeline);
        pass.set_bind_group(0, &gpu.sort_data_bg, &[]);
        pass.set_bind_group(1, &gpu.sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // ── Forces ──────────────────────────────────────────────────────────

    // Repulsion (initializes vertex_force_buf)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("repulsion"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.repulsion_pipeline);
        pass.set_bind_group(0, &gpu.repulsion_data_bg, &[]);
        pass.set_bind_group(1, &gpu.repulsion_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Topology forces (spring + planar + bulge, adds to vertex_force_buf)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("topo_forces"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.topo_forces_pipeline);
        pass.set_bind_group(0, &gpu.topo_data_bg, &[]);
        pass.set_bind_group(1, &gpu.topo_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Integrate positions
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("integrate"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.integrate_pipeline);
        pass.set_bind_group(0, &gpu.integrate_data_bg, &[]);
        pass.set_bind_group(1, &gpu.integrate_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // ── State evolution (Chebyshev) ─────────────────────────────────────

    // Upload chebyshev coefficients
    queue.write_buffer(&gpu.cheb_c0_buf, 0, bytemuck::bytes_of(&cheb_coeffs[0]));
    queue.write_buffer(&gpu.cheb_c1_buf, 0, bytemuck::bytes_of(&cheb_coeffs[1]));

    // Init: T_0 = state → cheb_a, T_1 = L(state) → cheb_b, result = c0*T0 + c1*T1
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cheb_init"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.chebyshev_init_pipeline);
        pass.set_bind_group(0, &gpu.cheb_init_data_bg, &[]);
        pass.set_bind_group(1, &gpu.cheb_init_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Chebyshev steps k=2..cheb_order
    for k in 2..cheb_order {
        queue.write_buffer(&gpu.cheb_coeff_buf, 0, bytemuck::bytes_of(&cheb_coeffs[k]));
        let bg_idx = (k - 2) % 3;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cheb_step"), timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.chebyshev_step_pipeline);
            pass.set_bind_group(0, &gpu.cheb_step_data_bgs[bg_idx], &[]);
            pass.set_bind_group(1, &gpu.cheb_step_params_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(n), 1, 1);
        }
    }

    // Growth function
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("growth"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.growth_pipeline);
        pass.set_bind_group(0, &gpu.growth_data_bg, &[]);
        pass.set_bind_group(1, &gpu.growth_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // ── Bounding box reduction (for next frame's spatial hash) ───────────

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bbox_clear"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.bbox_clear_pipeline);
        pass.set_bind_group(0, &gpu.bbox_data_bg, &[]);
        pass.set_bind_group(1, &gpu.bbox_params_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bbox_reduce"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.bbox_reduce_pipeline);
        pass.set_bind_group(0, &gpu.bbox_data_bg, &[]);
        pass.set_bind_group(1, &gpu.bbox_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Copy bbox atomics to readback staging buffer
    encoder.copy_buffer_to_buffer(&gpu.bbox_atomic_buf, 0, &gpu.bbox_readback_buf, 0, 16);

    queue.submit(Some(encoder.finish()));
}

// ── Render state (shared between main + render worlds via Arc) ─────────────────

struct RenderState {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    bind_group: Option<wgpu::BindGroup>,
    depth_view: Option<wgpu::TextureViewHandle>,
    depth_size: [u32; 2],
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            pipeline: None, bind_group_layout: None, bind_group: None,
            depth_view: None, depth_size: [0, 0],
        }
    }
}

// ── Model ──────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Model {
    window: Entity,
    mesh: Arc<HalfEdgeMesh>,
    gpu: Option<GpuCompute>,
    render_state: Arc<Mutex<RenderState>>,
    // Cached bounding box for rendering (updated on readback frames)
    cached_center: Vec3,
    cached_scale: f32,
    // Cached spatial hash bounding box from GPU reduction [min_x, min_y, max_x, max_y]
    cached_spatial_bbox: [f32; 4],
    spring_len: f32,
    elastic_constant: f32,
    repulsion_distance: f32,
    repulsion_strength: f32,
    bulge_strength: f32,
    planar_strength: f32,
    dt: f32,
    state_dt: f32,
    damping: f32,
    // Lenia-style params
    kernel_mu: f32,
    kernel_sigma: f32,
    growth_mu: f32,
    growth_sigma: f32,
    split_threshold: f32,
    split_chance: f32,
    cheb_order: usize,
    cheb_coeffs: [f32; 20],
    frame: u64,
    // Camera rotation
    camera_yaw: f32,
    camera_pitch: f32,
    zoom: f32,
    dragging: bool,
    last_mouse: Vec2,
    render_mode: u32,
    start_shape: StartShape,
    ico_nu: usize,
}

fn recompute_cheb_coeffs(model: &mut Model) {
    const MAX_ORDER: usize = 20;

    // Compute all raw Gaussian coefficients
    let mut raw = [0.0f32; MAX_ORDER];
    let mut total = 0.0f32;
    for k in 0..MAX_ORDER {
        let c = (-((k as f32 - model.kernel_mu).powi(2)) / (2.0 * model.kernel_sigma * model.kernel_sigma)).exp();
        raw[k] = c;
        total += c;
    }

    // Auto-select order: smallest N capturing >= 95% of total mass
    let threshold = 0.95 * total;
    let mut cumsum = 0.0f32;
    let mut order = MAX_ORDER;
    for k in 0..MAX_ORDER {
        cumsum += raw[k];
        if cumsum >= threshold {
            order = k + 1;
            break;
        }
    }
    model.cheb_order = order.max(2);

    // Normalize the used coefficients
    let mut sum = 0.0f32;
    for k in 0..model.cheb_order {
        sum += raw[k];
    }
    for k in 0..model.cheb_order {
        model.cheb_coeffs[k] = raw[k] / sum;
    }
    for k in model.cheb_order..MAX_ORDER {
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
    model.state_dt = 0.05 + random_f32() * 0.15;
    model.damping = 0.5;
    // Lenia-style params
    model.kernel_mu = 4.0 + random_f32() * 5.0;
    model.kernel_sigma = 0.5 + random_f32() * 2.0;
    model.growth_mu = random_f32();
    model.growth_sigma = 0.1 + random_f32() * 0.5;
    model.split_threshold = 0.5 + random_f32() * 0.4;
    model.split_chance = 0.01 + random_f32() * 0.09;

    recompute_cheb_coeffs(model);
}

// ── App ────────────────────────────────────────────────────────────────────────

fn main() {
    nannou::app(model)
        .update(update)
        .render(render)
        .run();
}

fn model(app: &App) -> Model {
    let w_id = app.new_window()
        .size(1000, 1000)
        .key_pressed(key_pressed)
        .mouse_pressed(mouse_pressed)
        .mouse_released(mouse_released)
        .mouse_moved(mouse_moved)
        .mouse_wheel(mouse_wheel)
        .build();

    let mut m = Model {
        window: w_id,
        mesh: Arc::from(HalfEdgeMesh::new()),
        gpu: None,
        render_state: Arc::new(Mutex::new(RenderState::default())),
        cached_center: Vec3::ZERO,
        cached_scale: 1.0,
        cached_spatial_bbox: [f32::NEG_INFINITY; 4],
        spring_len: 30.0,
        elastic_constant: 0.1,
        repulsion_distance: 150.0,
        repulsion_strength: 8.0,
        bulge_strength: 10.0,
        planar_strength: 0.4,
        dt: 0.05,
        state_dt: 0.05,
        damping: 0.5,
        kernel_mu: 8.0,
        kernel_sigma: 1.0,
        growth_mu: 0.5,
        growth_sigma: 0.3,
        split_threshold: 0.7,
        split_chance: 0.05,
        cheb_order: 10,
        cheb_coeffs: [0.0; 20],
        frame: 0,
        camera_yaw: 0.0,
        camera_pitch: 0.0,
        zoom: 1.0,
        dragging: false,
        last_mouse: Vec2::ZERO,
        render_mode: 0,
        start_shape: StartShape::Circle,
        ico_nu: 4,
    };
    randomize_params(&mut m);
    m.mesh = Arc::from(make_start_mesh(m.start_shape, m.spring_len, m.ico_nu));
    m
}

fn update(app: &App, model: &mut Model) {
    // egui settings panel
    let mut egui_ctx = app.egui_for_window(model.window);
    let ctx = egui_ctx.get_mut();

    let mut kernel_changed = false;
    let mut shape_changed = false;
    egui::Window::new("Settings").show(&ctx, |ui| {
        ui.label("Start shape");
        let prev_shape = model.start_shape;
        egui::ComboBox::from_id_salt("start_shape")
            .selected_text(match model.start_shape {
                StartShape::Circle => "Circle",
                StartShape::Sphere => "Sphere",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut model.start_shape, StartShape::Circle, "Circle");
                ui.selectable_value(&mut model.start_shape, StartShape::Sphere, "Sphere");
            });
        if model.start_shape == StartShape::Sphere {
            let prev_nu = model.ico_nu;
            ui.add(egui::Slider::new(&mut model.ico_nu, 2..=8).text("subdivisions"));
            if model.ico_nu != prev_nu {
                shape_changed = true;
            }
        }
        if model.start_shape != prev_shape {
            shape_changed = true;
        }
        ui.separator();
        ui.label("Kernel (ring)");
        kernel_changed |= ui.add(egui::Slider::new(&mut model.kernel_mu, 0.0..=18.0).text("mu")).changed();
        kernel_changed |= ui.add(egui::Slider::new(&mut model.kernel_sigma, 0.1..=4.0).text("sigma")).changed();
        ui.label(format!("order: {} (auto)", model.cheb_order));
        ui.separator();
        ui.label("Growth function");
        ui.add(egui::Slider::new(&mut model.growth_mu, 0.0..=1.0).text("mu"));
        ui.add(egui::Slider::new(&mut model.growth_sigma, 0.01..=1.0).text("sigma"));
        ui.separator();
        ui.label("Split");
        ui.add(egui::Slider::new(&mut model.split_threshold, 0.0..=1.0).text("threshold"));
        ui.add(egui::Slider::new(&mut model.split_chance, 0.001..=0.5).text("chance"));
        ui.separator();
        ui.label("Physics");
        ui.add(egui::Slider::new(&mut model.spring_len, 5.0..=80.0).text("spring len"));
        ui.add(egui::Slider::new(&mut model.elastic_constant, 0.01..=0.5).text("elastic"));
        ui.add(egui::Slider::new(&mut model.repulsion_strength, 0.0..=10.0).text("repulsion"));
        ui.add(egui::Slider::new(&mut model.bulge_strength, 0.0..=30.0).text("bulge"));
        ui.add(egui::Slider::new(&mut model.planar_strength, 0.0..=0.5).text("planar"));
        ui.add(egui::Slider::new(&mut model.dt, 0.01..=0.3).text("force dt"));
        ui.add(egui::Slider::new(&mut model.state_dt, 0.01..=0.5).text("state dt"));
        ui.separator();
        ui.label("Controls");
        ui.label("R — Randomize params & reset");
        ui.label("T — Reset mesh (keep params)");
        ui.label("S — Save screenshot");
        ui.label("M — Toggle render mode");
        ui.label("Drag — Rotate camera");
        ui.label("Scroll — Zoom");
    });
    drop(egui_ctx);

    if kernel_changed {
        recompute_cheb_coeffs(model);
    }

    if shape_changed {
        model.mesh = Arc::from(make_start_mesh(model.start_shape, model.spring_len, model.ico_nu));
        model.frame = 0;
        model.camera_yaw = 0.0;
        model.camera_pitch = 0.0;
        model.zoom = 1.0;
        model.cached_spatial_bbox = [f32::NEG_INFINITY; 4];
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }

    model.frame += 1;

    // Initialize GPU on second frame (give window time to fully initialize)
    if model.gpu.is_none() {
        if model.frame < 2 { return; }
        let window = app.window(model.window);
        let device = window.device();
        model.gpu = Some(create_gpu_compute(&device));
        // Mark topology dirty so initial upload happens
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }

    // Spatial hash grid params from cached GPU bounding box (updated on readback frames)
    let mesh = &model.mesh;
    let bin_size = model.repulsion_distance;
    let bbox = model.cached_spatial_bbox;
    let (origin_x, origin_y, num_bins_x, num_bins_y) = if bbox[0].is_finite() {
        let ox = bbox[0] - bin_size;
        let oy = bbox[1] - bin_size;
        let nbx = (((bbox[2] - ox) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        let nby = (((bbox[3] - oy) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        (ox, oy, nbx, nby)
    } else {
        // First frame: compute from CPU mesh (GPU bbox not yet available)
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for v in 0..mesh.next_vertex {
            if mesh.vertex_idx[v] < 0 { continue; }
            min_x = min_x.min(mesh.vertex_pos[v].x);
            min_y = min_y.min(mesh.vertex_pos[v].y);
            max_x = max_x.max(mesh.vertex_pos[v].x);
            max_y = max_y.max(mesh.vertex_pos[v].y);
        }
        let ox = min_x - bin_size;
        let oy = min_y - bin_size;
        let nbx = (((max_x - ox) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        let nby = (((max_y - oy) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        (ox, oy, nbx, nby)
    };

    let gpu_params = GpuSimParams {
        num_vertices: mesh.next_vertex as u32,
        num_half_edges: mesh.next_half_edge as u32,
        repulsion_distance: model.repulsion_distance,
        spring_len: model.spring_len,
        elastic_constant: model.elastic_constant,
        bulge_strength: model.bulge_strength,
        planar_strength: model.planar_strength,
        dt: model.dt,
        origin_x,
        origin_y,
        bin_size,
        num_bins_x,
        num_bins_y,
        growth_mu: model.growth_mu,
        growth_sigma: model.growth_sigma,
        cheb_order: model.cheb_order as u32,
        repulsion_strength: model.repulsion_strength,
        state_dt: model.state_dt,
    };

    // GPU dispatch: physics + state evolution
    {
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        gpu_dispatch_frame(
            &device,
            &queue,
            gpu,
            &model.mesh,
            &gpu_params,
            model.cheb_order,
            &model.cheb_coeffs,
        );
    }

    // Readback + topology ops only every 10/20 frames (rendering reads GPU buffers directly)
    let needs_topology_ops = model.frame % 10 == 0;
    if needs_topology_ops || model.frame <= 1 {
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        let mesh = Arc::make_mut(&mut model.mesh);
        let bbox = readback_from_gpu(&device, &queue, gpu, mesh);
        model.cached_spatial_bbox = bbox;

        // Update cached bounding box for render uniforms
        let mut min_pos = Vec3::splat(f32::INFINITY);
        let mut max_pos = Vec3::splat(f32::NEG_INFINITY);
        let mut any_active = false;
        for v in 0..mesh.next_vertex {
            if mesh.vertex_idx[v] < 0 { continue; }
            any_active = true;
            min_pos = min_pos.min(mesh.vertex_pos[v]);
            max_pos = max_pos.max(mesh.vertex_pos[v]);
        }
        if any_active {
            model.cached_center = (min_pos + max_pos) * 0.5;
            let mut max_radius = 0.0f32;
            for v in 0..mesh.next_vertex {
                if mesh.vertex_idx[v] < 0 { continue; }
                let r = (mesh.vertex_pos[v] - model.cached_center).length();
                max_radius = max_radius.max(r);
            }
            max_radius *= 1.15;
            model.cached_scale = 8.0 / max_radius.max(1.0);
        }

        // Growth (every 10 frames)
        generate_new_triangles(mesh, model.split_threshold, model.split_chance);

        // Mesh refinement (every 20 frames)
        if model.frame % 20 == 0 {
            mesh.refine_mesh();
        }

        // Upload new mesh data to GPU immediately so render indices stay in sync
        upload_mesh_to_gpu(&queue, gpu, mesh);
        gpu.topology_dirty = false;

        // Rebuild render index buffer
        rebuild_render_indices(&queue, gpu, mesh);
    }

    // Update render uniforms every frame (camera may have changed)
    {
        let window = app.window(model.window);
        let queue = window.queue();
        let gpu = model.gpu.as_ref().unwrap();
        let sz = window.size_pixels();
        let aspect = sz.x as f32 / sz.y.max(1) as f32;
        update_render_uniforms(
            &queue, gpu,
            model.cached_center, model.cached_scale,
            model.camera_yaw, model.camera_pitch,
            model.zoom, model.render_mode, aspect,
        );
    }

    #[cfg(debug_assertions)]
    if model.frame % 100 == 0 {
        validate_mesh(&model.mesh);
    }
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32, msaa_samples: u32) -> wgpu::TextureViewHandle {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn init_render_state(
    device: &wgpu::Device,
    gpu: &GpuCompute,
    rs: &mut RenderState,
    texture_format: wgpu::TextureFormat,
    msaa_samples: u32,
    texture_size: [u32; 2],
) {
    let vs = wgpu::ShaderStages::VERTEX;
    let fs = wgpu::ShaderStages::VERTEX_FRAGMENT;

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: vs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: vs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: fs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gpu.vertex_pos_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gpu.vertex_state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gpu.render_index_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gpu.render_uniform_buf.as_entire_binding() },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh_render"),
        source: wgpu::ShaderSource::Wgsl(MESH_RENDER_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pl"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mesh_render"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None, // double-sided, use front_facing in shader
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    let depth_view = create_depth_texture(device, texture_size[0], texture_size[1], msaa_samples);

    rs.bind_group_layout = Some(layout);
    rs.bind_group = Some(bind_group);
    rs.pipeline = Some(pipeline);
    rs.depth_view = Some(depth_view);
    rs.depth_size = texture_size;
}

fn render(_render_app: &RenderApp, model: &Model, mut frame: Frame) {
    frame.clear(Color::BLACK);

    let Some(ref gpu) = model.gpu else { return; };
    if gpu.num_render_tris == 0 { return; }

    let mut rs = model.render_state.lock().unwrap();
    let texture_size = frame.texture_size();
    let msaa_samples = frame.texture_msaa_samples();

    if rs.pipeline.is_none() {
        let device = frame.device();
        let texture_format = frame.texture_format();
        init_render_state(device, gpu, &mut rs, texture_format, msaa_samples, texture_size);
    }

    // Recreate depth texture on resize
    if rs.depth_size != texture_size {
        let device = frame.device();
        rs.depth_view = Some(create_depth_texture(device, texture_size[0], texture_size[1], msaa_samples));
        rs.depth_size = texture_size;
    }

    let pipeline = rs.pipeline.as_ref().unwrap();
    let bind_group = rs.bind_group.as_ref().unwrap();
    let depth_view = rs.depth_view.as_ref().unwrap();
    let num_verts = gpu.num_render_tris * 3;

    let texture_view = frame.texture_view();
    let mut encoder = frame.command_encoder();

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..num_verts, 0..1);
    }
}

fn key_pressed(app: &App, model: &mut Model, key: KeyCode) {
    if key == KeyCode::KeyS {
        app.window(model.window)
            .save_screenshot(app.exe_name().unwrap() + ".png");
    }
    if key == KeyCode::KeyM {
        model.render_mode = (model.render_mode + 1) % 2;
    }
    if key == KeyCode::KeyR {
        randomize_params(model);
        model.mesh = Arc::from(make_start_mesh(model.start_shape, model.spring_len, model.ico_nu));
        model.frame = 0;
        model.camera_yaw = 0.0;
        model.camera_pitch = 0.0;
        model.zoom = 1.0;
        model.cached_spatial_bbox = [f32::NEG_INFINITY; 4];
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }
    if key == KeyCode::KeyT {
        model.mesh = Arc::from(make_start_mesh(model.start_shape, model.spring_len, model.ico_nu));
        model.frame = 0;
        model.camera_yaw = 0.0;
        model.camera_pitch = 0.0;
        model.zoom = 1.0;
        model.cached_spatial_bbox = [f32::NEG_INFINITY; 4];
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
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

fn mouse_wheel(_app: &App, model: &mut Model, wheel: MouseWheel) {
    let factor = 1.0 + wheel.y * 0.1;
    model.zoom = (model.zoom * factor).clamp(0.1, 20.0);
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
