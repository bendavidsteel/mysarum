use nannou::prelude::*;

// ── Constants ──────────────────────────────────────────────────────────────────

pub(crate) const MAX_VERTICES: usize = 8000;
pub(crate) const MAX_HALF_EDGES: usize = MAX_VERTICES * 6;
pub(crate) const MAX_FACES: usize = MAX_VERTICES * 2;
pub(crate) const EPSILON: f32 = 1e-6;
pub(crate) const N_RINGS: usize = 24;
pub(crate) const SEGMENTS_INNER: usize = 6;

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

    // Cached active count (incremented on alloc, currently never decremented)
    pub(crate) num_active_vertices: usize,

    // Reusable scratch buffers
    pub(crate) scratch_splits: Vec<(usize, usize, bool, bool)>,
    pub(crate) scratch_edge_counts: Vec<usize>,
    pub(crate) scratch_edges_to_check: Vec<usize>,
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
            num_active_vertices: 0,
            scratch_splits: Vec::with_capacity(256),
            scratch_edge_counts: Vec::with_capacity(MAX_VERTICES),
            scratch_edges_to_check: Vec::with_capacity(MAX_HALF_EDGES),
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
        self.num_active_vertices += 1;
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

    pub(crate) fn active_vertex_count(&self) -> usize {
        self.num_active_vertices
    }

    // ── Get half-edge from vertex a to vertex b ────────────────────────────

    pub(crate) fn get_vertex_half_edge(&self, vertex_a: usize, vertex_b: usize) -> Option<usize> {
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

    pub(crate) fn add_external_triangle(&mut self, vertex_a: usize, vertex_b: usize) -> bool {
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

    pub(crate) fn add_internal_edge_triangle(&mut self, vertex_a: usize, vertex_b: usize) -> bool {
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

    pub(crate) fn add_internal_triangles(&mut self, vertex_a: usize, vertex_b: usize) -> bool {
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

    pub(crate) fn refine_mesh(&mut self) {
        self.scratch_edge_counts.clear();
        self.scratch_edge_counts.resize(self.next_vertex, 0);
        for v in 0..self.next_vertex {
            if self.vertex_idx[v] >= 0 {
                self.scratch_edge_counts[v] = self.get_edge_count(v);
            }
        }

        // Collect edges to potentially flip (only consider he where dest < dest_twin to avoid duplicates)
        self.scratch_edges_to_check.clear();
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
                self.scratch_edges_to_check.push(he);
            }
        }

        // Need to iterate over a copy since we borrow self mutably for flip_edge
        let edges_to_check: Vec<usize> = self.scratch_edges_to_check.clone();

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

            let ec_a = self.scratch_edge_counts[va] as i32;
            let ec_b = self.scratch_edge_counts[vb] as i32;
            let ec_c = self.scratch_edge_counts[vc] as i32;
            let ec_d = self.scratch_edge_counts[vd] as i32;

            let no_flip_cost = (ec_a - 6).pow(2) + (ec_b - 6).pow(2) + (ec_c - 6).pow(2) + (ec_d - 6).pow(2);
            let flip_cost = (ec_a - 7).pow(2) + (ec_b - 7).pow(2) + (ec_c - 5).pow(2) + (ec_d - 5).pow(2);

            if flip_cost < no_flip_cost {
                if self.flip_edge(he) {
                    self.scratch_edge_counts[va] -= 1;
                    self.scratch_edge_counts[vb] -= 1;
                    self.scratch_edge_counts[vc] += 1;
                    self.scratch_edge_counts[vd] += 1;
                }
            }
        }
    }
}

// ── Growth: generate new triangles ─────────────────────────────────────────────

pub(crate) fn generate_new_triangles(mesh: &mut HalfEdgeMesh, split_threshold: f32, split_chance: f32) {
    // Collect vertices that want to split
    mesh.scratch_splits.clear();

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

        mesh.scratch_splits.push((src, dest, he_on_boundary, twin_on_boundary));
    }

    // Need to iterate over a copy since we mutate mesh
    let splits: Vec<(usize, usize, bool, bool)> = mesh.scratch_splits.clone();

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

// ── Debug validation ───────────────────────────────────────────────────────────

#[cfg(debug_assertions)]
pub(crate) fn validate_mesh(mesh: &HalfEdgeMesh) {
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
