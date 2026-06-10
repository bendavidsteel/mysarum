use nannou::prelude::*;

// ── Constants ──────────────────────────────────────────────────────────────────

pub(crate) const MAX_VERTICES: usize = 32000;
pub(crate) const MAX_HALF_EDGES: usize = MAX_VERTICES * 6;
pub(crate) const MAX_FACES: usize = MAX_VERTICES * 2;
pub(crate) const EPSILON: f32 = 1e-6;
pub(crate) const N_RINGS: usize = 24;
pub(crate) const SEGMENTS_INNER: usize = 6;

// ── Half-edge mesh (structure-of-arrays, heap-allocated) ───────────────────────

pub(crate) struct HalfEdgeMesh {
    // Face arrays
    pub(crate) face_idx: [i32; MAX_FACES],
    pub(crate) face_half_edge: [i32; MAX_FACES],

    // Vertex arrays
    pub(crate) vertex_idx: [i32; MAX_VERTICES],
    pub(crate) vertex_half_edge: [i32; MAX_VERTICES],
    pub(crate) vertex_pos: [Vec3; MAX_VERTICES],
    pub(crate) vertex_state: [f32; MAX_VERTICES],
    // Pinned vertices are excluded from physics (positions never move) and
    // from intrinsic edge growth. Uploaded to the GPU as vertex_pos.w == 0.0.
    pub(crate) vertex_pinned: [bool; MAX_VERTICES],
    // Grow-at-dot mode: 1.0 marks a fixed "source" vertex whose growth potential
    // (vertex_state) is held at 1.0 every frame; the potential diffuses outward
    // via a heat equation. New vertices from splits default to 0.0 (no natural
    // growth), so the growth region stays anchored to the original dot.
    pub(crate) vertex_source: [f32; MAX_VERTICES],

    // Half-edge arrays
    pub(crate) half_edge_idx: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_twin: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_dest: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_face: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_next: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_prev: [i32; MAX_HALF_EDGES],
    pub(crate) half_edge_intrinsic_len: Vec<f32>,

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
        // Allocate directly on the heap to avoid stack overflow with large arrays
        let mut mesh: Box<Self> = unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        };
        // Set sentinel values for index arrays (-1 = inactive)
        mesh.face_idx.fill(-1);
        mesh.face_half_edge.fill(-1);
        mesh.vertex_idx.fill(-1);
        mesh.vertex_half_edge.fill(-1);
        mesh.half_edge_idx.fill(-1);
        mesh.half_edge_twin.fill(-1);
        mesh.half_edge_dest.fill(-1);
        mesh.half_edge_face.fill(-1);
        mesh.half_edge_next.fill(-1);
        mesh.half_edge_prev.fill(-1);
        // Vec fields are zero-initialized (null ptr, 0 len/cap) — replace with real Vecs
        mesh.half_edge_intrinsic_len = vec![0.0f32; MAX_HALF_EDGES];
        mesh.scratch_splits = Vec::with_capacity(256);
        mesh.scratch_edge_counts = Vec::with_capacity(MAX_VERTICES);
        mesh.scratch_edges_to_check = Vec::with_capacity(MAX_HALF_EDGES);
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

    /// Clone entirely on the heap, avoiding the ~5MB stack allocation
    /// that `Clone::clone()` would require.
    pub(crate) fn clone_boxed(&self) -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            // Bitwise copy all fixed-size arrays and scalars
            std::ptr::copy_nonoverlapping(
                self as *const Self as *const u8,
                ptr as *mut u8,
                layout.size(),
            );
            // The Vec fields were bitwise-copied (shared heap pointers).
            // Overwrite with proper clones using ptr::write (no drop of old value).
            std::ptr::write(&mut (*ptr).half_edge_intrinsic_len, self.half_edge_intrinsic_len.clone());
            std::ptr::write(&mut (*ptr).scratch_splits, self.scratch_splits.clone());
            std::ptr::write(&mut (*ptr).scratch_edge_counts, self.scratch_edge_counts.clone());
            std::ptr::write(&mut (*ptr).scratch_edges_to_check, self.scratch_edges_to_check.clone());
            Box::from_raw(ptr)
        }
    }

    /// Compute intrinsic edge lengths from current extrinsic (3D) vertex positions.
    /// Call after building the mesh or after any operation that creates new geometry.
    pub(crate) fn compute_intrinsic_lengths(&mut self) {
        for he in 0..self.next_half_edge {
            if self.half_edge_idx[he] < 0 { continue; }
            let dst = self.half_edge_dest[he];
            let twin = self.half_edge_twin[he];
            if dst < 0 || twin < 0 { continue; }
            let src = self.half_edge_dest[twin as usize];
            if src < 0 { continue; }
            let d = self.vertex_pos[dst as usize] - self.vertex_pos[src as usize];
            let len = d.length();
            // Guard against coincident endpoints so we never seed a zero rest length.
            self.half_edge_intrinsic_len[he] = len.max(EPSILON);
        }
    }

    /// Symmetrize intrinsic lengths: set each half-edge and its twin to their
    /// average. The growth shader grows each side independently (a vertex only
    /// writes its outgoing half-edges), and the spring shader enforces the
    /// twin-averaged rest length — but the CPU topology ops (Delaunay flips,
    /// split median formulas) read single-sided values. Left unsymmetrized, a
    /// divergent he/twin pair presents a triangle-inequality-violating triangle
    /// to the flip unfold, which then mints spuriously long edges. Call after
    /// reading lengths back from the GPU, before any topology op.
    pub(crate) fn symmetrize_intrinsic_lengths(&mut self) {
        for he in 0..self.next_half_edge {
            if self.half_edge_idx[he] < 0 { continue; }
            let twin = self.half_edge_twin[he];
            if twin < 0 || (twin as usize) < he { continue; }
            let twin = twin as usize;
            let avg = (self.half_edge_intrinsic_len[he] + self.half_edge_intrinsic_len[twin]) * 0.5;
            self.half_edge_intrinsic_len[he] = avg;
            self.half_edge_intrinsic_len[twin] = avg;
        }
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

        // Save intrinsic length of AB before repurposing half-edges
        let l_ab = self.half_edge_intrinsic_len[half_edge_ab];
        let l_bc = self.half_edge_intrinsic_len[half_edge_bc];
        let l_ca = self.half_edge_intrinsic_len[half_edge_ca];

        // New vertex with z-perturbation to allow 3D buckling
        let mut new_pos = (self.vertex_pos[vertex_a] + self.vertex_pos[vertex_b]) * 0.5;
        let edge_len = (self.vertex_pos[vertex_b] - self.vertex_pos[vertex_a]).length();
        new_pos.z += (random_f32() - 0.5) * edge_len * 0.05;
        self.vertex_pos[vertex_d] = new_pos;
        self.vertex_state[vertex_d] = (self.vertex_state[vertex_a] + self.vertex_state[vertex_b]) * 0.5;
        self.vertex_pinned[vertex_d] = self.vertex_pinned[vertex_a] && self.vertex_pinned[vertex_b];
        // Grow-at-dot: source only if both endpoints are sources (AND).
        self.vertex_source[vertex_d] =
            if self.vertex_source[vertex_a] > 0.5 && self.vertex_source[vertex_b] > 0.5 { 1.0 } else { 0.0 };
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

        // Intrinsic lengths: AD and DB are halves of AB; DC via median formula
        let l_ad = l_ab * 0.5;
        let l_db = l_ab * 0.5;
        let l_dc_sq = (2.0 * l_ca * l_ca + 2.0 * l_bc * l_bc - l_ab * l_ab) / 4.0;
        // Floor at EPSILON: a triangle-inequality-violating split would yield a
        // zero length, which poisons later Delaunay checks and flip unfolds.
        let l_dc = l_dc_sq.max(0.0).sqrt().max(EPSILON);
        self.half_edge_intrinsic_len[half_edge_ad] = l_ad;  // repurposed from ab
        self.half_edge_intrinsic_len[half_edge_da] = l_ad;
        self.half_edge_intrinsic_len[half_edge_db] = l_db;
        self.half_edge_intrinsic_len[half_edge_bd] = l_db;  // repurposed from ba
        self.half_edge_intrinsic_len[half_edge_dc] = l_dc;
        self.half_edge_intrinsic_len[half_edge_cd] = l_dc;

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

        // Save intrinsic lengths before repurposing half-edges
        let l_ab = self.half_edge_intrinsic_len[half_edge_ab];
        let l_bc = self.half_edge_intrinsic_len[half_edge_bc_old];
        let l_ca = self.half_edge_intrinsic_len[half_edge_ca];
        let l_ad = self.half_edge_intrinsic_len[half_edge_ad];
        let l_db = self.half_edge_intrinsic_len[half_edge_db];

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
        self.vertex_pinned[vertex_e] = self.vertex_pinned[vertex_a] && self.vertex_pinned[vertex_b];
        // Grow-at-dot: source only if both endpoints are sources (AND).
        self.vertex_source[vertex_e] =
            if self.vertex_source[vertex_a] > 0.5 && self.vertex_source[vertex_b] > 0.5 { 1.0 } else { 0.0 };
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

        // Intrinsic lengths: AE/EB halves of AB; EC/ED via median formula
        let l_ae = l_ab * 0.5;
        let l_eb = l_ab * 0.5;
        // Floor at EPSILON: see add_internal_edge_triangle's median formula.
        let l_ec_sq = (2.0 * l_ca * l_ca + 2.0 * l_bc * l_bc - l_ab * l_ab) / 4.0;
        let l_ec = l_ec_sq.max(0.0).sqrt().max(EPSILON);
        let l_ed_sq = (2.0 * l_ad * l_ad + 2.0 * l_db * l_db - l_ab * l_ab) / 4.0;
        let l_ed = l_ed_sq.max(0.0).sqrt().max(EPSILON);

        self.half_edge_intrinsic_len[half_edge_ae] = l_ae;  // repurposed from ab
        self.half_edge_intrinsic_len[half_edge_ea] = l_ae;
        self.half_edge_intrinsic_len[half_edge_eb] = l_eb;
        self.half_edge_intrinsic_len[half_edge_be] = l_eb;  // repurposed from ba
        self.half_edge_intrinsic_len[half_edge_ec] = l_ec;
        self.half_edge_intrinsic_len[half_edge_ce] = l_ec;
        self.half_edge_intrinsic_len[half_edge_ed] = l_ed;
        self.half_edge_intrinsic_len[half_edge_de] = l_ed;

        true
    }

    // ── Flip edge ──────────────────────────────────────────────────────────

    fn flip_edge(&mut self, half_edge_ab: usize, max_flip_len: f32) -> bool {
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

        // Link condition: if C and D are already adjacent, flipping would create
        // a double edge. The GPU fan walks terminate by destination comparison,
        // so a double edge silently truncates them — part of the fan stops
        // receiving spring constraints and drifts apart unboundedly.
        if self.get_vertex_half_edge(vertex_c, vertex_d).is_some() {
            return false;
        }

        // Compute new intrinsic edge length for CD by unfolding the two triangles
        let lab = self.half_edge_intrinsic_len[half_edge_ab];
        let lbc = self.half_edge_intrinsic_len[half_edge_bc];
        let lca = self.half_edge_intrinsic_len[half_edge_ca];
        let lad = self.half_edge_intrinsic_len[half_edge_ad];
        let ldb = self.half_edge_intrinsic_len[half_edge_db];

        // Only flip if both intrinsic triangles are metrically valid. On
        // triangle-inequality-violating input the clamped unfold degenerates to
        // a colinear placement and the "diagonal" comes out near lca + lad —
        // a spuriously long edge that then compounds through later flips.
        if !triangle_inequality_holds(lab, lbc, lca) || !triangle_inequality_holds(lab, lad, ldb) {
            return false;
        }

        let lcd = intrinsic_flip_length(lab, lca, lbc, lad, ldb);

        // Backstop: never mint an edge longer than the system-wide ceiling
        // (growth and springs both cap at spring_len * 3) or degenerate-short.
        if lcd > max_flip_len || lcd < EPSILON {
            return false;
        }

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

        // New intrinsic edge length for the flipped edge
        self.half_edge_intrinsic_len[half_edge_dc] = lcd;
        self.half_edge_intrinsic_len[half_edge_cd] = lcd;

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

    // ── Mesh refinement (intrinsic Delaunay flips) ──────────────────────────

    pub(crate) fn refine_mesh(&mut self, max_flip_len: f32) {
        // Collect interior edges to check (deduplicate: only where dest < dest_twin)
        self.scratch_edges_to_check.clear();
        for he in 0..self.next_half_edge {
            if self.half_edge_idx[he] < 0 { continue; }
            if self.half_edge_face[he] == -1 { continue; }
            let twin = self.half_edge_twin[he];
            if twin < 0 || self.half_edge_face[twin as usize] == -1 { continue; }
            let dest_a = self.half_edge_dest[he];
            let dest_b = self.half_edge_dest[twin as usize];
            if dest_a < dest_b {
                self.scratch_edges_to_check.push(he);
            }
        }

        let edges_to_check: Vec<usize> = self.scratch_edges_to_check.clone();

        for he in edges_to_check {
            if self.half_edge_idx[he] < 0 { continue; }
            let twin_i = self.half_edge_twin[he];
            if twin_i < 0 { continue; }
            let twin = twin_i as usize;
            if self.half_edge_face[he] == -1 || self.half_edge_face[twin] == -1 { continue; }

            // Check intrinsic Delaunay criterion: sum of opposite angles > π → flip
            let he_bc = self.half_edge_next[he];
            let he_ad = self.half_edge_next[twin];
            if he_bc < 0 || he_ad < 0 { continue; }
            let he_ca = self.half_edge_next[he_bc as usize];
            let he_db = self.half_edge_next[he_ad as usize];
            if he_ca < 0 || he_db < 0 { continue; }

            let lab = self.half_edge_intrinsic_len[he];
            let lbc = self.half_edge_intrinsic_len[he_bc as usize];
            let lca = self.half_edge_intrinsic_len[he_ca as usize];
            let lad = self.half_edge_intrinsic_len[he_ad as usize];
            let ldb = self.half_edge_intrinsic_len[he_db as usize];

            if !is_intrinsic_delaunay(lab, lbc, lca, lad, ldb) {
                self.flip_edge(he, max_flip_len);
            }
        }
    }

    // ── Binary STL export ──────────────────────────────────────────────────
    //
    // Writes every active triangle to a binary STL file using the current 3D
    // embedding (`vertex_pos`). Triangle winding mirrors `rebuild_render_indices`
    // (dest of the face's three half-edges, CCW), and per-face normals are
    // computed from the geometry so the orientation matches the winding.
    //
    // NOTE: the grown mesh is an open surface (a 2-manifold with boundary), so
    // the exported STL is NOT watertight. To 3D print it, give it thickness
    // (e.g. Blender's Solidify modifier) or slice it in vase mode. Coordinates
    // are raw simulation units — scale to physical size in the slicer.
    pub(crate) fn export_stl(&self, path: &std::path::Path) -> std::io::Result<usize> {
        use std::io::Write;

        // Collect active triangles as (v0, v1, v2) vertex indices.
        let mut tris: Vec<[usize; 3]> = Vec::new();
        for f in 0..self.next_face {
            if self.face_idx[f] < 0 {
                continue;
            }
            let he0 = self.face_half_edge[f];
            if he0 < 0 {
                continue;
            }
            let he0 = he0 as usize;
            let he1 = self.half_edge_next[he0];
            if he1 < 0 {
                continue;
            }
            let he1 = he1 as usize;
            let he2 = self.half_edge_next[he1];
            if he2 < 0 {
                continue;
            }
            let he2 = he2 as usize;

            let v0 = self.half_edge_dest[he0];
            let v1 = self.half_edge_dest[he1];
            let v2 = self.half_edge_dest[he2];
            if v0 < 0 || v1 < 0 || v2 < 0 {
                continue;
            }
            let (v0, v1, v2) = (v0 as usize, v1 as usize, v2 as usize);
            if self.vertex_idx[v0] < 0 || self.vertex_idx[v1] < 0 || self.vertex_idx[v2] < 0 {
                continue;
            }
            tris.push([v0, v1, v2]);
        }

        // Binary STL: 80-byte header + u32 triangle count + 50 bytes per triangle.
        let mut buf: Vec<u8> = Vec::with_capacity(84 + tris.len() * 50);
        buf.extend_from_slice(&[0u8; 80]); // header (must not begin with "solid")
        buf.extend_from_slice(&(tris.len() as u32).to_le_bytes());

        let write_vec3 = |buf: &mut Vec<u8>, v: Vec3| {
            buf.extend_from_slice(&v.x.to_le_bytes());
            buf.extend_from_slice(&v.y.to_le_bytes());
            buf.extend_from_slice(&v.z.to_le_bytes());
        };

        for [v0, v1, v2] in &tris {
            let p0 = self.vertex_pos[*v0];
            let p1 = self.vertex_pos[*v1];
            let p2 = self.vertex_pos[*v2];
            let normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            write_vec3(&mut buf, normal);
            write_vec3(&mut buf, p0);
            write_vec3(&mut buf, p1);
            write_vec3(&mut buf, p2);
            buf.extend_from_slice(&0u16.to_le_bytes()); // attribute byte count
        }

        let mut file = std::fs::File::create(path)?;
        file.write_all(&buf)?;
        Ok(tris.len())
    }
}

// ── Intrinsic geometry helpers ─────────────────────────────────────────────────

/// Check the triangle inequality with a small relative tolerance. Intrinsic
/// lengths are decoupled from the embedding, so violations can and do occur;
/// callers use this to skip operations whose math is meaningless on such input.
fn triangle_inequality_holds(a: f32, b: f32, c: f32) -> bool {
    let tol = (a + b + c) * 1e-4;
    a <= b + c + tol && b <= a + c + tol && c <= a + b + tol
}

/// Check if edge AB is locally Delaunay using intrinsic edge lengths.
/// Edge AB with opposite vertices C (in triangle ABC) and D (in triangle ABD).
/// Arguments: lab=|AB|, lbc=|BC|, lca=|CA|, lad=|AD|, ldb=|DB|
/// Returns true if the edge is Delaunay (opposite angles sum ≤ π).
fn is_intrinsic_delaunay(lab: f32, lbc: f32, lca: f32, lad: f32, ldb: f32) -> bool {
    let lab2 = lab * lab;
    // Cosines are clamped to [-1, 1]: intrinsic lengths can violate the triangle
    // inequality (growth/splits are decoupled from the embedding), and unclamped
    // values make the criterion garbage on degenerate data.
    // Angle at C in triangle ACB: opposite edge is AB
    let cos_c = ((lca * lca + lbc * lbc - lab2) / (2.0 * lca * lbc).max(EPSILON)).clamp(-1.0, 1.0);
    let sin_c = (1.0 - cos_c * cos_c).max(0.0).sqrt();
    // Angle at D in triangle ADB: opposite edge is AB
    let cos_d = ((lad * lad + ldb * ldb - lab2) / (2.0 * lad * ldb).max(EPSILON)).clamp(-1.0, 1.0);
    let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();
    // sin(α_C + α_D) < 0 means sum > π → not Delaunay
    let sin_sum = sin_c * cos_d + cos_c * sin_d;
    sin_sum >= -EPSILON
}

/// Compute the intrinsic edge length of CD after flipping edge AB→CD.
/// Unfolds the two triangles ACB and ADB into a plane and measures |CD|.
/// Arguments: lab=|AB|, lca=|CA|, lbc=|BC|, lad=|AD|, ldb=|DB|
fn intrinsic_flip_length(lab: f32, lca: f32, lbc: f32, lad: f32, ldb: f32) -> f32 {
    if lab < EPSILON { return 0.0; }
    // Cosines are clamped to [-1, 1] (project degenerate triangles onto a line):
    // unclamped, a triangle-inequality-violating configuration yields |cos| >> 1
    // and an astronomically long flipped edge, which the springs then enforce.
    // Place A at origin, B at (lab, 0)
    // C in upper half-plane (triangle ACB)
    let cos_a_acb = ((lab * lab + lca * lca - lbc * lbc) / (2.0 * lab * lca).max(EPSILON)).clamp(-1.0, 1.0);
    let sin_a_acb = (1.0 - cos_a_acb * cos_a_acb).max(0.0).sqrt();
    let cx = lca * cos_a_acb;
    let cy = lca * sin_a_acb;
    // D in lower half-plane (triangle ADB)
    let cos_a_adb = ((lab * lab + lad * lad - ldb * ldb) / (2.0 * lab * lad).max(EPSILON)).clamp(-1.0, 1.0);
    let sin_a_adb = (1.0 - cos_a_adb * cos_a_adb).max(0.0).sqrt();
    let dx = lad * cos_a_adb;
    let dy = -lad * sin_a_adb; // opposite side of AB
    let diffx = cx - dx;
    let diffy = cy - dy;
    (diffx * diffx + diffy * diffy).sqrt()
}

// ── Helper: is this half-edge the longest (intrinsic) in its face? ────────────

fn is_longest_in_face(mesh: &HalfEdgeMesh, he: usize, len: f32) -> bool {
    let face = mesh.half_edge_face[he];
    if face < 0 { return true; }
    let next = mesh.half_edge_next[he] as usize;
    let prev = mesh.half_edge_prev[he] as usize;
    // Use averaged intrinsic lengths (same as split_long_edges threshold check)
    let next_twin = mesh.half_edge_twin[next];
    let prev_twin = mesh.half_edge_twin[prev];
    let next_len = if next_twin >= 0 {
        (mesh.half_edge_intrinsic_len[next] + mesh.half_edge_intrinsic_len[next_twin as usize]) * 0.5
    } else {
        mesh.half_edge_intrinsic_len[next]
    };
    let prev_len = if prev_twin >= 0 {
        (mesh.half_edge_intrinsic_len[prev] + mesh.half_edge_intrinsic_len[prev_twin as usize]) * 0.5
    } else {
        mesh.half_edge_intrinsic_len[prev]
    };
    len >= next_len && len >= prev_len
}

// ── Growth: split edges that exceed an intrinsic length threshold ─────────────

pub(crate) fn split_long_edges(mesh: &mut HalfEdgeMesh, max_edge_len: f32) -> bool {
    mesh.scratch_splits.clear();

    for he in 0..mesh.next_half_edge {
        if mesh.half_edge_idx[he] < 0 { continue; }

        let twin_i = mesh.half_edge_twin[he];
        if twin_i < 0 { continue; }
        let twin = twin_i as usize;

        let dst = mesh.half_edge_dest[he];
        let src = mesh.half_edge_dest[twin];
        if src < 0 || dst < 0 { continue; }
        if src >= dst { continue; }

        // Use average of both half-edges' intrinsic lengths (grown from each side)
        let len = (mesh.half_edge_intrinsic_len[he]
                 + mesh.half_edge_intrinsic_len[twin]) * 0.5;
        if len < max_edge_len { continue; }

        if !is_longest_in_face(mesh, he, len) { continue; }
        if mesh.half_edge_face[twin] >= 0 && !is_longest_in_face(mesh, twin, len) { continue; }

        let src = src as usize;
        let dst = dst as usize;
        let he_on_boundary = mesh.half_edge_face[he] == -1;
        let twin_on_boundary = mesh.half_edge_face[twin] == -1;

        mesh.scratch_splits.push((src, dst, he_on_boundary, twin_on_boundary));
    }

    do_splits(mesh)
}

// ── Shared: execute collected splits ────────────────────────────────────────────

fn do_splits(mesh: &mut HalfEdgeMesh) -> bool {
    let mut splits: Vec<(usize, usize, bool, bool)> = mesh.scratch_splits.clone();

    // Shuffle to avoid systematic bias toward low-indexed vertices.
    // Without this, splits are always processed in vertex index order,
    // so the same spatial region (where low indices cluster) consistently
    // gets its splits executed first while later splits fail due to
    // topology changes from earlier ones.
    for i in (1..splits.len()).rev() {
        let j = (random_f32() * (i + 1) as f32) as usize;
        splits.swap(i, j.min(i));
    }

    let mut hit_max = false;
    for (src, dest, _he_on_boundary, _twin_on_boundary) in splits {
        if mesh.active_vertex_count() >= MAX_VERTICES - 5 {
            hit_max = true;
            break;
        }

        // Re-check edge existence and boundary status (may have changed from earlier splits)
        let he = match mesh.get_vertex_half_edge(src, dest) {
            Some(h) => h,
            None => continue, // edge no longer exists
        };
        let twin_i = mesh.half_edge_twin[he];
        if twin_i < 0 { continue; }
        let twin = twin_i as usize;

        let he_boundary = mesh.half_edge_face[he] == -1;
        let twin_boundary = mesh.half_edge_face[twin] == -1;

        if he_boundary && twin_boundary {
            continue; // invalid: both sides boundary
        } else if he_boundary || twin_boundary {
            if twin_boundary {
                mesh.add_internal_edge_triangle(src, dest);
            } else {
                mesh.add_internal_edge_triangle(dest, src);
            }
        } else {
            mesh.add_internal_triangles(src, dest);
        }
    }
    hit_max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_builders::make_icosphere;

    /// Count distinct edges between each vertex pair; a healthy triangulation
    /// never has two edges connecting the same two vertices.
    fn assert_no_double_edges(mesh: &HalfEdgeMesh) {
        use std::collections::HashSet;
        let mut seen: HashSet<(i32, i32)> = HashSet::new();
        for he in 0..mesh.next_half_edge {
            if mesh.half_edge_idx[he] < 0 { continue; }
            let twin = mesh.half_edge_twin[he];
            if twin < 0 { continue; }
            let a = mesh.half_edge_dest[twin as usize];
            let b = mesh.half_edge_dest[he];
            if a < b {
                assert!(seen.insert((a, b)), "double edge between vertices {a} and {b}");
            }
        }
    }

    /// Regression test for the runaway-edge-length bug: divergent one-sided
    /// intrinsic lengths used to present triangle-inequality-violating
    /// triangles to the Delaunay flip, whose clamped unfold then minted edges
    /// near lca + lad — far past every other cap in the system — and the
    /// garbage compounded across refine passes.
    #[test]
    fn refine_on_garbage_lengths_mints_no_long_edges() {
        let spring_len = 30.0;
        let max_flip_len = spring_len * 3.0;
        let mut mesh = make_icosphere(spring_len * 4.0, 8);
        mesh.compute_intrinsic_lengths();

        // Simulate worst-case one-sided growth divergence: every even-indexed
        // half-edge saturated at the growth cap, its twin left untouched.
        for he in 0..mesh.next_half_edge {
            if mesh.half_edge_idx[he] < 0 { continue; }
            if he % 2 == 0 {
                mesh.half_edge_intrinsic_len[he] = max_flip_len;
            }
        }
        mesh.symmetrize_intrinsic_lengths();

        let max_before = mesh.half_edge_intrinsic_len[..mesh.next_half_edge]
            .iter().cloned().fold(0.0f32, f32::max);

        // Several refine passes: the old bug compounded across passes.
        for _ in 0..5 {
            mesh.refine_mesh(max_flip_len);
        }

        validate_mesh(&mesh);
        assert_no_double_edges(&mesh);
        let max_after = mesh.half_edge_intrinsic_len[..mesh.next_half_edge]
            .iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_after <= max_before.max(max_flip_len) + EPSILON,
            "refine minted a long edge: max {max_after} > {max_before}"
        );
    }

    #[test]
    fn symmetrize_averages_he_and_twin() {
        let mut mesh = make_icosphere(100.0, 4);
        mesh.compute_intrinsic_lengths();
        let he = (0..mesh.next_half_edge)
            .find(|&h| mesh.half_edge_idx[h] >= 0 && mesh.half_edge_twin[h] >= 0)
            .unwrap();
        let twin = mesh.half_edge_twin[he] as usize;
        mesh.half_edge_intrinsic_len[he] = 10.0;
        mesh.half_edge_intrinsic_len[twin] = 30.0;
        mesh.symmetrize_intrinsic_lengths();
        assert_eq!(mesh.half_edge_intrinsic_len[he], 20.0);
        assert_eq!(mesh.half_edge_intrinsic_len[twin], 20.0);
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
