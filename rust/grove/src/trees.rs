// CPU self-organising tree growth — port of selforganising.cpp +
// shadowpropagation.cpp + trees.h. A central tree grows via the basipetal
// (Q accumulation) / acropetal (vigour distribution) passes of Runions et al.;
// bud growth directions are biased toward less-shadowed space.
//
// The metamer graph is stored as an index-based arena (rather than the OF
// shared_ptr graph) so `Trees` is Send + Sync and can live behind a Mutex in
// the nannou model.

use nannou::prelude::*;

use crate::camera::WORLD;
use crate::gpu::Segment;

const MAX_METAMERS: usize = 12_000;
const MAX_PASS: usize = 10;

// Shadow grid is a downscaled copy of the world box (×0.25 → 160³ ≈ 16 MB).
const SHADOW_SCALE: f32 = 0.25;

#[derive(Clone)]
struct Metamer {
    terminal: Option<usize>,
    axillary: Option<usize>,
    parent:   Option<usize>,
    terminal_growth_dir: Vec3,
    axillary_dir:        Vec3,
    axillary_growth_dir: Vec3,
    terminal_q: f32,
    axillary_q: f32,
    width:      f32,
    length:     f32,
    pos:        Vec3,
    start_pos:  Vec3,
    direction:  Vec3,
    tree_idx:   usize,
}

impl Metamer {
    fn new() -> Self {
        Metamer {
            terminal: None, axillary: None, parent: None,
            terminal_growth_dir: Vec3::ZERO, axillary_dir: Vec3::ZERO, axillary_growth_dir: Vec3::ZERO,
            terminal_q: 0.0, axillary_q: 0.0,
            width: 0.5, length: 1.0,
            pos: Vec3::ZERO, start_pos: Vec3::ZERO, direction: Vec3::Y, tree_idx: 0,
        }
    }
}

#[derive(Clone)]
struct Tree {
    idx: usize,
    growth_weight: f32,
    default_weight: f32,
    tropism_weight: f32,
    tropism_dir: Vec3,
    initial_width: f32,
    base_length: f32,
    axillary_angle: f32,
    lambda: f32,
    v: f32,
    alpha: f32,
    perception_angle: f32,
    perception_factor: f32,
    branch_exp: f32,
    root: usize,
}

struct Shadows {
    data: Vec<f32>,
    w: usize, h: usize, d: usize,
    c: f32, a: f32, b: f32, qmax: i32,
}

impl Shadows {
    fn new(world: f32) -> Self {
        let w = (world * SHADOW_SCALE) as usize;
        let mut s = Shadows { data: vec![0.0; w * w * w], w, h: w, d: w, c: 1.0, a: 0.5, b: 1.1, qmax: 18 };
        let mut seed: u32 = 0x9e37_79b9;
        for v in s.data.iter_mut() {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            *v = (seed as f32 / u32::MAX as f32) * (s.c / 3.0);
        }
        s
    }

    fn at(&self, p: Vec3) -> f32 {
        let x = (p.x * SHADOW_SCALE) as i32;
        let y = (p.y * SHADOW_SCALE) as i32;
        let z = (p.z * SHADOW_SCALE) as i32;
        if x < 0 || y < 0 || z < 0 || x >= self.w as i32 || y >= self.h as i32 || z >= self.d as i32 { return self.c; }
        self.data[x as usize + y as usize * self.w + z as usize * self.w * self.h]
    }

    fn deposit(&mut self, pos: Vec3, sign: f32) {
        let (w, h, d) = (self.w as i32, self.h as i32, self.d as i32);
        let cx = (pos.x * SHADOW_SCALE) as i32;
        let cy = (pos.y * SHADOW_SCALE) as i32;
        let cz = (pos.z * SHADOW_SCALE) as i32;
        for q in 0..=self.qmax {
            let add = sign * self.a * self.b.powi(-q);
            for i in -q..=q {
                for k in -q..=q {
                    let x = (cx + i).clamp(0, w - 1);
                    let y = (cy - q).clamp(0, h - 1);
                    let z = (cz + k).clamp(0, d - 1);
                    self.data[x as usize + y as usize * self.w + z as usize * self.w * self.h] += add;
                }
            }
        }
    }

    /// (q, terminal_growth_dir, axillary_growth_dir) for a bud.
    fn bud_env(&self, m: &Metamer, tree: &Tree) -> (f32, Vec3, Vec3) {
        let pos = m.pos;
        let q = (self.c - self.at(pos) + self.a).max(0.0);

        let radius = ((tree.perception_factor * m.length * SHADOW_SCALE) as i32).clamp(1, 6);
        let (gw, gh, gd) = (self.w as i32, self.h as i32, self.d as i32);
        let cx = (pos.x * SHADOW_SCALE) as i32;
        let cy = (pos.y * SHADOW_SCALE) as i32;
        let cz = (pos.z * SHADOW_SCALE) as i32;

        let mut min_t = self.c; let mut dir_t = m.direction;
        let mut min_a = self.c; let mut dir_a = m.axillary_dir;
        let half = tree.perception_angle * 0.5;

        for i in -radius..=radius {
            for j in -radius..=radius {
                for k in -radius..=radius {
                    if i == 0 && j == 0 && k == 0 { continue; }
                    let x = cx + i; let y = cy + j; let z = cz + k;
                    if x < 0 || y < 0 || z < 0 || x >= gw || y >= gh || z >= gd { continue; }
                    let to = Vec3::new(i as f32, j as f32, k as f32);
                    let shadow = self.data[x as usize + y as usize * self.w + z as usize * self.w * self.h];
                    if angle_between(to, m.direction) < half && shadow < min_t {
                        min_t = shadow; dir_t = to.normalize_or_zero();
                    }
                    if angle_between(to, m.axillary_dir) < half && shadow < min_a {
                        min_a = shadow; dir_a = to.normalize_or_zero();
                    }
                }
            }
        }
        (q, dir_t, dir_a)
    }
}

fn angle_between(a: Vec3, b: Vec3) -> f32 {
    let la = a.length(); let lb = b.length();
    if la < 1e-5 || lb < 1e-5 { return PI; }
    (a.dot(b) / (la * lb)).clamp(-1.0, 1.0).acos()
}

fn rotate_around(v: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let axis = axis.normalize_or_zero();
    if axis.length() < 1e-5 { return v; }
    let (s, c) = angle.sin_cos();
    v * c + axis.cross(v) * s + axis * axis.dot(v) * (1.0 - c)
}

struct BasipetalState { idx: usize, tree_idx: usize, processed: bool }
struct AcropetalState { idx: usize, v: f32, tree_idx: usize }

pub struct Trees {
    env: Shadows,
    trees: Vec<Tree>,
    nodes: Vec<Metamer>,
    process_step: i32,
    basipetal: Vec<BasipetalState>,
    acropetal: Vec<AcropetalState>,
    rng: u32,
    grew: bool,
}

impl Trees {
    pub fn new() -> Self {
        let mut t = Trees {
            env: Shadows::new(WORLD),
            trees: Vec::new(),
            nodes: Vec::new(),
            process_step: 0,
            basipetal: Vec::new(),
            acropetal: Vec::new(),
            rng: 0xC0FF_EE11,
            grew: true,   // force an initial upload of the root segment
        };
        let pos = Vec3::new(WORLD * 0.5, 0.0, WORLD * 0.5);
        let tree = t.add_tree(pos, Vec3::Y, 0);
        t.trees.push(tree);
        t
    }

    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13; self.rng ^= self.rng >> 17; self.rng ^= self.rng << 5;
        self.rng as f32 / u32::MAX as f32
    }
    fn rand_range(&mut self, a: f32, b: f32) -> f32 { a + (b - a) * self.rand() }
    fn rand_unit(&mut self) -> Vec3 {
        Vec3::new(self.rand_range(-1.0, 1.0), self.rand_range(-1.0, 1.0), self.rand_range(-1.0, 1.0)).normalize_or_zero()
    }

    fn add_tree(&mut self, start_pos: Vec3, dir: Vec3, tree_idx: usize) -> Tree {
        let base_length = 10.0;
        let initial_width = 0.5;
        let axillary_angle = 0.2 * PI;
        let pos = start_pos + dir * base_length;

        let mut m = Metamer::new();
        m.length = base_length;
        m.pos = pos;
        m.start_pos = start_pos;
        m.direction = dir;
        m.width = initial_width;
        let rand_orth = self.rand_unit().cross(dir).normalize_or_zero();
        m.axillary_dir = rotate_around(dir, rand_orth, axillary_angle);

        let root = self.nodes.len();
        self.nodes.push(m);
        self.env.deposit(pos, 1.0);

        Tree {
            idx: tree_idx,
            growth_weight: 0.4, default_weight: 0.2, tropism_weight: 0.1,
            tropism_dir: Vec3::Y,
            initial_width, base_length, axillary_angle,
            lambda: 0.52, v: 0.0, alpha: 2.0,
            perception_angle: 0.3 * PI, perception_factor: 5.0, branch_exp: 2.0,
            root,
        }
    }

    pub fn update(&mut self, wind_strength: f32, wind_dir: Vec2, activity: f32) -> bool {
        self.grew = false;
        for tr in self.trees.iter_mut() {
            tr.tropism_dir = Vec3::new(wind_dir.x * wind_strength, tr.tropism_dir.y, wind_dir.y * wind_strength);
            tr.alpha = 2.0 * activity;
        }
        if self.nodes.len() >= MAX_METAMERS { return false; }

        match self.process_step {
            0 => { self.process_step = 1; }
            1 => {
                if self.basipetal.is_empty() { self.start_basipetal(); }
                self.basipetal_pass();
                if self.basipetal.is_empty() { self.process_step = 2; }
            }
            2 => {
                if self.acropetal.is_empty() { self.start_acropetal(); }
                self.acropetal_pass();
                if self.acropetal.is_empty() { self.process_step = 0; }
            }
            _ => { self.process_step = 0; }
        }
        self.grew
    }

    fn start_basipetal(&mut self) {
        for tr in &self.trees {
            self.basipetal.push(BasipetalState { idx: tr.root, tree_idx: tr.idx, processed: false });
        }
    }

    fn basipetal_pass(&mut self) {
        let mut count = 0;
        while count < MAX_PASS {
            let Some(top) = self.basipetal.last_mut() else { break; };
            if !top.processed {
                top.processed = true;
                let tree_idx = top.tree_idx;
                let idx = top.idx;
                let (term, axil) = (self.nodes[idx].terminal, self.nodes[idx].axillary);
                let mut pushed = false;
                if let Some(t) = term { self.basipetal.push(BasipetalState { idx: t, tree_idx, processed: false }); pushed = true; }
                if let Some(a) = axil { self.basipetal.push(BasipetalState { idx: a, tree_idx, processed: false }); pushed = true; }
                if pushed { continue; }
            }
            let state = self.basipetal.pop().unwrap();
            let tree = self.trees[state.tree_idx].clone();
            let idx = state.idx;

            let (term, axil) = (self.nodes[idx].terminal, self.nodes[idx].axillary);
            if term.is_none() || axil.is_none() {
                let (q, dt, da) = self.env.bud_env(&self.nodes[idx], &tree);
                let n = &mut self.nodes[idx];
                n.terminal_q = q; n.axillary_q = q;
                n.terminal_growth_dir = dt; n.axillary_growth_dir = da;
            }

            let mut sum_width_sq = 0.0;
            if let Some(t) = term {
                let (tq, aq, w) = (self.nodes[t].terminal_q, self.nodes[t].axillary_q, self.nodes[t].width);
                self.nodes[idx].terminal_q = tq + aq;
                sum_width_sq += w.powf(tree.branch_exp);
            }
            if let Some(a) = axil {
                let (tq, aq, w) = (self.nodes[a].terminal_q, self.nodes[a].axillary_q, self.nodes[a].width);
                self.nodes[idx].axillary_q = tq + aq;
                sum_width_sq += w.powf(tree.branch_exp);
            }
            self.nodes[idx].width = if sum_width_sq > 0.0 { sum_width_sq.powf(1.0 / tree.branch_exp) } else { tree.initial_width };

            if self.nodes[idx].parent.is_none() {
                let (tq, aq) = (self.nodes[idx].terminal_q, self.nodes[idx].axillary_q);
                let v = tree.alpha * (tq + aq);
                let tr = &mut self.trees[state.tree_idx];
                tr.v = v;
                if v > 1000.0 { tr.lambda = 0.46; tr.tropism_dir = Vec3::new(0.0, -1.0, 0.0); }
            }
            count += 1;
        }
    }

    fn start_acropetal(&mut self) {
        for tr in &self.trees {
            self.acropetal.push(AcropetalState { idx: tr.root, v: tr.v, tree_idx: tr.idx });
        }
    }

    fn acropetal_pass(&mut self) {
        let mut count = 0;
        while let Some(state) = self.acropetal.pop() {
            if count >= MAX_PASS { self.acropetal.push(state); break; }
            let tree = self.trees[state.tree_idx].clone();
            let n = &self.nodes[state.idx];
            let (term, axil, term_q, axil_q) = (n.terminal, n.axillary, n.terminal_q, n.axillary_q);

            let lambda = tree.lambda;
            let l_qm = lambda * term_q;
            let l_ql = (1.0 - lambda) * axil_q;
            let denom = l_qm + l_ql;
            let (vm, vl) = if denom > 1e-6 { (state.v * l_qm / denom, state.v * l_ql / denom) } else { (0.0, 0.0) };

            if let Some(t) = term {
                self.acropetal.push(AcropetalState { idx: t, v: vm, tree_idx: state.tree_idx });
            } else if vm >= 1.0 {
                self.add_terminal_shoot(state.idx, vm, state.tree_idx);
            }
            if let Some(a) = axil {
                self.acropetal.push(AcropetalState { idx: a, v: vl, tree_idx: state.tree_idx });
            } else if vl >= 1.0 {
                self.add_axillary_shoot(state.idx, vl, state.tree_idx);
            }
            count += 1;
        }
    }

    fn add_terminal_shoot(&mut self, idx: usize, v: f32, tree_idx: usize) {
        let (dir, growth) = (self.nodes[idx].direction, self.nodes[idx].terminal_growth_dir);
        self.add_shoot(idx, v, dir, growth, true, tree_idx);
    }
    fn add_axillary_shoot(&mut self, idx: usize, v: f32, tree_idx: usize) {
        let (dir, growth) = (self.nodes[idx].axillary_dir, self.nodes[idx].axillary_growth_dir);
        self.add_shoot(idx, v, dir, growth, false, tree_idx);
    }

    fn add_shoot(&mut self, idx: usize, v: f32, default_dir: Vec3, growth_dir: Vec3, mut is_terminal: bool, tree_idx: usize) {
        let tree = self.trees[tree_idx].clone();
        let n = v.floor() as i32;
        if n <= 0 { return; }
        let l = v / n as f32;

        let internal = self.nodes[idx].terminal.is_some() && self.nodes[idx].axillary.is_some();
        if internal { let p = self.nodes[idx].pos; self.env.deposit(p, -1.0); }

        let mut base = idx;
        for _ in 0..n {
            let dir = (tree.default_weight * default_dir
                     + tree.growth_weight * growth_dir
                     + tree.tropism_weight * tree.tropism_dir).normalize_or_zero();
            let base_pos = self.nodes[base].pos;
            let pos = base_pos + l * tree.base_length * dir;
            if pos.x < 0.0 || pos.x > WORLD || pos.y < 0.0 || pos.y > WORLD || pos.z < 0.0 || pos.z > WORLD { break; }
            let new_idx = self.add_metamer(base, pos, dir, is_terminal, tree_idx);
            base = new_idx;
            is_terminal = true;
        }
    }

    fn add_metamer(&mut self, parent: usize, pos: Vec3, dir: Vec3, is_terminal: bool, tree_idx: usize) -> usize {
        let tree = self.trees[tree_idx].clone();
        let start_pos = self.nodes[parent].pos;

        let mut m = Metamer::new();
        m.parent = Some(parent);
        m.tree_idx = tree_idx;
        m.pos = pos;
        m.start_pos = start_pos;
        m.direction = dir;
        m.length = (pos - start_pos).length();
        m.width = tree.initial_width;
        let rand_orth = self.rand_unit().cross(dir).normalize_or_zero();
        m.axillary_dir = rotate_around(dir, rand_orth, tree.axillary_angle);

        let new_idx = self.nodes.len();
        self.nodes.push(m);
        if is_terminal { self.nodes[parent].terminal = Some(new_idx); }
        else           { self.nodes[parent].axillary = Some(new_idx); }

        self.env.deposit(pos, 1.0);
        self.grew = true;
        new_idx
    }

    pub fn segments(&self) -> Vec<Segment> {
        self.nodes.iter().map(|n| {
            let tip_width = n.terminal.map(|t| self.nodes[t].width).unwrap_or(n.width);
            Segment {
                p0: [n.start_pos.x, n.start_pos.y, n.start_pos.z, n.width],
                p1: [n.pos.x, n.pos.y, n.pos.z, tip_width],
                color: [1.0, 1.0, 1.0, 1.0],
            }
        }).collect()
    }
}
