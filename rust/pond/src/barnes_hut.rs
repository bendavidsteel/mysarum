/// CPU Barnes-Hut quadtree with dual-tree traversal for ~O(N) repulsion.
/// Force computation matches znah/graphs: Morton-code sorting, flat DFS arrays,
/// skip pointers, tree-vs-tree traversal, parent-based downward propagation.
///
/// Force law: f = charge * dx / (1 + r²)  (no direction normalization)
/// This gives magnitude charge * r / (1 + r²): soft-cored, peaks at r=1, falls as 1/r.

const LEAF_SIZE: usize = 16;
const MAX_LEVEL: u32 = 10;

/// Dilate a 10-bit integer to occupy even bits (for 2D Morton code).
fn dilate2(mut x: u32) -> u32 {
    x &= 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    x
}

fn morton2d(x: u32, y: u32) -> u32 {
    dilate2(x) | (dilate2(y) << 1)
}

/// Minimum-image displacement for toroidal wrapping.
#[inline]
fn wrap_delta(d: f32, world_half: f32) -> f32 {
    if d > world_half { d - world_half * 2.0 }
    else if d < -world_half { d + world_half * 2.0 }
    else { d }
}

/// Brute-force O(N²) repulsion for verification / small N.
pub fn brute_force_repulsion(
    positions: &[(f32, f32)],
    charge: f32,
    max_dist: f32,
    epsilon: f32,
    world_half: f32,
) -> Vec<(f32, f32)> {
    let n = positions.len();
    let mut forces = vec![(0.0f32, 0.0f32); n];
    let max_dist2 = max_dist * max_dist;
    let eps2 = epsilon * epsilon;
    for i in 0..n {
        let (px, py) = positions[i];
        for j in (i + 1)..n {
            let dx = wrap_delta(positions[j].0 - px, world_half);
            let dy = wrap_delta(positions[j].1 - py, world_half);
            let d2 = dx * dx + dy * dy;
            if d2 < max_dist2 {
                let c = charge / (eps2 + d2);
                forces[i].0 -= c * dx;
                forces[i].1 -= c * dy;
                forces[j].0 += c * dx;
                forces[j].1 += c * dy;
            }
        }
    }
    forces
}

#[derive(Clone)]
pub struct BarnesHut {
    // Morton-sorted (code, original_index) pairs
    coded: Vec<(u32, usize)>,
    // Flat tree arrays in DFS order
    starts: Vec<usize>,
    ends: Vec<usize>,
    nexts: Vec<usize>,
    parents: Vec<usize>,
    com_x: Vec<f32>,
    com_y: Vec<f32>,
    extent: Vec<f32>,
    tree_extent: f32,
    // Pre-allocated work buffers for dual-tree traversal
    node_fx: Vec<f32>,
    node_fy: Vec<f32>,
    stack: Vec<(usize, usize)>,
}

impl BarnesHut {
    pub fn new() -> Self {
        Self {
            coded: Vec::new(),
            starts: Vec::new(),
            ends: Vec::new(),
            nexts: Vec::new(),
            parents: Vec::new(),
            com_x: Vec::new(),
            com_y: Vec::new(),
            extent: Vec::new(),
            tree_extent: 0.0,
            node_fx: Vec::new(),
            node_fy: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// Approximate leaf cell extent from previous frame (for jitter scaling).
    pub fn last_leaf_extent(&self) -> f32 {
        if self.tree_extent <= 0.0 { return 1.0; }
        // Typical leaf depth for N points: log4(N/LEAF_SIZE), clamped
        let n = self.coded.len().max(1) as f32;
        let depth = ((n / LEAF_SIZE as f32).log2() / 2.0).ceil().max(1.0).min(MAX_LEVEL as f32);
        self.tree_extent / (1u32 << depth as u32) as f32
    }

    /// Compute repulsion forces using dual-tree Barnes-Hut.
    /// `jitter` is a (jx, jy) offset applied to the tree origin each frame
    /// to randomize cell boundaries and prevent grid artifacts.
    /// `epsilon` is the soft-core radius for the force law: f = charge * dx / (eps² + r²).
    /// `world_half` enables toroidal minimum-image convention for wrapping.
    pub fn compute_repulsion(
        &mut self,
        positions: &[(f32, f32)],
        charge: f32,
        max_dist: f32,
        theta: f32,
        jitter: (f32, f32),
        epsilon: f32,
        world_half: f32,
    ) -> Vec<(f32, f32)> {
        let n = positions.len();
        if n == 0 {
            return Vec::new();
        }

        self.build(positions, jitter);
        self.dual_tree_forces(positions, charge, max_dist, theta * theta, epsilon, world_half)
    }

    fn build(&mut self, positions: &[(f32, f32)], jitter: (f32, f32)) {
        let n = positions.len();

        // Bounding box — centered around data centroid (matches znah)
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for &(x, y) in positions {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        let extent = (max_x - min_x).max(max_y - min_y).max(1.0);
        self.tree_extent = extent;

        // Center the bounding box + apply jitter to randomize cell boundaries
        let cx = (min_x + max_x) * 0.5;
        let cy = (min_y + max_y) * 0.5;
        let lo_x = cx - extent * 0.5 + jitter.0;
        let lo_y = cy - extent * 0.5 + jitter.1;

        // Morton codes
        let scale = 1023.0 / extent;
        self.coded.clear();
        self.coded.reserve(n);
        for (i, &(x, y)) in positions.iter().enumerate() {
            let mx = ((x - lo_x) * scale).clamp(0.0, 1023.0) as u32;
            let my = ((y - lo_y) * scale).clamp(0.0, 1023.0) as u32;
            self.coded.push((morton2d(mx, my), i));
        }
        self.coded.sort_unstable_by_key(|&(code, _)| code);

        // Build tree
        self.starts.clear();
        self.ends.clear();
        self.nexts.clear();
        self.parents.clear();
        self.com_x.clear();
        self.com_y.clear();
        self.extent.clear();
        self.build_node(0, n, 0, 0);

        // Compute centers of mass using parent-based accumulation (matches znah)
        let node_count = self.starts.len();

        // Zero centers
        self.com_x.iter_mut().for_each(|v| *v = 0.0);
        self.com_y.iter_mut().for_each(|v| *v = 0.0);

        // Reverse pass: leaves sum their points, then propagate up to parents
        for ni in (0..node_count).rev() {
            if self.is_leaf(ni) {
                for k in self.starts[ni]..self.ends[ni] {
                    let idx = self.coded[k].1;
                    self.com_x[ni] += positions[idx].0;
                    self.com_y[ni] += positions[idx].1;
                }
            }
            let p = self.parents[ni];
            if p != ni {
                self.com_x[p] += self.com_x[ni];
                self.com_y[p] += self.com_y[ni];
            }
        }

        // Divide by mass to get center of mass
        for ni in 0..node_count {
            let mass = (self.ends[ni] - self.starts[ni]) as f32;
            if mass > 0.0 {
                self.com_x[ni] /= mass;
                self.com_y[ni] /= mass;
            }
        }
    }

    fn build_node(&mut self, start: usize, end: usize, level: u32, parent: usize) {
        let ni = self.starts.len();
        self.starts.push(start);
        self.ends.push(end);
        self.nexts.push(0); // placeholder
        self.parents.push(parent);
        self.com_x.push(0.0);
        self.com_y.push(0.0);
        self.extent.push(self.tree_extent / (1u32 << level) as f32);

        let count = end - start;
        if count <= LEAF_SIZE || level >= MAX_LEVEL {
            self.nexts[ni] = self.starts.len();
            return;
        }

        // Split into 4 quadrants by Morton code bits at this level
        let shift = 2 * (MAX_LEVEL - 1 - level);
        let mut boundaries = [start; 5];
        let mut cursor = start;
        for q in 0..4u32 {
            while cursor < end && ((self.coded[cursor].0 >> shift) & 3) == q {
                cursor += 1;
            }
            boundaries[q as usize + 1] = cursor;
        }

        for q in 0..4 {
            let s = boundaries[q];
            let e = boundaries[q + 1];
            if s < e {
                self.build_node(s, e, level + 1, ni);
            }
        }

        self.nexts[ni] = self.starts.len();
    }

    #[inline]
    fn is_leaf(&self, ni: usize) -> bool {
        self.nexts[ni] == ni + 1
    }

    /// Dual-tree Barnes-Hut matching znah/graphs force computation.
    fn dual_tree_forces(
        &mut self,
        positions: &[(f32, f32)],
        charge: f32,
        max_dist: f32,
        theta2: f32,
        epsilon: f32,
        world_half: f32,
    ) -> Vec<(f32, f32)> {
        let n = positions.len();
        let node_count = self.starts.len();
        if node_count == 0 {
            return vec![(0.0, 0.0); n];
        }

        let max_dist2 = max_dist * max_dist;
        let eps2 = epsilon * epsilon;

        // Per-point forces (from direct leaf-leaf interactions)
        let mut forces = vec![(0.0f32, 0.0f32); n];

        // Per-node accumulated force-per-point (propagated downward after traversal)
        self.node_fx.clear();
        self.node_fx.resize(node_count, 0.0);
        self.node_fy.clear();
        self.node_fy.resize(node_count, 0.0);

        // Stack of (nodeA, nodeB) pairs, starting with root-vs-root
        self.stack.clear();
        self.stack.push((0, 0));

        while let Some((a, b)) = self.stack.pop() {
            let dx = wrap_delta(self.com_x[b] - self.com_x[a], world_half);
            let dy = wrap_delta(self.com_y[b] - self.com_y[a], world_half);
            let l2 = dx * dx + dy * dy;

            let w_a = self.extent[a];
            let w_b = self.extent[b];
            let combined_w = w_a + w_b;

            let leaf_a = self.is_leaf(a);
            let leaf_b = self.is_leaf(b);

            if a != b && combined_w * combined_w < theta2 * l2 {
                // Far enough: node-to-node interaction
                if l2 < max_dist2 {
                    let mass_a = (self.ends[a] - self.starts[a]) as f32;
                    let mass_b = (self.ends[b] - self.starts[b]) as f32;
                    let common = charge / (eps2 + l2);

                    // Repulsion: A pushed away from B
                    let ca = mass_b * common;
                    self.node_fx[a] -= ca * dx;
                    self.node_fy[a] -= ca * dy;

                    // Repulsion: B pushed away from A
                    let cb = mass_a * common;
                    self.node_fx[b] += cb * dx;
                    self.node_fy[b] += cb * dy;
                }
            } else if leaf_a && leaf_b {
                // Both leaves: direct point-to-point computation
                let start_a = self.starts[a];
                let end_a = self.ends[a];
                let j_start_base = self.starts[b];
                let end_b = self.ends[b];
                let is_self = a == b;

                for ki in start_a..end_a {
                    let i = self.coded[ki].1;
                    let (px, py) = positions[i];
                    let j_start = if is_self { ki + 1 } else { j_start_base };
                    for kj in j_start..end_b {
                        let j = self.coded[kj].1;
                        let pdx = wrap_delta(positions[j].0 - px, world_half);
                        let pdy = wrap_delta(positions[j].1 - py, world_half);
                        let pl2 = pdx * pdx + pdy * pdy;
                        if pl2 < max_dist2 {
                            let c = charge / (eps2 + pl2);
                            forces[i].0 -= c * pdx;
                            forces[i].1 -= c * pdy;
                            forces[j].0 += c * pdx;
                            forces[j].1 += c * pdy;
                        }
                    }
                }
            } else if a == b {
                // Self-interaction: enumerate all unique pairs of children
                let mut m = a + 1;
                while m < self.nexts[a] {
                    let mut nn = m;
                    while nn < self.nexts[a] {
                        self.stack.push((m, nn));
                        nn = self.nexts[nn];
                    }
                    m = self.nexts[m];
                }
            } else if !leaf_a && (leaf_b || w_a >= w_b) {
                // Split A
                let mut m = a + 1;
                while m < self.nexts[a] {
                    self.stack.push((m, b));
                    m = self.nexts[m];
                }
            } else {
                // Split B
                let mut m = b + 1;
                while m < self.nexts[b] {
                    self.stack.push((a, m));
                    m = self.nexts[m];
                }
            }
        }

        // Downward pass: propagate node forces using parent pointers (matches znah).
        // DFS order guarantees parent is processed before children.
        for ni in 1..node_count {
            let p = self.parents[ni];
            if p != ni {
                self.node_fx[ni] += self.node_fx[p];
                self.node_fy[ni] += self.node_fy[p];
            }
        }

        // Distribute leaf-node forces to their contained points
        for ni in 0..node_count {
            if !self.is_leaf(ni) {
                continue;
            }
            let fx = self.node_fx[ni];
            let fy = self.node_fy[ni];
            for k in self.starts[ni]..self.ends[ni] {
                let i = self.coded[k].1;
                forces[i].0 += fx;
                forces[i].1 += fy;
            }
        }

        forces
    }
}
