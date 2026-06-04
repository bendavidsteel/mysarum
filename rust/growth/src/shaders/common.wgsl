struct SimParams {
    num_vertices: u32,
    num_half_edges: u32,
    repulsion_distance: f32,
    spring_len: f32,
    compliance: f32,
    bulge_strength: f32,
    smoothing_strength: f32,
    dt: f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    bin_size: f32,
    num_bins_x: u32,
    num_bins_y: u32,
    num_bins_z: u32,
    growth_mu: f32,
    growth_sigma: f32,
    cheb_order: u32,
    repulsion_strength: f32,
    state_dt: f32,
    damping: f32,
    growth_rate: f32,
    xpbd_iterations: u32,
    bending_compliance: f32,
    // SOR under-relaxation factor applied at the end of each XPBD projection pass.
    // 1.0 = full Jacobi step; < 1.0 damps the correction so chained passes
    // (springs → bending → collision) don't overshoot each other. ~0.5-0.8 is safe.
    relaxation: f32,
    // State rule: 0 = Lenia cellular automata, 1 = static dot seed.
    growth_mode: u32,
    // Per-frame seed mixed into the growth RNG (see growth.wgsl).
    frame_seed: u32,
    // Padding to keep Rust/WGSL struct sizes in sync (16-byte alignment).
    _pad2: f32,
}

const EPSILON: f32 = 1e-6;

struct BinInfo {
    bin_index: u32,
    bin_x: u32,
    bin_y: u32,
    bin_z: u32,
}

fn get_bin_info(pos_x: f32, pos_y: f32, pos_z: f32, params: SimParams) -> BinInfo {
    let bx = u32(max(0.0, (pos_x - params.origin_x) / params.bin_size));
    let by = u32(max(0.0, (pos_y - params.origin_y) / params.bin_size));
    let bz = u32(max(0.0, (pos_z - params.origin_z) / params.bin_size));
    let cx = min(bx, params.num_bins_x - 1u);
    let cy = min(by, params.num_bins_y - 1u);
    let cz = min(bz, params.num_bins_z - 1u);
    let idx = (cz * params.num_bins_y + cy) * params.num_bins_x + cx;
    return BinInfo(idx, cx, cy, cz);
}
