struct SimParams {
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
    _pad0: u32,
}

const EPSILON: f32 = 1e-6;

struct BinInfo {
    bin_index: u32,
    bin_x: u32,
    bin_y: u32,
}

fn get_bin_info(pos_x: f32, pos_y: f32, params: SimParams) -> BinInfo {
    let bx = u32(max(0.0, (pos_x - params.origin_x) / params.bin_size));
    let by = u32(max(0.0, (pos_y - params.origin_y) / params.bin_size));
    let cx = min(bx, params.num_bins_x - 1u);
    let cy = min(by, params.num_bins_y - 1u);
    return BinInfo(cy * params.num_bins_x + cx, cx, cy);
}
