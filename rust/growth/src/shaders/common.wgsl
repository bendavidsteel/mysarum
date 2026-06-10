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
    state_dt: f32,
    damping: f32,
    growth_rate: f32,
    xpbd_iterations: u32,
    bending_compliance: f32,
    // SOR under-relaxation factor applied at the end of each XPBD projection pass.
    // 1.0 = full Jacobi step; < 1.0 damps the correction so chained passes
    // (springs → bending → collision) don't overshoot each other. ~0.5-0.8 is safe.
    relaxation: f32,
    // State rule: 0 = Lenia cellular automata, 1 = phototropism
    // (state = vertex normal · overhead light direction), 2 = grow-at-dot
    // (heat-equation diffusion of a growth potential from fixed source vertices).
    growth_mode: u32,
    // Per-frame seed mixed into the growth RNG (see growth.wgsl).
    frame_seed: u32,
    // Ground plane: when floor_enabled > 0.5, no vertex may move below
    // z == floor_z (used by the hemisphere's flat bottom cap).
    floor_enabled: f32,
    floor_z: f32,
    // Anisotropic growth: preferred axis (0 = X, 1 = Y, 2 = Z) and how strongly
    // it is favoured. At strength 0 growth is isotropic (every edge grows
    // equally); at strength 1 edges parallel to the axis grow fully and edges
    // perpendicular to it not at all. See the differential growth in growth.wgsl.
    anisotropy_axis: u32,
    anisotropy_strength: f32,
    // Grow-at-dot heat equation: g += dot_diffusion * laplacian(g) - dot_decay * g
    // (direct per-step rates). dot_diffusion sets how fast the potential spreads;
    // dot_decay sets the falloff radius ~sqrt(dot_diffusion / dot_decay) edges
    // (larger decay = sharper/smaller growth dot; decay 0 fills the whole mesh).
    dot_diffusion: f32,
    dot_decay: f32,
}

const EPSILON: f32 = 1e-6;

// Clamp a position above the ground plane (no-op when the floor is disabled).
fn apply_floor(p: vec3<f32>, params: SimParams) -> vec3<f32> {
    if params.floor_enabled > 0.5 {
        return vec3<f32>(p.x, p.y, max(p.z, params.floor_z));
    }
    return p;
}

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
