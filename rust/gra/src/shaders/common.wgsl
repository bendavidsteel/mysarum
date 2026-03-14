struct SimParams {
    num_nodes: u32,
    num_connections: u32,
    spring_length: f32,
    spring_stiffness: f32,
    damping: f32,
    max_velocity: f32,
    growth_mu: f32,
    growth_sigma: f32,
    state_dt: f32,
    cheb_order: u32,
    _pad0: f32,
    _pad1: f32,
}

const EPSILON: f32 = 1e-6;
