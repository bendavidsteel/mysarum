struct SimParams {
    num_nodes: u32,
    num_connections: u32,
    spring_length: f32,
    spring_stiffness: f32,
    damping: f32,
    max_velocity: f32,
    state_dt: f32,
    cheb_order: u32,
    num_channels: u32,
    world_half: f32,
    _pad1: u32,
    _pad2: u32,
    growth_mu: vec4<f32>,
    growth_sigma: vec4<f32>,
    coupling_row0: vec4<f32>,
    coupling_row1: vec4<f32>,
    coupling_row2: vec4<f32>,
}

struct NodeGrowthParams {
    growth_mu: vec4<f32>,
    growth_sigma: vec4<f32>,
    coupling_row0: vec4<f32>,
    coupling_row1: vec4<f32>,
    coupling_row2: vec4<f32>,
    state_dt: f32,
    num_channels: u32,
    _pad0: u32,
    _pad1: u32,
}

const EPSILON: f32 = 1e-6;
