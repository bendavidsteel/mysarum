use bytemuck::{Pod, Zeroable};
use nannou::prelude::*;

pub(crate) const MAX_NODES: usize = 4000;
pub(crate) const MAX_CONNECTIONS: usize = 12000;
const MAX_ADJ_ENTRIES: usize = MAX_CONNECTIONS * 2;
const MAX_BINS_PER_DIM: u32 = 64;
pub(crate) const WORKGROUP_SIZE: u32 = 64;
const MAX_CHEB_ORDER: usize = 20;
const COARSE_DIM: u32 = 16;
const COARSE_CELLS: u32 = COARSE_DIM * COARSE_DIM;

// Shader sources (loaded at compile time, common prepended)
const FILL_BINS_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/fill_bins.wgsl"));
const PREFIX_SUM_WGSL: &str = include_str!("shaders/prefix_sum.wgsl");
const SORT_NODES_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/sort_nodes.wgsl"));
const REPULSION_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/repulsion.wgsl"));
const COARSE_GRID_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/coarse_grid.wgsl"));
const FAR_REPULSION_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/far_repulsion.wgsl"));
const SPRING_FORCES_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/spring_forces.wgsl"));
const CHEB_INIT_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/chebyshev_init.wgsl"));
const CHEB_STEP_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/chebyshev_step.wgsl"));
const GROWTH_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/growth.wgsl"));
const INTEGRATE_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/integrate.wgsl"));
const BBOX_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/bbox.wgsl"));

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuSimParams {
    pub(crate) num_nodes: u32,
    pub(crate) num_connections: u32,
    pub(crate) repulsion_distance: f32,
    pub(crate) repulsion_strength: f32,
    pub(crate) spring_length: f32,
    pub(crate) spring_stiffness: f32,
    pub(crate) damping: f32,
    pub(crate) max_velocity: f32,
    pub(crate) growth_mu: f32,
    pub(crate) growth_sigma: f32,
    pub(crate) state_dt: f32,
    pub(crate) cheb_order: u32,
    pub(crate) origin_x: f32,
    pub(crate) origin_y: f32,
    pub(crate) bin_size: f32,
    pub(crate) num_bins_x: u32,
    pub(crate) num_bins_y: u32,
    pub(crate) far_repulsion_strength: f32,
    pub(crate) coarse_bin_size: f32,
    pub(crate) _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct RenderUniforms {
    pub(crate) min_x: f32,
    pub(crate) min_y: f32,
    pub(crate) max_x: f32,
    pub(crate) max_y: f32,
    pub(crate) node_radius: f32,
    pub(crate) num_nodes: u32,
    pub(crate) num_connections: u32,
    pub(crate) window_aspect: f32,
}

pub(crate) fn dispatch_count(n: u32) -> u32 {
    (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

#[derive(Clone)]
pub(crate) struct GpuCompute {
    // Node buffers
    pub(crate) node_pos_buf: wgpu::Buffer,
    node_vel_buf: wgpu::Buffer,
    node_force_buf: wgpu::Buffer,
    pub(crate) node_state_buf: wgpu::Buffer,
    node_u_buf: wgpu::Buffer,

    // Adjacency list buffers
    adj_offset_buf: wgpu::Buffer,
    adj_list_buf: wgpu::Buffer,

    // Connection buffer (for line rendering)
    pub(crate) connection_buf: wgpu::Buffer,

    // Chebyshev buffers
    cheb_a_buf: wgpu::Buffer,
    cheb_b_buf: wgpu::Buffer,
    cheb_c_buf: wgpu::Buffer,
    cheb_result_buf: wgpu::Buffer,
    cheb_c0_buf: wgpu::Buffer,
    cheb_c1_buf: wgpu::Buffer,
    cheb_coeff_bufs: Vec<wgpu::Buffer>,

    // Coarse grid buffers (far-field repulsion)
    coarse_sum_x_buf: wgpu::Buffer,
    coarse_sum_y_buf: wgpu::Buffer,
    coarse_count_buf: wgpu::Buffer,

    // Spatial hash buffers
    bin_size_buf: wgpu::Buffer,
    bin_offset_buf: wgpu::Buffer,
    bin_offset_tmp_buf: wgpu::Buffer,
    sorted_idx_buf: wgpu::Buffer,

    // Params
    sim_params_buf: wgpu::Buffer,

    // Readback
    pos_readback_buf: wgpu::Buffer,
    state_readback_buf: wgpu::Buffer,
    u_readback_buf: wgpu::Buffer,

    // Bounding box reduction
    bbox_atomic_buf: wgpu::Buffer,
    bbox_readback_buf: wgpu::Buffer,

    // Pipelines
    clear_bins_pipeline: wgpu::ComputePipeline,
    fill_bins_pipeline: wgpu::ComputePipeline,
    prefix_sum_pipeline: wgpu::ComputePipeline,
    sort_clear_pipeline: wgpu::ComputePipeline,
    sort_nodes_pipeline: wgpu::ComputePipeline,
    coarse_clear_pipeline: wgpu::ComputePipeline,
    coarse_accumulate_pipeline: wgpu::ComputePipeline,
    repulsion_pipeline: wgpu::ComputePipeline,
    far_repulsion_pipeline: wgpu::ComputePipeline,
    spring_forces_pipeline: wgpu::ComputePipeline,
    chebyshev_init_pipeline: wgpu::ComputePipeline,
    chebyshev_step_pipeline: wgpu::ComputePipeline,
    growth_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,
    bbox_clear_pipeline: wgpu::ComputePipeline,
    bbox_reduce_pipeline: wgpu::ComputePipeline,

    // Bind groups - spatial hash
    fill_bins_bg: [wgpu::BindGroup; 3],
    prefix_sum_bgs: Vec<wgpu::BindGroup>,
    sort_data_bg: wgpu::BindGroup,
    sort_params_bg: wgpu::BindGroup,

    // Bind groups - coarse grid
    coarse_data_bg: wgpu::BindGroup,
    coarse_params_bg: wgpu::BindGroup,

    // Bind groups - forces
    repulsion_data_bg: wgpu::BindGroup,
    repulsion_params_bg: wgpu::BindGroup,
    far_repulsion_data_bg: wgpu::BindGroup,
    far_repulsion_params_bg: wgpu::BindGroup,
    spring_data_bg: wgpu::BindGroup,
    spring_params_bg: wgpu::BindGroup,

    // Bind groups - chebyshev
    cheb_init_data_bg: wgpu::BindGroup,
    cheb_init_params_bg: wgpu::BindGroup,
    cheb_step_data_bgs: [wgpu::BindGroup; 3],
    cheb_step_params_bgs: Vec<wgpu::BindGroup>,

    // Bind groups - growth + integrate + bbox
    growth_data_bg: wgpu::BindGroup,
    growth_params_bg: wgpu::BindGroup,
    integrate_data_bg: wgpu::BindGroup,
    integrate_params_bg: wgpu::BindGroup,
    bbox_data_bg: wgpu::BindGroup,
    bbox_params_bg: wgpu::BindGroup,

    // Render buffers
    pub(crate) render_uniform_buf: wgpu::Buffer,

    // Config
    max_bins: u32,
    pub(crate) topology_dirty: bool,
    pub(crate) num_nodes: u32,
    pub(crate) num_connections: u32,
}

pub(crate) fn create_gpu_compute(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuCompute {
    let pos_buf_size = (MAX_NODES * 16) as u64;      // vec4<f32>
    let vel_buf_size = pos_buf_size;
    let force_buf_size = pos_buf_size;
    let state_buf_size = (MAX_NODES * 4) as u64;      // f32
    let adj_offset_size = ((MAX_NODES + 1) * 4) as u64; // u32
    let adj_list_size = (MAX_ADJ_ENTRIES * 4) as u64;
    let conn_buf_size = (MAX_CONNECTIONS * 2 * 4) as u64; // pairs of u32
    let max_bins = MAX_BINS_PER_DIM * MAX_BINS_PER_DIM;
    let bin_buf_size = ((max_bins + 1) * 4) as u64;
    let idx_buf_size = (MAX_NODES * 4) as u64;

    let storage_rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let storage_r = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;

    // Create buffers
    let node_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("node_pos"), size: pos_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let node_vel_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("node_vel"), size: vel_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let node_force_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("node_force"), size: force_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let node_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("node_state"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let node_u_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("node_u"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let adj_offset_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adj_offset"), size: adj_offset_size, usage: storage_r, mapped_at_creation: false,
    });
    let adj_list_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adj_list"), size: adj_list_size, usage: storage_r, mapped_at_creation: false,
    });
    let connection_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("connections"), size: conn_buf_size, usage: storage_r, mapped_at_creation: false,
    });

    // Chebyshev buffers
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
    let cheb_c0_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_c0"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let cheb_c1_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_c1"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let cheb_coeff_bufs: Vec<wgpu::Buffer> = (0..MAX_CHEB_ORDER)
        .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("cheb_coeff_{}", i)),
            size: 4,
            usage: uniform,
            mapped_at_creation: false,
        }))
        .collect();

    // Coarse grid buffers (256 cells each)
    let coarse_buf_size = (COARSE_CELLS * 4) as u64;
    let coarse_sum_x_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("coarse_sum_x"), size: coarse_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let coarse_sum_y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("coarse_sum_y"), size: coarse_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let coarse_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("coarse_count"), size: coarse_buf_size, usage: storage_rw, mapped_at_creation: false,
    });

    // Spatial hash buffers
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
    let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim_params"), size: std::mem::size_of::<GpuSimParams>() as u64, usage: uniform, mapped_at_creation: false,
    });
    let render_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_uniform"), size: std::mem::size_of::<RenderUniforms>() as u64, usage: uniform, mapped_at_creation: false,
    });
    let pos_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pos_readback"), size: pos_buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let state_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("state_readback"), size: state_buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let u_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("u_readback"), size: state_buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let bbox_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bbox_atomic"), size: 16, usage: storage_rw, mapped_at_creation: false,
    });
    let bbox_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bbox_readback"), size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });

    // Pre-create prefix sum step buffers
    let max_prefix_steps = ((max_bins + 1) as f32).log2().ceil() as usize;

    // Create shader modules
    let fill_bins_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fill_bins"), source: wgpu::ShaderSource::Wgsl(FILL_BINS_WGSL.into()),
    });
    let prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("prefix_sum"), source: wgpu::ShaderSource::Wgsl(PREFIX_SUM_WGSL.into()),
    });
    let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sort_nodes"), source: wgpu::ShaderSource::Wgsl(SORT_NODES_WGSL.into()),
    });
    let coarse_grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("coarse_grid"), source: wgpu::ShaderSource::Wgsl(COARSE_GRID_WGSL.into()),
    });
    let repulsion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("repulsion"), source: wgpu::ShaderSource::Wgsl(REPULSION_WGSL.into()),
    });
    let far_repulsion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("far_repulsion"), source: wgpu::ShaderSource::Wgsl(FAR_REPULSION_WGSL.into()),
    });
    let spring_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("spring_forces"), source: wgpu::ShaderSource::Wgsl(SPRING_FORCES_WGSL.into()),
    });
    let cheb_init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cheb_init"), source: wgpu::ShaderSource::Wgsl(CHEB_INIT_WGSL.into()),
    });
    let cheb_step_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cheb_step"), source: wgpu::ShaderSource::Wgsl(CHEB_STEP_WGSL.into()),
    });
    let growth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("growth"), source: wgpu::ShaderSource::Wgsl(GROWTH_WGSL.into()),
    });
    let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("integrate"), source: wgpu::ShaderSource::Wgsl(INTEGRATE_WGSL.into()),
    });
    let bbox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bbox"), source: wgpu::ShaderSource::Wgsl(BBOX_WGSL.into()),
    });

    let cs = wgpu::ShaderStages::COMPUTE;

    // ── Bind group layouts ──────────────────────────────────────────────────

    let fill_bins_pos_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)
        .build(device);
    let params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)
        .build(device);
    let bins_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)
        .build(device);

    let prefix_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)
        .storage_buffer(cs, false, false)
        .uniform_buffer(cs, false)
        .build(device);

    let sort_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // node_pos
        .storage_buffer(cs, false, false)  // sorted_idx
        .storage_buffer(cs, false, true)   // bin_offset
        .storage_buffer(cs, false, false)  // bin_counter
        .build(device);

    // Coarse grid: sum_x(RW) + sum_y(RW) + count(RW) + node_pos(R)
    let coarse_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // coarse_sum_x
        .storage_buffer(cs, false, false)  // coarse_sum_y
        .storage_buffer(cs, false, false)  // coarse_count
        .storage_buffer(cs, false, true)   // node_pos
        .build(device);

    let repulsion_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // node_pos
        .storage_buffer(cs, false, false)  // node_force
        .storage_buffer(cs, false, true)   // sorted_idx
        .storage_buffer(cs, false, true)   // bin_offset
        .build(device);

    // Far repulsion: sum_x(RW) + sum_y(RW) + count(RW) + node_pos(R) + node_force(RW)
    let far_repulsion_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // coarse_sum_x
        .storage_buffer(cs, false, false)  // coarse_sum_y
        .storage_buffer(cs, false, false)  // coarse_count
        .storage_buffer(cs, false, true)   // node_pos
        .storage_buffer(cs, false, false)  // node_force
        .build(device);

    // Spring forces: pos(R) + force(RW) + adj_offset(R) + adj_list(R)
    let spring_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // node_pos
        .storage_buffer(cs, false, false)  // node_force
        .storage_buffer(cs, false, true)   // adj_offset
        .storage_buffer(cs, false, true)   // adj_list
        .build(device);

    // Chebyshev init: state(R) + t_a(RW) + t_b(RW) + result(RW) + adj_offset(R) + adj_list(R)
    let cheb_init_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // node_state
        .storage_buffer(cs, false, false)  // t_a
        .storage_buffer(cs, false, false)  // t_b
        .storage_buffer(cs, false, false)  // result
        .storage_buffer(cs, false, true)   // adj_offset
        .storage_buffer(cs, false, true)   // adj_list
        .build(device);
    let cheb_init_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)  // params
        .uniform_buffer(cs, false)  // c0
        .uniform_buffer(cs, false)  // c1
        .build(device);

    // Chebyshev step: t_curr(R) + t_prev(R) + t_next(RW) + result(RW) + adj_offset(R) + adj_list(R)
    let cheb_step_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // t_curr
        .storage_buffer(cs, false, true)   // t_prev
        .storage_buffer(cs, false, false)  // t_next
        .storage_buffer(cs, false, false)  // result
        .storage_buffer(cs, false, true)   // adj_offset
        .storage_buffer(cs, false, true)   // adj_list
        .build(device);
    let cheb_step_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)  // params
        .uniform_buffer(cs, false)  // coeff
        .build(device);

    // Growth: state(RW) + u(RW) + result(R) + pos(R)
    let growth_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // node_state
        .storage_buffer(cs, false, false)  // node_u
        .storage_buffer(cs, false, true)   // result
        .storage_buffer(cs, false, true)   // node_pos
        .build(device);

    let integrate_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // node_pos
        .storage_buffer(cs, false, false)  // node_vel
        .storage_buffer(cs, false, true)   // node_force
        .build(device);

    let bbox_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // node_pos
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
    let coarse_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("coarse_pl"),
        bind_group_layouts: &[&coarse_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let repulsion_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("repulsion_pl"),
        bind_group_layouts: &[&repulsion_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let far_repulsion_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("far_repulsion_pl"),
        bind_group_layouts: &[&far_repulsion_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let spring_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("spring_pl"),
        bind_group_layouts: &[&spring_data_layout, &params_layout],
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
    let integrate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("integrate_pl"),
        bind_group_layouts: &[&integrate_data_layout, &params_layout],
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
    let sort_nodes_pipeline = make_pipeline("sort_nodes", &sort_pl, &sort_shader, "sort_nodes");
    let coarse_clear_pipeline = make_pipeline("coarse_clear", &coarse_pl, &coarse_grid_shader, "coarse_clear");
    let coarse_accumulate_pipeline = make_pipeline("coarse_accumulate", &coarse_pl, &coarse_grid_shader, "coarse_accumulate");
    let repulsion_pipeline = make_pipeline("repulsion", &repulsion_pl, &repulsion_shader, "main");
    let far_repulsion_pipeline = make_pipeline("far_repulsion", &far_repulsion_pl, &far_repulsion_shader, "main");
    let spring_forces_pipeline = make_pipeline("spring_forces", &spring_pl, &spring_shader, "main");
    let chebyshev_init_pipeline = make_pipeline("cheb_init", &cheb_init_pl, &cheb_init_shader, "main");
    let chebyshev_step_pipeline = make_pipeline("cheb_step", &cheb_step_pl, &cheb_step_shader, "main");
    let growth_pipeline = make_pipeline("growth", &growth_pl, &growth_shader, "main");
    let integrate_pipeline = make_pipeline("integrate", &integrate_pl, &integrate_shader, "main");
    let bbox_clear_pipeline = make_pipeline("bbox_clear", &bbox_pl, &bbox_shader, "bbox_clear");
    let bbox_reduce_pipeline = make_pipeline("bbox_reduce", &bbox_pl, &bbox_shader, "bbox_reduce");

    // ── Bind groups ─────────────────────────────────────────────────────────

    let fill_bins_bg = [
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&node_pos_buf, 0, None)
            .build(device, &fill_bins_pos_layout),
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&sim_params_buf, 0, None)
            .build(device, &params_layout),
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&bin_size_buf, 0, None)
            .build(device, &bins_layout),
    ];

    // Pre-create prefix sum bind groups (one per step with correct source/dest/step)
    let mut prefix_sum_bgs = Vec::with_capacity(max_prefix_steps);
    for i in 0..max_prefix_steps {
        let step_size = 1u32 << i;
        let step_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_step"), size: 4, usage: uniform, mapped_at_creation: false,
        });
        queue.write_buffer(&step_buf, 0, bytemuck::bytes_of(&step_size));

        let (src, dst) = if i == 0 {
            (&bin_size_buf, &bin_offset_buf)
        } else if i % 2 == 1 {
            (&bin_offset_buf, &bin_offset_tmp_buf)
        } else {
            (&bin_offset_tmp_buf, &bin_offset_buf)
        };

        let bg = wgpu::BindGroupBuilder::new()
            .buffer_bytes(src, 0, None)
            .buffer_bytes(dst, 0, None)
            .buffer_bytes(&step_buf, 0, None)
            .build(device, &prefix_layout);
        prefix_sum_bgs.push(bg);
    }

    let sort_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&node_pos_buf, 0, None)
        .buffer_bytes(&sorted_idx_buf, 0, None)
        .buffer_bytes(&bin_offset_buf, 0, None)
        .buffer_bytes(&bin_size_buf, 0, None) // reuse as counter
        .build(device, &sort_data_layout);
    let sort_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    let coarse_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&coarse_sum_x_buf, 0, None)
        .buffer_bytes(&coarse_sum_y_buf, 0, None)
        .buffer_bytes(&coarse_count_buf, 0, None)
        .buffer_bytes(&node_pos_buf, 0, None)
        .build(device, &coarse_data_layout);
    let coarse_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    let repulsion_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&node_pos_buf, 0, None)
        .buffer_bytes(&node_force_buf, 0, None)
        .buffer_bytes(&sorted_idx_buf, 0, None)
        .buffer_bytes(&bin_offset_buf, 0, None)
        .build(device, &repulsion_data_layout);
    let repulsion_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    let far_repulsion_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&coarse_sum_x_buf, 0, None)
        .buffer_bytes(&coarse_sum_y_buf, 0, None)
        .buffer_bytes(&coarse_count_buf, 0, None)
        .buffer_bytes(&node_pos_buf, 0, None)
        .buffer_bytes(&node_force_buf, 0, None)
        .build(device, &far_repulsion_data_layout);
    let far_repulsion_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    let spring_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&node_pos_buf, 0, None)
        .buffer_bytes(&node_force_buf, 0, None)
        .buffer_bytes(&adj_offset_buf, 0, None)
        .buffer_bytes(&adj_list_buf, 0, None)
        .build(device, &spring_data_layout);
    let spring_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // Chebyshev init bind groups
    let cheb_init_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&node_state_buf, 0, None)
        .buffer_bytes(&cheb_a_buf, 0, None)
        .buffer_bytes(&cheb_b_buf, 0, None)
        .buffer_bytes(&cheb_result_buf, 0, None)
        .buffer_bytes(&adj_offset_buf, 0, None)
        .buffer_bytes(&adj_list_buf, 0, None)
        .build(device, &cheb_init_data_layout);
    let cheb_init_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .buffer_bytes(&cheb_c0_buf, 0, None)
        .buffer_bytes(&cheb_c1_buf, 0, None)
        .build(device, &cheb_init_params_layout);

    // Chebyshev step bind groups (3 rotations: curr→prev→next)
    let cheb_step_data_bgs = [
        // [0]: curr=B, prev=A, next=C
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_result_buf, 0, None)
            .buffer_bytes(&adj_offset_buf, 0, None)
            .buffer_bytes(&adj_list_buf, 0, None)
            .build(device, &cheb_step_data_layout),
        // [1]: curr=C, prev=B, next=A
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_result_buf, 0, None)
            .buffer_bytes(&adj_offset_buf, 0, None)
            .buffer_bytes(&adj_list_buf, 0, None)
            .build(device, &cheb_step_data_layout),
        // [2]: curr=A, prev=C, next=B
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_result_buf, 0, None)
            .buffer_bytes(&adj_offset_buf, 0, None)
            .buffer_bytes(&adj_list_buf, 0, None)
            .build(device, &cheb_step_data_layout),
    ];
    let cheb_step_params_bgs: Vec<wgpu::BindGroup> = cheb_coeff_bufs.iter()
        .map(|coeff_buf| {
            wgpu::BindGroupBuilder::new()
                .buffer_bytes(&sim_params_buf, 0, None)
                .buffer_bytes(coeff_buf, 0, None)
                .build(device, &cheb_step_params_layout)
        })
        .collect();

    let growth_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&node_state_buf, 0, None)
        .buffer_bytes(&node_u_buf, 0, None)
        .buffer_bytes(&cheb_result_buf, 0, None)
        .buffer_bytes(&node_pos_buf, 0, None)
        .build(device, &growth_data_layout);
    let growth_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    let integrate_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&node_pos_buf, 0, None)
        .buffer_bytes(&node_vel_buf, 0, None)
        .buffer_bytes(&node_force_buf, 0, None)
        .build(device, &integrate_data_layout);
    let integrate_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    let bbox_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&node_pos_buf, 0, None)
        .buffer_bytes(&bbox_atomic_buf, 0, None)
        .build(device, &bbox_data_layout);
    let bbox_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    GpuCompute {
        node_pos_buf, node_vel_buf, node_force_buf, node_state_buf, node_u_buf,
        adj_offset_buf, adj_list_buf, connection_buf,
        cheb_a_buf, cheb_b_buf, cheb_c_buf, cheb_result_buf,
        cheb_c0_buf, cheb_c1_buf, cheb_coeff_bufs,
        coarse_sum_x_buf, coarse_sum_y_buf, coarse_count_buf,
        bin_size_buf, bin_offset_buf, bin_offset_tmp_buf, sorted_idx_buf,
        sim_params_buf,
        pos_readback_buf, state_readback_buf, u_readback_buf,
        bbox_atomic_buf, bbox_readback_buf,
        clear_bins_pipeline, fill_bins_pipeline, prefix_sum_pipeline,
        sort_clear_pipeline, sort_nodes_pipeline,
        coarse_clear_pipeline, coarse_accumulate_pipeline,
        repulsion_pipeline, far_repulsion_pipeline,
        spring_forces_pipeline,
        chebyshev_init_pipeline, chebyshev_step_pipeline,
        growth_pipeline, integrate_pipeline,
        bbox_clear_pipeline, bbox_reduce_pipeline,
        fill_bins_bg, prefix_sum_bgs,
        sort_data_bg, sort_params_bg,
        coarse_data_bg, coarse_params_bg,
        repulsion_data_bg, repulsion_params_bg,
        far_repulsion_data_bg, far_repulsion_params_bg,
        spring_data_bg, spring_params_bg,
        cheb_init_data_bg, cheb_init_params_bg,
        cheb_step_data_bgs, cheb_step_params_bgs,
        growth_data_bg, growth_params_bg,
        integrate_data_bg, integrate_params_bg,
        bbox_data_bg, bbox_params_bg,
        render_uniform_buf,
        max_bins,
        topology_dirty: true,
        num_nodes: 0,
        num_connections: 0,
    }
}

/// Upload node positions, states, velocities, adjacency, and connections to GPU.
pub(crate) fn upload_topology(
    queue: &wgpu::Queue,
    gpu: &mut GpuCompute,
    positions: &[(f32, f32)],
    states: &[f32],
    connections: &[(usize, usize)],
) {
    let n = positions.len().min(MAX_NODES);
    let num_conn = connections.len().min(MAX_CONNECTIONS);
    gpu.num_nodes = n as u32;
    gpu.num_connections = num_conn as u32;

    // Upload positions (vec4: x, y, 0, active=1)
    let pos_data: Vec<[f32; 4]> = positions[..n].iter()
        .map(|&(x, y)| [x, y, 0.0, 1.0])
        .collect();
    queue.write_buffer(&gpu.node_pos_buf, 0, bytemuck::cast_slice(&pos_data));

    // Upload states
    queue.write_buffer(&gpu.node_state_buf, 0, bytemuck::cast_slice(&states[..n]));

    // Clear velocities
    let zeros = vec![[0.0f32; 4]; n];
    queue.write_buffer(&gpu.node_vel_buf, 0, bytemuck::cast_slice(&zeros));

    // Build and upload adjacency list (only connections within valid node range)
    let mut neighbors: Vec<Vec<usize>> = vec![vec![]; n];
    for &(a, b) in &connections[..num_conn] {
        if a < n && b < n {
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    }
    let mut adj_offset = Vec::with_capacity(n + 1);
    let mut adj_list = Vec::new();
    let mut idx = 0u32;
    for i in 0..n {
        adj_offset.push(idx);
        for &nb in &neighbors[i] {
            adj_list.push(nb as u32);
            idx += 1;
        }
    }
    adj_offset.push(idx);
    queue.write_buffer(&gpu.adj_offset_buf, 0, bytemuck::cast_slice(&adj_offset));
    if !adj_list.is_empty() {
        queue.write_buffer(&gpu.adj_list_buf, 0, bytemuck::cast_slice(&adj_list));
    }

    // Upload connection pairs for line rendering
    let conn_flat: Vec<u32> = connections[..num_conn].iter()
        .filter(|&&(a, b)| a < n && b < n)
        .flat_map(|&(a, b)| [a as u32, b as u32])
        .collect();
    gpu.num_connections = (conn_flat.len() / 2) as u32;
    if !conn_flat.is_empty() {
        queue.write_buffer(&gpu.connection_buf, 0, bytemuck::cast_slice(&conn_flat));
    }
}

fn sortable_to_float(s: u32) -> f32 {
    let mask = if (s & 0x80000000) == 0 { 0xFFFFFFFFu32 } else { 0x80000000u32 };
    f32::from_bits(s ^ mask)
}

/// Read back only the bounding box (16 bytes).
pub(crate) fn readback_bbox_only(device: &wgpu::Device, gpu: &GpuCompute) -> [f32; 4] {
    let bbox_slice = gpu.bbox_readback_buf.slice(..);
    bbox_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait).unwrap();

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

/// Full readback: positions, states, and u values.
pub(crate) struct ReadbackData {
    pub(crate) positions: Vec<(f32, f32)>,
    pub(crate) states: Vec<f32>,
    pub(crate) u_values: Vec<f32>,
}

pub(crate) fn readback_full(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &GpuCompute, num_nodes: usize) -> ReadbackData {
    let pos_size = (num_nodes * 16) as u64;
    let state_size = (num_nodes * 4) as u64;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(&gpu.node_pos_buf, 0, &gpu.pos_readback_buf, 0, pos_size);
    encoder.copy_buffer_to_buffer(&gpu.node_state_buf, 0, &gpu.state_readback_buf, 0, state_size);
    encoder.copy_buffer_to_buffer(&gpu.node_u_buf, 0, &gpu.u_readback_buf, 0, state_size);
    queue.submit(Some(encoder.finish()));

    let pos_slice = gpu.pos_readback_buf.slice(..pos_size);
    pos_slice.map_async(wgpu::MapMode::Read, |_| {});
    let state_slice = gpu.state_readback_buf.slice(..state_size);
    state_slice.map_async(wgpu::MapMode::Read, |_| {});
    let u_slice = gpu.u_readback_buf.slice(..state_size);
    u_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait).unwrap();

    let positions = {
        let data = pos_slice.get_mapped_range();
        let floats: &[[f32; 4]] = bytemuck::cast_slice(&data);
        floats[..num_nodes].iter().map(|f| (f[0], f[1])).collect()
    };
    gpu.pos_readback_buf.unmap();

    let states = {
        let data = state_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        floats[..num_nodes].to_vec()
    };
    gpu.state_readback_buf.unmap();

    let u_values = {
        let data = u_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        floats[..num_nodes].to_vec()
    };
    gpu.u_readback_buf.unmap();

    ReadbackData { positions, states, u_values }
}

pub(crate) fn update_render_uniforms(queue: &wgpu::Queue, gpu: &GpuCompute, uniforms: &RenderUniforms) {
    queue.write_buffer(&gpu.render_uniform_buf, 0, bytemuck::bytes_of(uniforms));
}

pub(crate) fn gpu_dispatch_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &mut GpuCompute,
    params: &GpuSimParams,
    cheb_order: usize,
    cheb_coeffs: &[f32; 20],
) {
    let n = params.num_nodes;
    let num_bins_total = params.num_bins_x * params.num_bins_y + 1;

    // Upload sim params
    queue.write_buffer(&gpu.sim_params_buf, 0, bytemuck::bytes_of(params));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute_encoder"),
    });

    // ── Spatial hash (5 passes) ─────────────────────────────────────────

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

    // Prefix sum
    let num_prefix_steps = (num_bins_total as f32).log2().ceil() as u32;
    for i in 0..num_prefix_steps {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prefix_sum"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.prefix_sum_pipeline);
        pass.set_bind_group(0, &gpu.prefix_sum_bgs[i as usize], &[]);
        pass.dispatch_workgroups(dispatch_count(num_bins_total), 1, 1);
    }
    // If even number of steps > 1, result is in tmp; copy back
    if num_prefix_steps > 1 && num_prefix_steps % 2 == 0 {
        encoder.copy_buffer_to_buffer(
            &gpu.bin_offset_tmp_buf, 0,
            &gpu.bin_offset_buf, 0,
            (num_bins_total * 4) as u64,
        );
    }

    // Clear bin counters for sort
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_clear"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.sort_clear_pipeline);
        pass.set_bind_group(0, &gpu.sort_data_bg, &[]);
        pass.set_bind_group(1, &gpu.sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(num_bins_total), 1, 1);
    }

    // Sort nodes by bin
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_nodes"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.sort_nodes_pipeline);
        pass.set_bind_group(0, &gpu.sort_data_bg, &[]);
        pass.set_bind_group(1, &gpu.sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // ── Coarse grid (far-field summary) ─────────────────────────────────

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("coarse_clear"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.coarse_clear_pipeline);
        pass.set_bind_group(0, &gpu.coarse_data_bg, &[]);
        pass.set_bind_group(1, &gpu.coarse_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(COARSE_CELLS), 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("coarse_accumulate"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.coarse_accumulate_pipeline);
        pass.set_bind_group(0, &gpu.coarse_data_bg, &[]);
        pass.set_bind_group(1, &gpu.coarse_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // ── Forces ──────────────────────────────────────────────────────────

    // Repulsion near-field (initializes force buffer)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("repulsion"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.repulsion_pipeline);
        pass.set_bind_group(0, &gpu.repulsion_data_bg, &[]);
        pass.set_bind_group(1, &gpu.repulsion_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Far-field repulsion (adds to force buffer from coarse grid)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("far_repulsion"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.far_repulsion_pipeline);
        pass.set_bind_group(0, &gpu.far_repulsion_data_bg, &[]);
        pass.set_bind_group(1, &gpu.far_repulsion_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Spring forces (adds to force buffer)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("spring_forces"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.spring_forces_pipeline);
        pass.set_bind_group(0, &gpu.spring_data_bg, &[]);
        pass.set_bind_group(1, &gpu.spring_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Integrate (velocity update + position update)
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

    // Init: T_0 = state → cheb_a, T_1 = W(state) → cheb_b, result = c0*T0 + c1*T1
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
        if k < gpu.cheb_coeff_bufs.len() {
            queue.write_buffer(&gpu.cheb_coeff_bufs[k], 0, bytemuck::bytes_of(&cheb_coeffs[k]));
        }
    }
    for k in 2..cheb_order {
        let bg_idx = (k - 2) % 3;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cheb_step"), timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.chebyshev_step_pipeline);
            pass.set_bind_group(0, &gpu.cheb_step_data_bgs[bg_idx], &[]);
            pass.set_bind_group(1, &gpu.cheb_step_params_bgs[k], &[]);
            pass.dispatch_workgroups(dispatch_count(n), 1, 1);
        }
    }

    // Growth function (state update from Chebyshev result)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("growth"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.growth_pipeline);
        pass.set_bind_group(0, &gpu.growth_data_bg, &[]);
        pass.set_bind_group(1, &gpu.growth_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // ── Bounding box reduction ──────────────────────────────────────────

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

    // Copy bbox to readback
    encoder.copy_buffer_to_buffer(&gpu.bbox_atomic_buf, 0, &gpu.bbox_readback_buf, 0, 16);

    queue.submit(Some(encoder.finish()));
}
