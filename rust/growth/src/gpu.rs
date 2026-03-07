use bytemuck::{Pod, Zeroable};
use nannou::prelude::*;

use crate::mesh::{HalfEdgeMesh, MAX_VERTICES, MAX_HALF_EDGES, MAX_FACES};

const MAX_BINS_PER_DIM: u32 = 64;
pub(crate) const WORKGROUP_SIZE: u32 = 64;

// Shader sources (loaded at compile time, common prepended)
const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");
const FILL_BINS_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/fill_bins.wgsl"));
const PREFIX_SUM_WGSL: &str = include_str!("shaders/prefix_sum.wgsl");
const SORT_VERTICES_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/sort_vertices.wgsl"));
const REPULSION_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/repulsion.wgsl"));
const TOPO_FORCES_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/topo_forces.wgsl"));
const INTEGRATE_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/integrate.wgsl"));
const CHEBYSHEV_INIT_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/chebyshev_init.wgsl"));
const CHEBYSHEV_STEP_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/chebyshev_step.wgsl"));
const GROWTH_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/growth.wgsl"));
const BBOX_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/bbox.wgsl"));

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuSimParams {
    pub(crate) num_vertices: u32,
    pub(crate) num_half_edges: u32,
    pub(crate) repulsion_distance: f32,
    pub(crate) spring_len: f32,
    pub(crate) elastic_constant: f32,
    pub(crate) bulge_strength: f32,
    pub(crate) planar_strength: f32,
    pub(crate) dt: f32,
    pub(crate) origin_x: f32,
    pub(crate) origin_y: f32,
    pub(crate) origin_z: f32,
    pub(crate) bin_size: f32,
    pub(crate) num_bins_x: u32,
    pub(crate) num_bins_y: u32,
    pub(crate) num_bins_z: u32,
    pub(crate) growth_mu: f32,
    pub(crate) growth_sigma: f32,
    pub(crate) cheb_order: u32,
    pub(crate) repulsion_strength: f32,
    pub(crate) state_dt: f32,
    pub(crate) damping: f32,
    pub(crate) _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct RenderUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) center: [f32; 4],       // xyz = mesh center, w = scale
    pub(crate) light: [f32; 4],        // xyz = direction, w = ambient
    pub(crate) render_mode: [f32; 4],  // x = mode, yzw unused
}

pub(crate) fn dispatch_count(n: u32) -> u32 {
    (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

#[derive(Clone)]
pub(crate) struct GpuCompute {
    // Vertex buffers
    pub(crate) vertex_pos_buf: wgpu::Buffer,
    vertex_force_buf: wgpu::Buffer,

    // Spatial hash buffers
    bin_size_buf: wgpu::Buffer,
    bin_offset_buf: wgpu::Buffer,
    bin_offset_tmp_buf: wgpu::Buffer,
    sorted_idx_buf: wgpu::Buffer,
    prefix_sum_step_buf: wgpu::Buffer,

    // Topology buffers (packed: vec4<i32> = dest, twin, next, face)
    he_packed_buf: wgpu::Buffer,
    vertex_he_buf: wgpu::Buffer,

    // State evolution buffers
    pub(crate) vertex_state_buf: wgpu::Buffer,
    vertex_u_buf: wgpu::Buffer,
    cheb_a_buf: wgpu::Buffer,
    cheb_b_buf: wgpu::Buffer,
    cheb_c_buf: wgpu::Buffer,
    cheb_result_buf: wgpu::Buffer,
    cheb_coeff_buf: wgpu::Buffer,  // single f32 uniform
    cheb_c0_buf: wgpu::Buffer,    // uniform for init
    cheb_c1_buf: wgpu::Buffer,    // uniform for init

    // Params
    sim_params_buf: wgpu::Buffer,

    // Readback
    pos_readback_buf: wgpu::Buffer,
    state_readback_buf: wgpu::Buffer,

    // Bounding box reduction
    bbox_atomic_buf: wgpu::Buffer,
    bbox_readback_buf: wgpu::Buffer,
    bbox_clear_pipeline: wgpu::ComputePipeline,
    bbox_reduce_pipeline: wgpu::ComputePipeline,
    bbox_data_bg: wgpu::BindGroup,
    bbox_params_bg: wgpu::BindGroup,

    // Pipelines
    clear_bins_pipeline: wgpu::ComputePipeline,
    fill_bins_pipeline: wgpu::ComputePipeline,
    prefix_sum_pipeline: wgpu::ComputePipeline,
    sort_clear_pipeline: wgpu::ComputePipeline,
    sort_vertices_pipeline: wgpu::ComputePipeline,
    repulsion_pipeline: wgpu::ComputePipeline,
    topo_forces_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,
    chebyshev_init_pipeline: wgpu::ComputePipeline,
    chebyshev_step_pipeline: wgpu::ComputePipeline,
    growth_pipeline: wgpu::ComputePipeline,

    // Bind groups - spatial hash
    fill_bins_bg: [wgpu::BindGroup; 3],  // [pos, params, bins]
    prefix_sum_bgs: [wgpu::BindGroup; 3], // ping-pong: [0]bin_size→offset, [1]offset→tmp, [2]tmp→offset
    sort_data_bg: wgpu::BindGroup,
    sort_params_bg: wgpu::BindGroup,

    // Bind groups - forces
    repulsion_data_bg: wgpu::BindGroup,
    repulsion_params_bg: wgpu::BindGroup,
    topo_data_bg: wgpu::BindGroup,
    topo_params_bg: wgpu::BindGroup,
    integrate_data_bg: wgpu::BindGroup,
    integrate_params_bg: wgpu::BindGroup,

    // Bind groups - chebyshev
    cheb_init_data_bg: wgpu::BindGroup,
    cheb_init_params_bg: wgpu::BindGroup,
    cheb_step_data_bgs: [wgpu::BindGroup; 3], // rotating A→B→C
    cheb_step_params_bgs: Vec<wgpu::BindGroup>,
    growth_data_bg: wgpu::BindGroup,
    growth_params_bg: wgpu::BindGroup,

    // Render buffers (read by vertex shader directly)
    pub(crate) render_index_buf: wgpu::Buffer,
    pub(crate) render_uniform_buf: wgpu::Buffer,
    pub(crate) num_render_tris: u32,

    // Config
    max_bins: u32,
    pub(crate) topology_dirty: bool,

    // Pre-created prefix sum step buffers (one per possible step)
    prefix_sum_step_bufs: Vec<wgpu::Buffer>,

    // Pre-created chebyshev coefficient buffers (one per possible order)
    cheb_coeff_bufs: Vec<wgpu::Buffer>,

    // Reusable staging buffers
    pos_staging: Vec<[f32; 4]>,
    he_staging: Vec<[i32; 4]>,
    index_staging: Vec<u32>,
}

pub(crate) fn create_gpu_compute(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuCompute {
    let pos_buf_size = (MAX_VERTICES * 16) as u64;   // vec4<f32>
    let force_buf_size = pos_buf_size;
    let max_bins = MAX_BINS_PER_DIM * MAX_BINS_PER_DIM * MAX_BINS_PER_DIM;
    let bin_buf_size = ((max_bins + 1) * 4) as u64;   // u32 per bin + 1
    let idx_buf_size = (MAX_VERTICES * 4) as u64;      // u32 per vertex
    let he_packed_size = (MAX_HALF_EDGES * 16) as u64; // vec4<i32>
    let state_buf_size = (MAX_VERTICES * 4) as u64;    // f32 per vertex

    let storage_rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let storage_r = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;

    // Create buffers
    let vertex_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_pos"), size: pos_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let vertex_force_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_force"), size: force_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
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
    let prefix_sum_step_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prefix_sum_step"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let he_packed_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("he_packed"), size: he_packed_size, usage: storage_r, mapped_at_creation: false,
    });
    let vertex_he_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_he"), size: idx_buf_size, usage: storage_r, mapped_at_creation: false,
    });
    let vertex_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_state"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let vertex_u_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_u"), size: state_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
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
    let cheb_coeff_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_coeff"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let cheb_c0_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_c0"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let cheb_c1_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_c1"), size: 4, usage: uniform, mapped_at_creation: false,
    });
    let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim_params"), size: std::mem::size_of::<GpuSimParams>() as u64, usage: uniform, mapped_at_creation: false,
    });
    // Render buffers: index buffer (3 u32 per face) + uniform buffer
    let render_index_buf_size = (MAX_FACES * 3 * 4) as u64;
    let render_index_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_index"), size: render_index_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let render_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_uniform"), size: std::mem::size_of::<RenderUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pos_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pos_readback"), size: pos_buf_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let state_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("state_readback"), size: state_buf_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let bbox_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bbox_atomic"), size: 24, usage: storage_rw, mapped_at_creation: false,
    });
    let bbox_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bbox_readback"), size: 24, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });

    // Pre-create prefix sum step buffers (max 16 steps for 256*256 bins)
    let max_prefix_steps = ((max_bins + 1) as f32).log2().ceil() as usize;
    let mut prefix_sum_step_bufs = Vec::with_capacity(max_prefix_steps);
    for i in 0..max_prefix_steps {
        let step_size = 1u32 << i;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_sum_step_pre"),
            size: 4,
            usage: uniform,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&step_size));
        prefix_sum_step_bufs.push(buf);
    }

    // Pre-create chebyshev coefficient buffers (max 20)
    let cheb_coeff_bufs: Vec<wgpu::Buffer> = (0..20)
        .map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cheb_coeff_pre"),
                size: 4,
                usage: uniform,
                mapped_at_creation: false,
            })
        })
        .collect();

    // Create shader modules
    let fill_bins_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fill_bins"), source: wgpu::ShaderSource::Wgsl(FILL_BINS_WGSL.into()),
    });
    let prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("prefix_sum"), source: wgpu::ShaderSource::Wgsl(PREFIX_SUM_WGSL.into()),
    });
    let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sort_vertices"), source: wgpu::ShaderSource::Wgsl(SORT_VERTICES_WGSL.into()),
    });
    let repulsion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("repulsion"), source: wgpu::ShaderSource::Wgsl(REPULSION_WGSL.into()),
    });
    let topo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("topo_forces"), source: wgpu::ShaderSource::Wgsl(TOPO_FORCES_WGSL.into()),
    });
    let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("integrate"), source: wgpu::ShaderSource::Wgsl(INTEGRATE_WGSL.into()),
    });
    let cheb_init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cheb_init"), source: wgpu::ShaderSource::Wgsl(CHEBYSHEV_INIT_WGSL.into()),
    });
    let cheb_step_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cheb_step"), source: wgpu::ShaderSource::Wgsl(CHEBYSHEV_STEP_WGSL.into()),
    });
    let growth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("growth"), source: wgpu::ShaderSource::Wgsl(GROWTH_WGSL.into()),
    });
    let bbox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bbox"), source: wgpu::ShaderSource::Wgsl(BBOX_WGSL.into()),
    });

    let cs = wgpu::ShaderStages::COMPUTE;

    // ── Bind group layouts ──────────────────────────────────────────────────

    // fill_bins: group0=pos(R), group1=params(U), group2=bins(RW)
    let fill_bins_pos_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)
        .build(device);
    let params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)
        .build(device);
    let bins_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)
        .build(device);

    // prefix_sum: source(R), dest(RW), step(U)
    let prefix_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)
        .storage_buffer(cs, false, false)
        .uniform_buffer(cs, false)
        .build(device);

    // sort: group0=pos(R)+sorted(RW)+offset(R)+counter(RW), group1=params(U)
    let sort_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_pos
        .storage_buffer(cs, false, false)  // sorted_idx
        .storage_buffer(cs, false, true)   // bin_offset
        .storage_buffer(cs, false, false)  // bin_counter (reuse bin_size)
        .build(device);

    // repulsion: group0=pos(R)+force(RW)+sorted(R)+offset(R), group1=params(U)
    let repulsion_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_pos
        .storage_buffer(cs, false, false)  // vertex_force
        .storage_buffer(cs, false, true)   // sorted_idx
        .storage_buffer(cs, false, true)   // bin_offset
        .build(device);

    // topo_forces: group0=pos(R)+force(RW)+he_packed(R)+vertex_he(R), group1=params(U)
    let topo_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_pos
        .storage_buffer(cs, false, false)  // vertex_force
        .storage_buffer(cs, false, true)   // he_packed
        .storage_buffer(cs, false, true)   // vertex_he
        .build(device);

    // integrate: group0=pos(RW)+force(R), group1=params(U)
    let integrate_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // vertex_pos (RW)
        .storage_buffer(cs, false, true)   // vertex_force (R)
        .build(device);

    // chebyshev_init: group0=state(R)+t_a(RW)+t_b(RW)+result(RW)+he_packed(R)+vertex_he(R)
    //                 group1=params(U)+c0(U)+c1(U)
    let cheb_init_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_state
        .storage_buffer(cs, false, false)  // t_a
        .storage_buffer(cs, false, false)  // t_b
        .storage_buffer(cs, false, false)  // result
        .storage_buffer(cs, false, true)   // he_packed
        .storage_buffer(cs, false, true)   // vertex_he
        .build(device);
    let cheb_init_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)  // params
        .uniform_buffer(cs, false)  // c0
        .uniform_buffer(cs, false)  // c1
        .build(device);

    // chebyshev_step: group0=t_curr(R)+t_prev(R)+t_next(RW)+result(RW)+he_packed(R)+vertex_he(R)
    //                 group1=params(U)+coeff(U)
    let cheb_step_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // t_curr
        .storage_buffer(cs, false, true)   // t_prev
        .storage_buffer(cs, false, false)  // t_next
        .storage_buffer(cs, false, false)  // result
        .storage_buffer(cs, false, true)   // he_packed
        .storage_buffer(cs, false, true)   // vertex_he
        .build(device);
    let cheb_step_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)  // params
        .uniform_buffer(cs, false)  // coeff
        .build(device);

    // growth: group0=state(RW)+u(RW)+result(R)+pos(R), group1=params(U)
    let growth_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // vertex_state
        .storage_buffer(cs, false, false)  // vertex_u
        .storage_buffer(cs, false, true)   // result
        .storage_buffer(cs, false, true)   // vertex_pos (active check)
        .build(device);

    // bbox: group0=pos(R)+bbox_atomic(RW), group1=params(U)
    let bbox_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // vertex_pos
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
    let repulsion_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("repulsion_pl"),
        bind_group_layouts: &[&repulsion_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let topo_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("topo_pl"),
        bind_group_layouts: &[&topo_data_layout, &params_layout],
        push_constant_ranges: &[],
    });
    let integrate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("integrate_pl"),
        bind_group_layouts: &[&integrate_data_layout, &params_layout],
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
    let sort_vertices_pipeline = make_pipeline("sort_vertices", &sort_pl, &sort_shader, "sort_vertices");
    let repulsion_pipeline = make_pipeline("repulsion", &repulsion_pl, &repulsion_shader, "main");
    let topo_forces_pipeline = make_pipeline("topo_forces", &topo_pl, &topo_shader, "main");
    let integrate_pipeline = make_pipeline("integrate", &integrate_pl, &integrate_shader, "main");
    let chebyshev_init_pipeline = make_pipeline("cheb_init", &cheb_init_pl, &cheb_init_shader, "main");
    let chebyshev_step_pipeline = make_pipeline("cheb_step", &cheb_step_pl, &cheb_step_shader, "main");
    let growth_pipeline = make_pipeline("growth", &growth_pl, &growth_shader, "main");
    let bbox_clear_pipeline = make_pipeline("bbox_clear", &bbox_pl, &bbox_shader, "bbox_clear");
    let bbox_reduce_pipeline = make_pipeline("bbox_reduce", &bbox_pl, &bbox_shader, "bbox_reduce");

    // ── Bind groups ─────────────────────────────────────────────────────────

    // fill_bins bind groups
    let fill_bins_bg = [
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&vertex_pos_buf, 0, None)
            .build(device, &fill_bins_pos_layout),
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&sim_params_buf, 0, None)
            .build(device, &params_layout),
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&bin_size_buf, 0, None)
            .build(device, &bins_layout),
    ];

    // prefix sum bind groups (ping-pong)
    let prefix_sum_bgs = [
        // [0] bin_size → bin_offset
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&bin_size_buf, 0, None)
            .buffer_bytes(&bin_offset_buf, 0, None)
            .buffer_bytes(&prefix_sum_step_buf, 0, None)
            .build(device, &prefix_layout),
        // [1] bin_offset → bin_offset_tmp
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&bin_offset_buf, 0, None)
            .buffer_bytes(&bin_offset_tmp_buf, 0, None)
            .buffer_bytes(&prefix_sum_step_buf, 0, None)
            .build(device, &prefix_layout),
        // [2] bin_offset_tmp → bin_offset
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&bin_offset_tmp_buf, 0, None)
            .buffer_bytes(&bin_offset_buf, 0, None)
            .buffer_bytes(&prefix_sum_step_buf, 0, None)
            .build(device, &prefix_layout),
    ];

    // sort bind groups
    let sort_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&sorted_idx_buf, 0, None)
        .buffer_bytes(&bin_offset_buf, 0, None)
        .buffer_bytes(&bin_size_buf, 0, None) // reuse as counter
        .build(device, &sort_data_layout);
    let sort_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // repulsion bind groups
    let repulsion_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&vertex_force_buf, 0, None)
        .buffer_bytes(&sorted_idx_buf, 0, None)
        .buffer_bytes(&bin_offset_buf, 0, None)
        .build(device, &repulsion_data_layout);
    let repulsion_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // topo forces bind groups
    let topo_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&vertex_force_buf, 0, None)
        .buffer_bytes(&he_packed_buf, 0, None)
        .buffer_bytes(&vertex_he_buf, 0, None)
        .build(device, &topo_data_layout);
    let topo_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // integrate bind groups
    let integrate_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&vertex_force_buf, 0, None)
        .build(device, &integrate_data_layout);
    let integrate_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // chebyshev init bind groups
    let cheb_init_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_state_buf, 0, None)
        .buffer_bytes(&cheb_a_buf, 0, None)
        .buffer_bytes(&cheb_b_buf, 0, None)
        .buffer_bytes(&cheb_result_buf, 0, None)
        .buffer_bytes(&he_packed_buf, 0, None)
        .buffer_bytes(&vertex_he_buf, 0, None)
        .build(device, &cheb_init_data_layout);
    let cheb_init_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .buffer_bytes(&cheb_c0_buf, 0, None)
        .buffer_bytes(&cheb_c1_buf, 0, None)
        .build(device, &cheb_init_params_layout);

    // chebyshev step bind groups (3 rotations: curr→prev→next)
    let cheb_step_data_bgs = [
        // [0]: curr=B, prev=A, next=C
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_result_buf, 0, None)
            .buffer_bytes(&he_packed_buf, 0, None)
            .buffer_bytes(&vertex_he_buf, 0, None)
            .build(device, &cheb_step_data_layout),
        // [1]: curr=C, prev=B, next=A
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_result_buf, 0, None)
            .buffer_bytes(&he_packed_buf, 0, None)
            .buffer_bytes(&vertex_he_buf, 0, None)
            .build(device, &cheb_step_data_layout),
        // [2]: curr=A, prev=C, next=B
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_result_buf, 0, None)
            .buffer_bytes(&he_packed_buf, 0, None)
            .buffer_bytes(&vertex_he_buf, 0, None)
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

    // growth bind groups
    let growth_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_state_buf, 0, None)
        .buffer_bytes(&vertex_u_buf, 0, None)
        .buffer_bytes(&cheb_result_buf, 0, None)
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .build(device, &growth_data_layout);
    let growth_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // bbox bind groups
    let bbox_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&vertex_pos_buf, 0, None)
        .buffer_bytes(&bbox_atomic_buf, 0, None)
        .build(device, &bbox_data_layout);
    let bbox_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    GpuCompute {
        vertex_pos_buf, vertex_force_buf,
        bin_size_buf, bin_offset_buf, bin_offset_tmp_buf, sorted_idx_buf, prefix_sum_step_buf,
        he_packed_buf, vertex_he_buf,
        vertex_state_buf, vertex_u_buf,
        cheb_a_buf, cheb_b_buf, cheb_c_buf, cheb_result_buf,
        cheb_coeff_buf, cheb_c0_buf, cheb_c1_buf,
        sim_params_buf,
        pos_readback_buf, state_readback_buf,
        bbox_atomic_buf, bbox_readback_buf,
        bbox_clear_pipeline, bbox_reduce_pipeline,
        bbox_data_bg, bbox_params_bg,
        clear_bins_pipeline, fill_bins_pipeline, prefix_sum_pipeline,
        sort_clear_pipeline, sort_vertices_pipeline,
        repulsion_pipeline, topo_forces_pipeline, integrate_pipeline,
        chebyshev_init_pipeline, chebyshev_step_pipeline, growth_pipeline,
        fill_bins_bg, prefix_sum_bgs,
        sort_data_bg, sort_params_bg,
        repulsion_data_bg, repulsion_params_bg,
        topo_data_bg, topo_params_bg,
        integrate_data_bg, integrate_params_bg,
        cheb_init_data_bg, cheb_init_params_bg,
        cheb_step_data_bgs, cheb_step_params_bgs,
        growth_data_bg, growth_params_bg,
        render_index_buf, render_uniform_buf,
        num_render_tris: 0,
        max_bins,
        topology_dirty: true,
        prefix_sum_step_bufs,
        cheb_coeff_bufs,
        pos_staging: vec![[0.0f32; 4]; MAX_VERTICES],
        he_staging: vec![[0i32; 4]; MAX_HALF_EDGES],
        index_staging: Vec::with_capacity(MAX_FACES * 3),
    }
}

pub(crate) fn upload_mesh_to_gpu(queue: &wgpu::Queue, gpu: &mut GpuCompute, mesh: &HalfEdgeMesh) {
    // Upload vertex positions (vec4: xyz + active flag in w) — reuse staging buffer
    let n = mesh.next_vertex;
    for v in 0..n {
        let active = if mesh.vertex_idx[v] >= 0 { 1.0f32 } else { -1.0 };
        gpu.pos_staging[v] = [mesh.vertex_pos[v].x, mesh.vertex_pos[v].y, mesh.vertex_pos[v].z, active];
    }
    // Zero out remaining entries up to MAX_VERTICES is unnecessary — GPU uses num_vertices param
    queue.write_buffer(&gpu.vertex_pos_buf, 0, bytemuck::cast_slice(&gpu.pos_staging[..n]));

    // Upload vertex states (only active range)
    queue.write_buffer(&gpu.vertex_state_buf, 0, bytemuck::cast_slice(&mesh.vertex_state[..n]));

    // Upload packed half-edge topology (vec4<i32>: dest, twin, next, face) — reuse staging buffer
    let nhe = mesh.next_half_edge;
    for he in 0..nhe {
        gpu.he_staging[he] = [
            mesh.half_edge_dest[he],
            mesh.half_edge_twin[he],
            mesh.half_edge_next[he],
            mesh.half_edge_face[he],
        ];
    }
    queue.write_buffer(&gpu.he_packed_buf, 0, bytemuck::cast_slice(&gpu.he_staging[..nhe]));

    // Upload vertex half-edge indices (only active range)
    queue.write_buffer(&gpu.vertex_he_buf, 0, bytemuck::cast_slice(&mesh.vertex_half_edge[..n]));
}

fn sortable_to_float(s: u32) -> f32 {
    let mask = if (s & 0x80000000) == 0 { 0xFFFFFFFFu32 } else { 0x80000000u32 };
    f32::from_bits(s ^ mask)
}

/// Read back only the bounding box (24 bytes) — cheap enough to run every frame.
pub(crate) fn readback_bbox_only(device: &wgpu::Device, gpu: &GpuCompute) -> [f32; 6] {
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
            sortable_to_float(uints[4]),
            sortable_to_float(uints[5]),
        ]
    };
    gpu.bbox_readback_buf.unmap();
    bbox
}

/// Read back positions, states, and GPU-computed bounding box.
/// Returns [min_x, min_y, max_x, max_y, min_z, max_z] from the GPU bbox reduction.
pub(crate) fn readback_from_gpu(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuCompute, mesh: &mut HalfEdgeMesh) -> [f32; 6] {
    let n = mesh.next_vertex;
    // Only copy the active vertex range instead of the full MAX_VERTICES
    let pos_size = (n * 16) as u64;
    let state_size = (n * 4) as u64;

    // Copy GPU buffers to staging (bbox was already copied in gpu_dispatch_frame)
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(&gpu.vertex_pos_buf, 0, &gpu.pos_readback_buf, 0, pos_size);
    encoder.copy_buffer_to_buffer(&gpu.vertex_state_buf, 0, &gpu.state_readback_buf, 0, state_size);
    queue.submit(Some(encoder.finish()));

    // Map all readback buffers, then poll once
    let pos_slice = gpu.pos_readback_buf.slice(..pos_size);
    pos_slice.map_async(wgpu::MapMode::Read, |_| {});
    let state_slice = gpu.state_readback_buf.slice(..state_size);
    state_slice.map_async(wgpu::MapMode::Read, |_| {});
    let bbox_slice = gpu.bbox_readback_buf.slice(..);
    bbox_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait).unwrap();

    // Read positions
    {
        let data = pos_slice.get_mapped_range();
        let floats: &[[f32; 4]] = bytemuck::cast_slice(&data);
        for v in 0..n {
            mesh.vertex_pos[v] = Vec3::new(floats[v][0], floats[v][1], floats[v][2]);
        }
    }
    gpu.pos_readback_buf.unmap();

    // Read states
    {
        let data = state_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        mesh.vertex_state[..n].copy_from_slice(&floats[..n]);
    }
    gpu.state_readback_buf.unmap();

    // Read bounding box (sortable uint → float conversion) — now 6 values
    let bbox = {
        let data = bbox_slice.get_mapped_range();
        let uints: &[u32] = bytemuck::cast_slice(&data);
        [
            sortable_to_float(uints[0]),
            sortable_to_float(uints[1]),
            sortable_to_float(uints[2]),
            sortable_to_float(uints[3]),
            sortable_to_float(uints[4]),
            sortable_to_float(uints[5]),
        ]
    };
    gpu.bbox_readback_buf.unmap();

    bbox
}

pub(crate) fn rebuild_render_indices(queue: &wgpu::Queue, gpu: &mut GpuCompute, mesh: &HalfEdgeMesh) {
    gpu.index_staging.clear();
    for f in 0..mesh.next_face {
        if mesh.face_idx[f] < 0 { continue; }
        let he0 = mesh.face_half_edge[f];
        if he0 < 0 { continue; }
        let he0 = he0 as usize;
        let he1 = mesh.half_edge_next[he0];
        if he1 < 0 { continue; }
        let he1 = he1 as usize;
        let he2 = mesh.half_edge_next[he1];
        if he2 < 0 { continue; }
        let he2 = he2 as usize;

        let v0 = mesh.half_edge_dest[he0];
        let v1 = mesh.half_edge_dest[he1];
        let v2 = mesh.half_edge_dest[he2];
        if v0 < 0 || v1 < 0 || v2 < 0 { continue; }
        if mesh.vertex_idx[v0 as usize] < 0
            || mesh.vertex_idx[v1 as usize] < 0
            || mesh.vertex_idx[v2 as usize] < 0
        {
            continue;
        }
        gpu.index_staging.push(v0 as u32);
        gpu.index_staging.push(v1 as u32);
        gpu.index_staging.push(v2 as u32);
    }
    gpu.num_render_tris = (gpu.index_staging.len() / 3) as u32;
    queue.write_buffer(&gpu.render_index_buf, 0, bytemuck::cast_slice(&gpu.index_staging));
}

pub(crate) fn update_render_uniforms(
    queue: &wgpu::Queue,
    gpu: &GpuCompute,
    center: Vec3,
    scale: f32,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    render_mode: u32,
    show_wireframe: bool,
    aspect: f32,
) {
    let rot = Mat4::from_rotation_x(pitch) * Mat4::from_rotation_y(yaw);
    let half_h = 10.0 / zoom;
    let half_w = half_h * aspect;
    let proj = Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, -1000.0, 1000.0);
    let view_proj = proj * rot;

    let uniforms = RenderUniforms {
        view_proj: view_proj.to_cols_array_2d(),
        center: [center.x, center.y, center.z, scale],
        light: [0.3, 0.4, 1.0, 0.2],
        render_mode: [render_mode as f32, if show_wireframe { 1.0 } else { 0.0 }, 0.0, 0.0],
    };
    queue.write_buffer(&gpu.render_uniform_buf, 0, bytemuck::bytes_of(&uniforms));
}

pub(crate) fn gpu_dispatch_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &mut GpuCompute,
    mesh: &HalfEdgeMesh,
    params: &GpuSimParams,
    cheb_order: usize,
    cheb_coeffs: &[f32; 20],
) {
    let n = params.num_vertices;
    let num_bins_total = params.num_bins_x * params.num_bins_y * params.num_bins_z + 1;

    // Upload sim params
    queue.write_buffer(&gpu.sim_params_buf, 0, bytemuck::bytes_of(params));

    // Upload positions (positions may have changed from CPU topology ops)
    if gpu.topology_dirty {
        upload_mesh_to_gpu(queue, gpu, mesh);
        gpu.topology_dirty = false;
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute_encoder"),
    });

    // ── Spatial hash (5 passes) ─────────────────────────────────────────

    // 1. Clear bin sizes
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

    // 2. Fill bin sizes
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

    // 3. Prefix sum
    let num_prefix_steps = (num_bins_total as f32).log2().ceil() as u32;
    for i in 0..num_prefix_steps {
        let step_size = 1u32 << i;
        queue.write_buffer(&gpu.prefix_sum_step_buf, 0, bytemuck::bytes_of(&step_size));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum"), timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.prefix_sum_pipeline);
            let bg_idx = if i == 0 { 0 } else if i % 2 == 1 { 1 } else { 2 };
            pass.set_bind_group(0, &gpu.prefix_sum_bgs[bg_idx], &[]);
            pass.dispatch_workgroups(dispatch_count(num_bins_total), 1, 1);
        }
    }
    // If even number of steps > 1, result is in tmp; copy back
    if num_prefix_steps > 1 && num_prefix_steps % 2 == 0 {
        encoder.copy_buffer_to_buffer(
            &gpu.bin_offset_tmp_buf, 0,
            &gpu.bin_offset_buf, 0,
            (num_bins_total * 4) as u64,
        );
    }

    // 4. Clear bin counters for sort
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_clear"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.sort_clear_pipeline);
        pass.set_bind_group(0, &gpu.sort_data_bg, &[]);
        pass.set_bind_group(1, &gpu.sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(num_bins_total), 1, 1);
    }

    // 5. Sort vertices by bin
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_vertices"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.sort_vertices_pipeline);
        pass.set_bind_group(0, &gpu.sort_data_bg, &[]);
        pass.set_bind_group(1, &gpu.sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // ── Forces ──────────────────────────────────────────────────────────

    // Repulsion (initializes vertex_force_buf)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("repulsion"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.repulsion_pipeline);
        pass.set_bind_group(0, &gpu.repulsion_data_bg, &[]);
        pass.set_bind_group(1, &gpu.repulsion_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Topology forces (spring + planar + bulge, adds to vertex_force_buf)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("topo_forces"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.topo_forces_pipeline);
        pass.set_bind_group(0, &gpu.topo_data_bg, &[]);
        pass.set_bind_group(1, &gpu.topo_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Integrate positions
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

    // Init: T_0 = state → cheb_a, T_1 = L(state) → cheb_b, result = c0*T0 + c1*T1
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cheb_init"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.chebyshev_init_pipeline);
        pass.set_bind_group(0, &gpu.cheb_init_data_bg, &[]);
        pass.set_bind_group(1, &gpu.cheb_init_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // Chebyshev steps k=2..cheb_order — upload coefficients to per-step buffers
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

    // Growth function
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("growth"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.growth_pipeline);
        pass.set_bind_group(0, &gpu.growth_data_bg, &[]);
        pass.set_bind_group(1, &gpu.growth_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n), 1, 1);
    }

    // ── Bounding box reduction (for next frame's spatial hash) ───────────

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

    // Copy bbox atomics to readback staging buffer
    encoder.copy_buffer_to_buffer(&gpu.bbox_atomic_buf, 0, &gpu.bbox_readback_buf, 0, 24);

    queue.submit(Some(encoder.finish()));
}
