use bytemuck::{Pod, Zeroable};
use nannou::prelude::*;

// ── Constants ────────────────────────────────────────────────────────────────

pub(crate) const MAX_PARTICLES: usize = 16384;
pub(crate) const MAX_GRA_NODES: usize = 2048;
pub(crate) const MAX_GRA_CONNECTIONS: usize = 6144;
const MAX_GRA_ADJ_ENTRIES: usize = MAX_GRA_CONNECTIONS * 2;
const WORKGROUP_SIZE: u32 = 256;
const GRA_WORKGROUP_SIZE: u32 = 64;

pub(crate) const SAMPLE_RATE: u32 = 44100;
pub(crate) const CHUNK_SIZE: u32 = 2048;
pub(crate) const NUM_CHANNELS: u32 = 2;
pub(crate) const NUM_AUDIO_STAGING_BUFS: usize = 4;
pub(crate) const CHUNK_FLOATS: u32 = CHUNK_SIZE * NUM_CHANNELS;
pub(crate) const NUM_MODAL_BANDS: usize = 8;
pub(crate) const MODAL_CHEB_ORDER: usize = 12;
const MAX_CHEB_ORDER: usize = 20;

// ── Shader sources ───────────────────────────────────────────────────────────

const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");

macro_rules! shader_with_common {
    ($file:expr) => {
        concat!(include_str!("shaders/common.wgsl"), include_str!($file))
    };
}

const PARTICLE_WGSL: &str = shader_with_common!("shaders/particle.wgsl");
const GRA_SPRING_WGSL: &str = shader_with_common!("shaders/gra_spring.wgsl");
const GRA_INTEGRATE_WGSL: &str = shader_with_common!("shaders/gra_integrate.wgsl");
const PARTICLE_BIN_WGSL: &str = shader_with_common!("shaders/particle_bin.wgsl");
const GRA_BIN_WGSL: &str = shader_with_common!("shaders/gra_bin.wgsl");
const BIN_PREFIX_SUM_WGSL: &str = include_str!("shaders/bin_prefix_sum.wgsl");
const PARTICLE_SORT_WGSL: &str = shader_with_common!("shaders/particle_sort.wgsl");
const GRA_SORT_WGSL: &str = shader_with_common!("shaders/gra_sort.wgsl");
const AUDIO_WGSL: &str = shader_with_common!("shaders/audio.wgsl");
const PHASE_UPDATE_WGSL: &str = shader_with_common!("shaders/phase_update.wgsl");
const BBOX_WGSL: &str = shader_with_common!("shaders/bbox.wgsl");
const MODAL_CHEB_INIT_WGSL: &str = shader_with_common!("shaders/modal_cheb_init.wgsl");
const MODAL_CHEB_STEP_WGSL: &str = shader_with_common!("shaders/modal_cheb_step.wgsl");
const MODAL_PHASE_UPDATE_WGSL: &str = shader_with_common!("shaders/modal_phase_update.wgsl");

// ── GPU data types ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct Particle {
    pub(crate) pos: [f32; 2],
    pub(crate) vel: [f32; 2],
    pub(crate) phase: f32,
    pub(crate) energy: f32,
    pub(crate) species: [f32; 2],
    pub(crate) alpha: [f32; 2],
    pub(crate) interaction: [f32; 2],
    pub(crate) amp_phase: f32,
    pub(crate) _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SimParams {
    pub(crate) world_half: f32,
    pub(crate) dt: f32,
    pub(crate) time: f32,
    pub(crate) num_particles: u32,
    pub(crate) particle_friction: f32,
    pub(crate) particle_mass: f32,
    pub(crate) particle_radius: f32,
    pub(crate) particle_collision_radius: f32,
    pub(crate) particle_collision_strength: f32,
    pub(crate) particle_max_force: f32,
    pub(crate) particle_copy_radius: f32,
    pub(crate) particle_copy_cos_sim: f32,
    pub(crate) particle_copy_prob: f32,
    pub(crate) p_bin_size: f32,
    pub(crate) p_num_bins_x: u32,
    pub(crate) p_num_bins_y: u32,
    pub(crate) num_gra_nodes: u32,
    pub(crate) num_gra_connections: u32,
    pub(crate) gra_spring_length: f32,
    pub(crate) gra_spring_stiffness: f32,
    pub(crate) gra_damping: f32,
    pub(crate) gra_max_velocity: f32,
    pub(crate) g_bin_size: f32,
    pub(crate) g_num_bins_x: u32,
    pub(crate) g_num_bins_y: u32,
    pub(crate) gra_repulsion_radius: f32,
    pub(crate) gra_repulsion_strength: f32,
    pub(crate) particle_friction_mu: f32,
    pub(crate) current_strength: f32,
    pub(crate) _pad0: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct AudioParams {
    pub(crate) sample_rate: f32,
    pub(crate) num_particles: u32,
    pub(crate) num_gra_nodes: u32,
    pub(crate) chunk_size: u32,
    pub(crate) volume: f32,
    pub(crate) _align_pad: f32, // padding to match WGSL vec2 alignment
    pub(crate) current_x: [f32; 2],
    pub(crate) current_y: [f32; 2],
    pub(crate) max_speed: f32,
    pub(crate) energy_scale: f32,
    pub(crate) gra_max_speed: f32,
    pub(crate) p_map_x0: f32,
    pub(crate) p_map_y0: f32,
    pub(crate) p_bin_size: f32,
    pub(crate) p_num_bins_x: u32,
    pub(crate) p_num_bins_y: u32,
    pub(crate) g_bin_size: f32,
    pub(crate) g_num_bins_x: u32,
    pub(crate) g_num_bins_y: u32,
    pub(crate) world_half: f32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
    pub(crate) _pad3: u32, // struct alignment padding (vec2 → 8-byte aligned, total must be multiple of 8)
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct ModalFreqs {
    pub(crate) lo: [f32; 4],
    pub(crate) hi: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct RenderUniforms {
    pub(crate) min_x: f32,
    pub(crate) min_y: f32,
    pub(crate) max_x: f32,
    pub(crate) max_y: f32,
    pub(crate) particle_size: f32,
    pub(crate) gra_node_radius: f32,
    pub(crate) num_particles: u32,
    pub(crate) num_gra_nodes: u32,
    pub(crate) num_gra_connections: u32,
    pub(crate) window_aspect: f32,
    pub(crate) world_half: f32,
    pub(crate) max_speed: f32,
    pub(crate) energy_scale: f32,
    pub(crate) current_strength: f32,
    pub(crate) time: f32,
    pub(crate) _pad0: u32,
}

fn dispatch_count(n: u32, workgroup: u32) -> u32 {
    (n + workgroup - 1) / workgroup
}

// ── Compute bin counts ───────────────────────────────────────────────────────

pub(crate) fn compute_bin_counts(world_half: f32, bin_size: f32) -> (u32, u32) {
    let world_size = world_half * 2.0;
    let n = (world_size / bin_size).ceil() as u32;
    (n, n)
}

// ── GPU state ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct GpuCompute {
    // ── Particle buffers (ping-pong) ──
    pub(crate) particle_bufs: [wgpu::Buffer; 2],
    pub(crate) particle_frame: usize, // 0 or 1, alternates each frame

    // ── Particle spatial hash ──
    pub(crate) p_bin_size_buf: wgpu::Buffer,
    pub(crate) p_bin_offset_buf: wgpu::Buffer, // prefix sum result
    pub(crate) p_num_bins: u32,                 // total bins (includes +1 padding)

    // ── GRA node buffers ──
    pub(crate) gra_pos_buf: wgpu::Buffer,
    pub(crate) gra_vel_buf: wgpu::Buffer,
    pub(crate) gra_force_buf: wgpu::Buffer,
    pub(crate) gra_state_buf: wgpu::Buffer,
    pub(crate) adj_offset_buf: wgpu::Buffer,
    pub(crate) adj_list_buf: wgpu::Buffer,
    pub(crate) connection_buf: wgpu::Buffer,

    // ── GRA spatial hash (for particle→GRA lookups) ──
    pub(crate) gra_sorted_pos_buf: wgpu::Buffer,
    pub(crate) g_bin_size_buf: wgpu::Buffer,
    pub(crate) g_bin_offset_buf: wgpu::Buffer,
    pub(crate) g_num_bins: u32,

    // ── Params ──
    pub(crate) sim_params_buf: wgpu::Buffer,
    pub(crate) audio_params_buf: wgpu::Buffer,
    pub(crate) render_uniform_buf: wgpu::Buffer,

    // ── GRA readback (double-buffered) ──
    pub(crate) gra_pos_readback_bufs: [wgpu::Buffer; 2],
    pub(crate) gra_readback_frame: usize,
    pub(crate) bbox_atomic_buf: wgpu::Buffer,
    pub(crate) bbox_readback_buf: wgpu::Buffer,

    // ── Audio ──
    pub(crate) audio_out_buf: wgpu::Buffer,
    pub(crate) audio_staging_bufs: Vec<wgpu::Buffer>,

    // ── Pipelines ──
    // Particle
    particle_sim_pipeline: wgpu::ComputePipeline,
    p_clear_bins_pipeline: wgpu::ComputePipeline,
    p_fill_bins_pipeline: wgpu::ComputePipeline,
    p_prefix_sum_pipeline: wgpu::ComputePipeline,
    p_sort_clear_pipeline: wgpu::ComputePipeline,
    p_sort_pipeline: wgpu::ComputePipeline,

    // GRA
    gra_spring_pipeline: wgpu::ComputePipeline,
    gra_integrate_pipeline: wgpu::ComputePipeline,
    g_clear_bins_pipeline: wgpu::ComputePipeline,
    g_fill_bins_pipeline: wgpu::ComputePipeline,
    g_prefix_sum_pipeline: wgpu::ComputePipeline,
    g_sort_clear_pipeline: wgpu::ComputePipeline,
    g_sort_pipeline: wgpu::ComputePipeline,
    bbox_clear_pipeline: wgpu::ComputePipeline,
    bbox_reduce_pipeline: wgpu::ComputePipeline,

    // Audio (combined particle + GRA modal)
    audio_pipeline: wgpu::ComputePipeline,
    particle_phase_pipeline: wgpu::ComputePipeline,
    modal_phase_update_pipeline: wgpu::ComputePipeline,

    // Modal synthesis buffers (Chebyshev decomposition)
    pub(crate) modal_amp_buf: wgpu::Buffer,
    pub(crate) modal_phase_buf: wgpu::Buffer,
    pub(crate) modal_freq_buf: wgpu::Buffer,
    modal_coeff_lo_bufs: Vec<wgpu::Buffer>,
    modal_coeff_hi_bufs: Vec<wgpu::Buffer>,
    cheb_a_buf: wgpu::Buffer,
    cheb_b_buf: wgpu::Buffer,
    cheb_c_buf: wgpu::Buffer,
    modal_cheb_init_pipeline: wgpu::ComputePipeline,
    modal_cheb_step_pipeline: wgpu::ComputePipeline,

    // ── Bind groups ──
    // Particle sim (2 variants for ping-pong)
    particle_sim_bgs: [wgpu::BindGroup; 2],     // group 0: particles[dst] + params + p_bin_offset
    particle_sim_gra_bg: wgpu::BindGroup,        // group 1: gra_sorted_pos + g_bin_offset

    // Particle binning
    p_bin_fill_bgs: [wgpu::BindGroup; 2],        // group 0: particles[src]
    p_bin_params_bg: wgpu::BindGroup,             // group 1: params
    p_bin_size_bg: wgpu::BindGroup,               // group 2: bin_size
    p_prefix_bg: wgpu::BindGroup,                 // group 0: source=bin_size, dest=bin_offset
    p_sort_data_bgs: [wgpu::BindGroup; 2],        // group 0: src=particles[X], dst=particles[Y], offset, size
    p_sort_params_bg: wgpu::BindGroup,             // group 1: params

    // GRA physics
    gra_spring_data_bg: wgpu::BindGroup,
    gra_spring_params_bg: wgpu::BindGroup,
    gra_integrate_data_bg: wgpu::BindGroup,
    gra_integrate_params_bg: wgpu::BindGroup,

    // GRA binning
    gra_bin_fill_data_bg: wgpu::BindGroup,
    gra_bin_params_bg: wgpu::BindGroup,
    gra_bin_size_bg: wgpu::BindGroup,
    g_prefix_bg: wgpu::BindGroup,
    gra_sort_data_bg: wgpu::BindGroup,
    gra_sort_params_bg: wgpu::BindGroup,

    // Bbox
    bbox_data_bg: wgpu::BindGroup,
    bbox_params_bg: wgpu::BindGroup,

    // Audio (2 variants for ping-pong particle buffer)
    audio_bgs: [wgpu::BindGroup; 2],
    particle_phase_bgs: [wgpu::BindGroup; 2],
    modal_phase_update_bg: wgpu::BindGroup,

    // Chebyshev decomposition bind groups
    modal_cheb_init_data_bg: wgpu::BindGroup,
    modal_cheb_init_params_bg: wgpu::BindGroup,
    modal_cheb_step_data_bgs: [wgpu::BindGroup; 3],
    modal_cheb_step_params_bgs: Vec<wgpu::BindGroup>,

    // Config
    pub(crate) topology_dirty: bool,
    pub(crate) num_gra_nodes: u32,
    pub(crate) num_gra_connections: u32,
    pub(crate) num_particles: u32,
}

// ── Create GPU compute state ─────────────────────────────────────────────────

pub(crate) fn create_gpu_compute(
    device: &wgpu::Device,
    world_half: f32,
    p_bin_size: f32,
    g_bin_size: f32,
) -> GpuCompute {
    let particle_buf_size = (MAX_PARTICLES * std::mem::size_of::<Particle>()) as u64;
    let gra_vec4_size = (MAX_GRA_NODES * 16) as u64;
    let adj_offset_size = ((MAX_GRA_NODES + 1) * 4) as u64;
    let adj_list_size = (MAX_GRA_ADJ_ENTRIES * 4) as u64;
    let conn_buf_size = (MAX_GRA_CONNECTIONS * 2 * 4) as u64;

    let storage_rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let storage_r = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;

    // Particle buffers (ping-pong)
    let particle_bufs = [
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_0"), size: particle_buf_size, usage: storage_rw, mapped_at_creation: false,
        }),
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_1"), size: particle_buf_size, usage: storage_rw, mapped_at_creation: false,
        }),
    ];

    // Particle spatial hash
    let (p_bins_x, p_bins_y) = compute_bin_counts(world_half, p_bin_size);
    let p_num_bins = p_bins_x * p_bins_y;
    let p_bin_buf_size = ((p_num_bins + 1) * 4) as u64;
    let p_bin_size_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p_bin_size"), size: p_bin_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let p_bin_offset_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p_bin_offset"), size: p_bin_buf_size, usage: storage_rw, mapped_at_creation: false,
    });

    // GRA node buffers
    let gra_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gra_pos"), size: gra_vec4_size, usage: storage_rw, mapped_at_creation: false,
    });
    let gra_vel_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gra_vel"), size: gra_vec4_size, usage: storage_rw, mapped_at_creation: false,
    });
    let gra_force_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gra_force"), size: gra_vec4_size, usage: storage_rw, mapped_at_creation: false,
    });
    let gra_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gra_state"), size: gra_vec4_size, usage: storage_rw, mapped_at_creation: false,
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

    // GRA spatial hash
    let (g_bins_x, g_bins_y) = compute_bin_counts(world_half, g_bin_size);
    let g_num_bins = g_bins_x * g_bins_y;
    let g_bin_buf_size = ((g_num_bins + 1) * 4) as u64;
    let gra_sorted_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gra_sorted_pos"), size: gra_vec4_size, usage: storage_rw, mapped_at_creation: false,
    });
    let g_bin_size_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g_bin_size"), size: g_bin_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let g_bin_offset_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g_bin_offset"), size: g_bin_buf_size, usage: storage_rw, mapped_at_creation: false,
    });

    // Params
    let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim_params"), size: std::mem::size_of::<SimParams>() as u64, usage: uniform, mapped_at_creation: false,
    });
    let audio_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_params"), size: std::mem::size_of::<AudioParams>() as u64, usage: uniform, mapped_at_creation: false,
    });
    let render_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_uniforms"), size: std::mem::size_of::<RenderUniforms>() as u64, usage: uniform, mapped_at_creation: false,
    });

    // GRA readback (double-buffered)
    let gra_pos_readback_bufs = [
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gra_pos_readback_0"), size: gra_vec4_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        }),
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gra_pos_readback_1"), size: gra_vec4_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        }),
    ];
    let bbox_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bbox_atomic"), size: 16, usage: storage_rw, mapped_at_creation: false,
    });
    let bbox_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bbox_readback"), size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });

    // Audio
    let audio_buf_size = (CHUNK_SIZE * NUM_CHANNELS * 4) as u64;
    let audio_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_out"), size: audio_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let audio_staging_bufs: Vec<wgpu::Buffer> = (0..NUM_AUDIO_STAGING_BUFS)
        .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("audio_staging_{}", i)),
            size: audio_buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
        .collect();

    // ── Modal synthesis buffers ─────────────────────────────────────────────
    let modal_buf_size = (MAX_GRA_NODES * 2 * 16) as u64; // 2 vec4 per node
    let modal_amp_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("modal_amp"), size: modal_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let modal_phase_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("modal_phase"), size: modal_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let modal_freq_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("modal_freq"),
        size: std::mem::size_of::<ModalFreqs>() as u64,
        usage: uniform,
        mapped_at_creation: false,
    });
    let modal_coeff_lo_bufs: Vec<wgpu::Buffer> = (0..MAX_CHEB_ORDER)
        .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("modal_coeff_lo_{}", i)),
            size: 16, usage: uniform, mapped_at_creation: false,
        }))
        .collect();
    let modal_coeff_hi_bufs: Vec<wgpu::Buffer> = (0..MAX_CHEB_ORDER)
        .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("modal_coeff_hi_{}", i)),
            size: 16, usage: uniform, mapped_at_creation: false,
        }))
        .collect();
    // Chebyshev temp buffers (vec4 per node, 3 for rotation)
    let cheb_node_buf_size = (MAX_GRA_NODES * 16) as u64;
    let cheb_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_a"), size: cheb_node_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let cheb_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_b"), size: cheb_node_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    let cheb_c_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheb_c"), size: cheb_node_buf_size, usage: storage_rw, mapped_at_creation: false,
    });
    // ── Shader modules ───────────────────────────────────────────────────────
    let cs = wgpu::ShaderStages::COMPUTE;

    let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle"), source: wgpu::ShaderSource::Wgsl(PARTICLE_WGSL.into()),
    });
    let gra_spring_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gra_spring"), source: wgpu::ShaderSource::Wgsl(GRA_SPRING_WGSL.into()),
    });
    let gra_integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gra_integrate"), source: wgpu::ShaderSource::Wgsl(GRA_INTEGRATE_WGSL.into()),
    });
    let p_bin_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle_bin"), source: wgpu::ShaderSource::Wgsl(PARTICLE_BIN_WGSL.into()),
    });
    let g_bin_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gra_bin"), source: wgpu::ShaderSource::Wgsl(GRA_BIN_WGSL.into()),
    });
    let prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("prefix_sum"), source: wgpu::ShaderSource::Wgsl(BIN_PREFIX_SUM_WGSL.into()),
    });
    let p_sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle_sort"), source: wgpu::ShaderSource::Wgsl(PARTICLE_SORT_WGSL.into()),
    });
    let g_sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gra_sort"), source: wgpu::ShaderSource::Wgsl(GRA_SORT_WGSL.into()),
    });
    let audio_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("audio"), source: wgpu::ShaderSource::Wgsl(AUDIO_WGSL.into()),
    });
    let phase_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("phase_update"), source: wgpu::ShaderSource::Wgsl(PHASE_UPDATE_WGSL.into()),
    });
    let bbox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bbox"), source: wgpu::ShaderSource::Wgsl(BBOX_WGSL.into()),
    });
    let modal_cheb_init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("modal_cheb_init"), source: wgpu::ShaderSource::Wgsl(MODAL_CHEB_INIT_WGSL.into()),
    });
    let modal_cheb_step_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("modal_cheb_step"), source: wgpu::ShaderSource::Wgsl(MODAL_CHEB_STEP_WGSL.into()),
    });
    let modal_phase_update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("modal_phase_update"), source: wgpu::ShaderSource::Wgsl(MODAL_PHASE_UPDATE_WGSL.into()),
    });

    // ── Bind group layouts ───────────────────────────────────────────────────

    let params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)
        .build(device);

    // Particle sim group 0: particles(RW) + params(U) + p_bin_offset(R)
    let particle_sim_layout_0 = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // particles
        .uniform_buffer(cs, false)         // params
        .storage_buffer(cs, false, true)   // p_bin_offset
        .build(device);

    // Particle sim group 1: gra_sorted_pos(R) + g_bin_offset(R)
    let particle_sim_layout_1 = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // gra_sorted_pos
        .storage_buffer(cs, false, true)   // g_bin_offset
        .build(device);

    // Particle bin fill group 0: particles(R)
    let p_bin_fill_layout_0 = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // particles
        .build(device);

    // Bin size group: bin_size(RW)
    let bin_size_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // bin_size (atomic)
        .build(device);

    // Prefix sum: source(R) + destination(RW)
    let prefix_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // source
        .storage_buffer(cs, false, false)  // destination
        .build(device);

    // Particle sort group 0: src(R) + dst(RW) + offset(R) + size(RW)
    let p_sort_layout_0 = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // source particles
        .storage_buffer(cs, false, false)  // dest particles
        .storage_buffer(cs, false, true)   // bin_offset
        .storage_buffer(cs, false, false)  // bin_size (atomic)
        .build(device);

    // GRA spring group 0: pos(R) + force(RW) + adj_offset(R) + adj_list(R)
    let gra_spring_layout_0 = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // gra_pos
        .storage_buffer(cs, false, false)  // gra_force
        .storage_buffer(cs, false, true)   // adj_offset
        .storage_buffer(cs, false, true)   // adj_list
        .build(device);

    // GRA integrate group 0: pos(RW) + vel(RW) + force(R)
    let gra_integrate_layout_0 = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // gra_pos
        .storage_buffer(cs, false, false)  // gra_vel
        .storage_buffer(cs, false, true)   // gra_force
        .build(device);

    // GRA bin fill group 0: gra_pos(R)
    let g_bin_fill_layout_0 = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // gra_pos
        .build(device);

    // GRA sort group 0: gra_pos(R) + sorted_pos(RW) + offset(R) + size(RW)
    let g_sort_layout_0 = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // gra_pos
        .storage_buffer(cs, false, false)  // gra_sorted_pos
        .storage_buffer(cs, false, true)   // bin_offset
        .storage_buffer(cs, false, false)  // bin_size (atomic)
        .build(device);

    // Bbox group 0: pos(R) + atomic(RW)
    let bbox_layout_0 = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // gra_pos
        .storage_buffer(cs, false, false)  // bbox_atomic
        .build(device);

    // Audio: particles(R) + p_bin_offset(R) + gra_sorted_pos(R) + g_bin_offset(R) + gra_vel(R) + modal_amp(R) + modal_phase(R) + audio_out(RW) + params(U) + freqs(U)
    let audio_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // particles
        .storage_buffer(cs, false, true)   // p_bin_offset
        .storage_buffer(cs, false, true)   // gra_sorted_pos
        .storage_buffer(cs, false, true)   // g_bin_offset
        .storage_buffer(cs, false, true)   // gra_vel
        .storage_buffer(cs, false, true)   // modal_amp
        .storage_buffer(cs, false, true)   // modal_phase
        .storage_buffer(cs, false, false)  // audio_out
        .uniform_buffer(cs, false)         // audio_params
        .uniform_buffer(cs, false)         // modal_freqs
        .storage_buffer(cs, false, true)   // gra_state (for freq_scale in .w)
        .build(device);

    // Particle phase: particles(RW) + audio_params(U)
    let particle_phase_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // particles
        .uniform_buffer(cs, false)         // audio_params
        .build(device);

    // Modal Chebyshev init data: gra_state(R) + t_a(RW) + t_b(RW) + modal_amp(RW) + adj_offset(R) + adj_list(R)
    let modal_cheb_init_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // gra_state
        .storage_buffer(cs, false, false)  // t_a
        .storage_buffer(cs, false, false)  // t_b
        .storage_buffer(cs, false, false)  // modal_amp
        .storage_buffer(cs, false, true)   // adj_offset
        .storage_buffer(cs, false, true)   // adj_list
        .build(device);

    // Modal Chebyshev init params: sim_params(U) + c0_lo(U) + c0_hi(U) + c1_lo(U) + c1_hi(U)
    let modal_init_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)  // sim_params
        .uniform_buffer(cs, false)  // c0_lo
        .uniform_buffer(cs, false)  // c0_hi
        .uniform_buffer(cs, false)  // c1_lo
        .uniform_buffer(cs, false)  // c1_hi
        .build(device);

    // Modal Chebyshev step data: t_curr(R) + t_prev(R) + t_next(RW) + modal_amp(RW) + adj_offset(R) + adj_list(R)
    let modal_cheb_step_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, true)   // t_curr
        .storage_buffer(cs, false, true)   // t_prev
        .storage_buffer(cs, false, false)  // t_next
        .storage_buffer(cs, false, false)  // modal_amp
        .storage_buffer(cs, false, true)   // adj_offset
        .storage_buffer(cs, false, true)   // adj_list
        .build(device);

    // Modal Chebyshev step params: sim_params(U) + ck_lo(U) + ck_hi(U)
    let modal_step_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(cs, false)  // sim_params
        .uniform_buffer(cs, false)  // ck_lo
        .uniform_buffer(cs, false)  // ck_hi
        .build(device);

    // Modal phase update: modal_phase(RW) + audio_params(U) + freqs(U)
    let modal_phase_update_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(cs, false, false)  // modal_phase
        .uniform_buffer(cs, false)         // audio_params
        .uniform_buffer(cs, false)         // modal_freqs
        .storage_buffer(cs, false, true)   // gra_state (for freq_scale in .w)
        .build(device);

    // ── Pipeline layouts ─────────────────────────────────────────────────────

    let particle_sim_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle_sim_pl"),
        bind_group_layouts: &[&particle_sim_layout_0, &particle_sim_layout_1],
        push_constant_ranges: &[],
    });

    let p_bin_fill_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("p_bin_fill_pl"),
        bind_group_layouts: &[&p_bin_fill_layout_0, &params_layout, &bin_size_layout],
        push_constant_ranges: &[],
    });

    let prefix_sum_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("prefix_sum_pl"),
        bind_group_layouts: &[&prefix_layout],
        push_constant_ranges: &[],
    });

    let p_sort_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("p_sort_pl"),
        bind_group_layouts: &[&p_sort_layout_0, &params_layout],
        push_constant_ranges: &[],
    });

    let gra_spring_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gra_spring_pl"),
        bind_group_layouts: &[&gra_spring_layout_0, &params_layout],
        push_constant_ranges: &[],
    });

    let gra_integrate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gra_integrate_pl"),
        bind_group_layouts: &[&gra_integrate_layout_0, &params_layout],
        push_constant_ranges: &[],
    });

    let g_bin_fill_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("g_bin_fill_pl"),
        bind_group_layouts: &[&g_bin_fill_layout_0, &params_layout, &bin_size_layout],
        push_constant_ranges: &[],
    });

    let g_sort_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("g_sort_pl"),
        bind_group_layouts: &[&g_sort_layout_0, &params_layout],
        push_constant_ranges: &[],
    });

    let bbox_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bbox_pl"),
        bind_group_layouts: &[&bbox_layout_0, &params_layout],
        push_constant_ranges: &[],
    });

    let audio_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("audio_pl"),
        bind_group_layouts: &[&audio_layout],
        push_constant_ranges: &[],
    });

    let particle_phase_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle_phase_pl"),
        bind_group_layouts: &[&particle_phase_layout],
        push_constant_ranges: &[],
    });

    let modal_cheb_init_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("modal_cheb_init_pl"),
        bind_group_layouts: &[&modal_cheb_init_data_layout, &modal_init_params_layout],
        push_constant_ranges: &[],
    });
    let modal_cheb_step_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("modal_cheb_step_pl"),
        bind_group_layouts: &[&modal_cheb_step_data_layout, &modal_step_params_layout],
        push_constant_ranges: &[],
    });
    let modal_phase_update_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("modal_phase_update_pl"),
        bind_group_layouts: &[&modal_phase_update_layout],
        push_constant_ranges: &[],
    });

    // ── Compute pipelines ────────────────────────────────────────────────────

    let make = |label, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    let particle_sim_pipeline = make("particle_sim", &particle_sim_pl, &particle_shader, "main");
    let p_clear_bins_pipeline = make("p_clear_bins", &p_bin_fill_pl, &p_bin_shader, "clear_bins");
    let p_fill_bins_pipeline = make("p_fill_bins", &p_bin_fill_pl, &p_bin_shader, "fill_bins");
    let p_prefix_sum_pipeline = make("p_prefix_sum", &prefix_sum_pl, &prefix_sum_shader, "prefix_sum");
    let p_sort_clear_pipeline = make("p_sort_clear", &p_sort_pl, &p_sort_shader, "clear_bins");
    let p_sort_pipeline = make("p_sort", &p_sort_pl, &p_sort_shader, "sort_particles");

    let gra_spring_pipeline = make("gra_spring", &gra_spring_pl, &gra_spring_shader, "main");
    let gra_integrate_pipeline = make("gra_integrate", &gra_integrate_pl, &gra_integrate_shader, "main");
    let g_clear_bins_pipeline = make("g_clear_bins", &g_bin_fill_pl, &g_bin_shader, "clear_bins");
    let g_fill_bins_pipeline = make("g_fill_bins", &g_bin_fill_pl, &g_bin_shader, "fill_bins");
    let g_prefix_sum_pipeline = make("g_prefix_sum", &prefix_sum_pl, &prefix_sum_shader, "prefix_sum");
    let g_sort_clear_pipeline = make("g_sort_clear", &g_sort_pl, &g_sort_shader, "clear_bins");
    let g_sort_pipeline = make("g_sort", &g_sort_pl, &g_sort_shader, "sort_nodes");
    let bbox_clear_pipeline = make("bbox_clear", &bbox_pl, &bbox_shader, "bbox_clear");
    let bbox_reduce_pipeline = make("bbox_reduce", &bbox_pl, &bbox_shader, "bbox_reduce");

    let audio_pipeline = make("audio", &audio_pl, &audio_shader, "main");
    let particle_phase_pipeline = make("particle_phase", &particle_phase_pl, &phase_shader, "update_particle_phase");

    let modal_cheb_init_pipeline = make("modal_cheb_init", &modal_cheb_init_pl, &modal_cheb_init_shader, "main");
    let modal_cheb_step_pipeline = make("modal_cheb_step", &modal_cheb_step_pl, &modal_cheb_step_shader, "main");
    let modal_phase_update_pipeline = make("modal_phase_update", &modal_phase_update_pl_layout, &modal_phase_update_shader, "main");

    // ── Bind groups ──────────────────────────────────────────────────────────

    // Particle sim (ping-pong)
    let particle_sim_bgs = [0, 1].map(|i| {
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&particle_bufs[i], 0, None)
            .buffer_bytes(&sim_params_buf, 0, None)
            .buffer_bytes(&p_bin_offset_buf, 0, None)
            .build(device, &particle_sim_layout_0)
    });

    let particle_sim_gra_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&gra_sorted_pos_buf, 0, None)
        .buffer_bytes(&g_bin_offset_buf, 0, None)
        .build(device, &particle_sim_layout_1);

    // Particle bin fill (read from either buffer)
    let p_bin_fill_bgs = [0, 1].map(|i| {
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&particle_bufs[i], 0, None)
            .build(device, &p_bin_fill_layout_0)
    });
    let p_bin_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);
    let p_bin_size_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&p_bin_size_buf, 0, None)
        .build(device, &bin_size_layout);
    let p_prefix_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&p_bin_size_buf, 0, None)
        .buffer_bytes(&p_bin_offset_buf, 0, None)
        .build(device, &prefix_layout);

    // Particle sort (src→dst, with ping-pong)
    let p_sort_data_bgs = [
        // [0]: sort from buf[0] to buf[1]
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&particle_bufs[0], 0, None)
            .buffer_bytes(&particle_bufs[1], 0, None)
            .buffer_bytes(&p_bin_offset_buf, 0, None)
            .buffer_bytes(&p_bin_size_buf, 0, None)
            .build(device, &p_sort_layout_0),
        // [1]: sort from buf[1] to buf[0]
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&particle_bufs[1], 0, None)
            .buffer_bytes(&particle_bufs[0], 0, None)
            .buffer_bytes(&p_bin_offset_buf, 0, None)
            .buffer_bytes(&p_bin_size_buf, 0, None)
            .build(device, &p_sort_layout_0),
    ];
    let p_sort_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // GRA physics
    let gra_spring_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&gra_pos_buf, 0, None)
        .buffer_bytes(&gra_force_buf, 0, None)
        .buffer_bytes(&adj_offset_buf, 0, None)
        .buffer_bytes(&adj_list_buf, 0, None)
        .build(device, &gra_spring_layout_0);
    let gra_spring_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);
    let gra_integrate_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&gra_pos_buf, 0, None)
        .buffer_bytes(&gra_vel_buf, 0, None)
        .buffer_bytes(&gra_force_buf, 0, None)
        .build(device, &gra_integrate_layout_0);
    let gra_integrate_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // GRA binning
    let gra_bin_fill_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&gra_pos_buf, 0, None)
        .build(device, &g_bin_fill_layout_0);
    let gra_bin_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);
    let gra_bin_size_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&g_bin_size_buf, 0, None)
        .build(device, &bin_size_layout);
    let g_prefix_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&g_bin_size_buf, 0, None)
        .buffer_bytes(&g_bin_offset_buf, 0, None)
        .build(device, &prefix_layout);
    let gra_sort_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&gra_pos_buf, 0, None)
        .buffer_bytes(&gra_sorted_pos_buf, 0, None)
        .buffer_bytes(&g_bin_offset_buf, 0, None)
        .buffer_bytes(&g_bin_size_buf, 0, None)
        .build(device, &g_sort_layout_0);
    let gra_sort_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // Bbox
    let bbox_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&gra_pos_buf, 0, None)
        .buffer_bytes(&bbox_atomic_buf, 0, None)
        .build(device, &bbox_layout_0);
    let bbox_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .build(device, &params_layout);

    // Audio (ping-pong for particle buffer, combined particle + GRA modal)
    let audio_bgs = [0, 1].map(|i| {
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&particle_bufs[i], 0, None)
            .buffer_bytes(&p_bin_offset_buf, 0, None)
            .buffer_bytes(&gra_sorted_pos_buf, 0, None)
            .buffer_bytes(&g_bin_offset_buf, 0, None)
            .buffer_bytes(&gra_vel_buf, 0, None)
            .buffer_bytes(&modal_amp_buf, 0, None)
            .buffer_bytes(&modal_phase_buf, 0, None)
            .buffer_bytes(&audio_out_buf, 0, None)
            .buffer_bytes(&audio_params_buf, 0, None)
            .buffer_bytes(&modal_freq_buf, 0, None)
            .buffer_bytes(&gra_state_buf, 0, None)
            .build(device, &audio_layout)
    });

    let particle_phase_bgs = [0, 1].map(|i| {
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&particle_bufs[i], 0, None)
            .buffer_bytes(&audio_params_buf, 0, None)
            .build(device, &particle_phase_layout)
    });

    // Modal Chebyshev init data bind group
    let modal_cheb_init_data_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&gra_state_buf, 0, None)
        .buffer_bytes(&cheb_a_buf, 0, None)
        .buffer_bytes(&cheb_b_buf, 0, None)
        .buffer_bytes(&modal_amp_buf, 0, None)
        .buffer_bytes(&adj_offset_buf, 0, None)
        .buffer_bytes(&adj_list_buf, 0, None)
        .build(device, &modal_cheb_init_data_layout);

    // Modal Chebyshev init params bind group
    let modal_cheb_init_params_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&sim_params_buf, 0, None)
        .buffer_bytes(&modal_coeff_lo_bufs[0], 0, None)
        .buffer_bytes(&modal_coeff_hi_bufs[0], 0, None)
        .buffer_bytes(&modal_coeff_lo_bufs[1], 0, None)
        .buffer_bytes(&modal_coeff_hi_bufs[1], 0, None)
        .build(device, &modal_init_params_layout);

    // Modal Chebyshev step data bind groups (3 rotations)
    let modal_cheb_step_data_bgs = [
        // [0]: curr=B, prev=A, next=C
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&modal_amp_buf, 0, None)
            .buffer_bytes(&adj_offset_buf, 0, None)
            .buffer_bytes(&adj_list_buf, 0, None)
            .build(device, &modal_cheb_step_data_layout),
        // [1]: curr=C, prev=B, next=A
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&modal_amp_buf, 0, None)
            .buffer_bytes(&adj_offset_buf, 0, None)
            .buffer_bytes(&adj_list_buf, 0, None)
            .build(device, &modal_cheb_step_data_layout),
        // [2]: curr=A, prev=C, next=B
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(&cheb_a_buf, 0, None)
            .buffer_bytes(&cheb_c_buf, 0, None)
            .buffer_bytes(&cheb_b_buf, 0, None)
            .buffer_bytes(&modal_amp_buf, 0, None)
            .buffer_bytes(&adj_offset_buf, 0, None)
            .buffer_bytes(&adj_list_buf, 0, None)
            .build(device, &modal_cheb_step_data_layout),
    ];

    // Modal Chebyshev step params bind groups (one per order k)
    let modal_cheb_step_params_bgs: Vec<wgpu::BindGroup> = (0..MAX_CHEB_ORDER)
        .map(|k| {
            wgpu::BindGroupBuilder::new()
                .buffer_bytes(&sim_params_buf, 0, None)
                .buffer_bytes(&modal_coeff_lo_bufs[k], 0, None)
                .buffer_bytes(&modal_coeff_hi_bufs[k], 0, None)
                .build(device, &modal_step_params_layout)
        })
        .collect();

    // Modal phase update bind group
    let modal_phase_update_bg = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&modal_phase_buf, 0, None)
        .buffer_bytes(&audio_params_buf, 0, None)
        .buffer_bytes(&modal_freq_buf, 0, None)
        .buffer_bytes(&gra_state_buf, 0, None)
        .build(device, &modal_phase_update_layout);

    GpuCompute {
        particle_bufs,
        particle_frame: 0,
        p_bin_size_buf, p_bin_offset_buf, p_num_bins: p_num_bins + 1,
        gra_pos_buf, gra_vel_buf, gra_force_buf, gra_state_buf,
        adj_offset_buf, adj_list_buf, connection_buf,
        gra_sorted_pos_buf, g_bin_size_buf, g_bin_offset_buf, g_num_bins: g_num_bins + 1,
        sim_params_buf, audio_params_buf, render_uniform_buf,
        gra_pos_readback_bufs, gra_readback_frame: 0,
        bbox_atomic_buf, bbox_readback_buf,
        audio_out_buf, audio_staging_bufs,
        particle_sim_pipeline, p_clear_bins_pipeline, p_fill_bins_pipeline,
        p_prefix_sum_pipeline, p_sort_clear_pipeline, p_sort_pipeline,
        gra_spring_pipeline, gra_integrate_pipeline,
        g_clear_bins_pipeline, g_fill_bins_pipeline, g_prefix_sum_pipeline,
        g_sort_clear_pipeline, g_sort_pipeline,
        bbox_clear_pipeline, bbox_reduce_pipeline,
        audio_pipeline, particle_phase_pipeline,
        modal_phase_update_pipeline,
        particle_sim_bgs, particle_sim_gra_bg,
        p_bin_fill_bgs, p_bin_params_bg, p_bin_size_bg,
        p_prefix_bg, p_sort_data_bgs, p_sort_params_bg,
        gra_spring_data_bg, gra_spring_params_bg,
        gra_integrate_data_bg, gra_integrate_params_bg,
        gra_bin_fill_data_bg, gra_bin_params_bg, gra_bin_size_bg,
        g_prefix_bg, gra_sort_data_bg, gra_sort_params_bg,
        bbox_data_bg, bbox_params_bg,
        audio_bgs, particle_phase_bgs,
        modal_phase_update_bg,
        modal_amp_buf, modal_phase_buf, modal_freq_buf,
        modal_coeff_lo_bufs, modal_coeff_hi_bufs,
        cheb_a_buf, cheb_b_buf, cheb_c_buf,
        modal_cheb_init_pipeline, modal_cheb_step_pipeline,
        modal_cheb_init_data_bg, modal_cheb_init_params_bg,
        modal_cheb_step_data_bgs, modal_cheb_step_params_bgs,
        topology_dirty: true,
        num_gra_nodes: 0,
        num_gra_connections: 0,
        num_particles: 0,
    }
}

// ── Upload functions ─────────────────────────────────────────────────────────

pub(crate) fn upload_particles(queue: &wgpu::Queue, gpu: &mut GpuCompute, particles: &[Particle]) {
    let n = particles.len().min(MAX_PARTICLES);
    gpu.num_particles = n as u32;
    queue.write_buffer(&gpu.particle_bufs[gpu.particle_frame], 0, bytemuck::cast_slice(&particles[..n]));
}

pub(crate) fn upload_gra_topology(
    queue: &wgpu::Queue,
    gpu: &mut GpuCompute,
    positions: &[(f32, f32)],
    states: &[[f32; 3]],
    connections: &[(usize, usize)],
) {
    let n = positions.len().min(MAX_GRA_NODES);
    let num_conn = connections.len().min(MAX_GRA_CONNECTIONS);
    gpu.num_gra_nodes = n as u32;
    gpu.num_gra_connections = num_conn as u32;

    let pos_data: Vec<[f32; 4]> = positions[..n].iter()
        .map(|&(x, y)| [x, y, 0.0, 1.0])
        .collect();
    queue.write_buffer(&gpu.gra_pos_buf, 0, bytemuck::cast_slice(&pos_data));

    let state_data: Vec<[f32; 4]> = states[..n].iter()
        .map(|&[r, g, b]| [r, g, b, 0.0])
        .collect();
    queue.write_buffer(&gpu.gra_state_buf, 0, bytemuck::cast_slice(&state_data));

    let zeros = vec![[0.0f32; 4]; n];
    queue.write_buffer(&gpu.gra_vel_buf, 0, bytemuck::cast_slice(&zeros));

    // Build adjacency list
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
    gpu.num_gra_connections = (conn_flat.len() / 2) as u32;
    if !conn_flat.is_empty() {
        queue.write_buffer(&gpu.connection_buf, 0, bytemuck::cast_slice(&conn_flat));
    }
}

pub(crate) fn upload_gra_forces(queue: &wgpu::Queue, gpu: &GpuCompute, forces: &[(f32, f32)]) {
    let force_data: Vec<[f32; 4]> = forces.iter()
        .map(|&(fx, fy)| [fx, fy, 0.0, 0.0])
        .collect();
    queue.write_buffer(&gpu.gra_force_buf, 0, bytemuck::cast_slice(&force_data));
}

/// Per-node trial visual/audio params passed from the trial system.
pub(crate) struct NodeTrialInfo {
    pub hue_base: f32,
    pub freq_scale: f32,
}

pub(crate) fn upload_gra_discrete_states(
    queue: &wgpu::Queue,
    gpu: &GpuCompute,
    discrete_states: &[u8],
    num_states: u8,
    trial_info: &[NodeTrialInfo],
) {
    let state_data: Vec<[f32; 4]> = discrete_states.iter()
        .enumerate()
        .map(|(i, &s)| {
            let info = &trial_info[i];
            let t = (s as f32) / ((num_states.max(2) - 1) as f32);
            let hue = ((info.hue_base + t * 110.0) % 360.0) / 360.0;
            let (r, g, b) = hsv_to_rgb(hue, 0.85, 0.95);
            [r, g, b, info.freq_scale]
        })
        .collect();
    queue.write_buffer(&gpu.gra_state_buf, 0, bytemuck::cast_slice(&state_data));
}

pub(crate) fn update_sim_params(queue: &wgpu::Queue, gpu: &GpuCompute, params: &SimParams) {
    queue.write_buffer(&gpu.sim_params_buf, 0, bytemuck::bytes_of(params));
}

pub(crate) fn update_audio_params(queue: &wgpu::Queue, gpu: &GpuCompute, params: &AudioParams) {
    queue.write_buffer(&gpu.audio_params_buf, 0, bytemuck::bytes_of(params));
}

pub(crate) fn update_render_uniforms(queue: &wgpu::Queue, gpu: &GpuCompute, uniforms: &RenderUniforms) {
    queue.write_buffer(&gpu.render_uniform_buf, 0, bytemuck::bytes_of(uniforms));
}

// ── Dispatch: particle spatial hash + sort ───────────────────────────────────

pub(crate) fn dispatch_particle_bin_sort(
    encoder: &mut wgpu::CommandEncoder,
    gpu: &mut GpuCompute,
) {
    let src = gpu.particle_frame;
    let dst = 1 - src;
    let num_particles = gpu.num_particles;

    // 1. Clear bin sizes
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("p_clear_bins"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.p_clear_bins_pipeline);
        pass.set_bind_group(0, &gpu.p_bin_fill_bgs[src], &[]);
        pass.set_bind_group(1, &gpu.p_bin_params_bg, &[]);
        pass.set_bind_group(2, &gpu.p_bin_size_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(gpu.p_num_bins, WORKGROUP_SIZE), 1, 1);
    }

    // 2. Fill bin sizes
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("p_fill_bins"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.p_fill_bins_pipeline);
        pass.set_bind_group(0, &gpu.p_bin_fill_bgs[src], &[]);
        pass.set_bind_group(1, &gpu.p_bin_params_bg, &[]);
        pass.set_bind_group(2, &gpu.p_bin_size_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(num_particles, WORKGROUP_SIZE), 1, 1);
    }

    // 3. Prefix sum
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("p_prefix_sum"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.p_prefix_sum_pipeline);
        pass.set_bind_group(0, &gpu.p_prefix_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // 4. Clear sort bins
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("p_sort_clear"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.p_sort_clear_pipeline);
        pass.set_bind_group(0, &gpu.p_sort_data_bgs[src], &[]);
        pass.set_bind_group(1, &gpu.p_sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(gpu.p_num_bins, WORKGROUP_SIZE), 1, 1);
    }

    // 5. Sort particles
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("p_sort"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.p_sort_pipeline);
        pass.set_bind_group(0, &gpu.p_sort_data_bgs[src], &[]);
        pass.set_bind_group(1, &gpu.p_sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(num_particles, WORKGROUP_SIZE), 1, 1);
    }

    // Flip: sorted data is now in buf[dst]
    gpu.particle_frame = dst;
}

// ── Dispatch: GRA spatial hash + sort ────────────────────────────────────────

pub(crate) fn dispatch_gra_bin_sort(
    encoder: &mut wgpu::CommandEncoder,
    gpu: &GpuCompute,
) {
    let num_gra = gpu.num_gra_nodes;

    // 1. Clear
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("g_clear_bins"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.g_clear_bins_pipeline);
        pass.set_bind_group(0, &gpu.gra_bin_fill_data_bg, &[]);
        pass.set_bind_group(1, &gpu.gra_bin_params_bg, &[]);
        pass.set_bind_group(2, &gpu.gra_bin_size_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(gpu.g_num_bins, WORKGROUP_SIZE), 1, 1);
    }

    // 2. Fill
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("g_fill_bins"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.g_fill_bins_pipeline);
        pass.set_bind_group(0, &gpu.gra_bin_fill_data_bg, &[]);
        pass.set_bind_group(1, &gpu.gra_bin_params_bg, &[]);
        pass.set_bind_group(2, &gpu.gra_bin_size_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(num_gra, WORKGROUP_SIZE), 1, 1);
    }

    // 3. Prefix sum
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("g_prefix_sum"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.g_prefix_sum_pipeline);
        pass.set_bind_group(0, &gpu.g_prefix_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // 4. Clear sort bins
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("g_sort_clear"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.g_sort_clear_pipeline);
        pass.set_bind_group(0, &gpu.gra_sort_data_bg, &[]);
        pass.set_bind_group(1, &gpu.gra_sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(gpu.g_num_bins, WORKGROUP_SIZE), 1, 1);
    }

    // 5. Sort
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("g_sort"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.g_sort_pipeline);
        pass.set_bind_group(0, &gpu.gra_sort_data_bg, &[]);
        pass.set_bind_group(1, &gpu.gra_sort_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(num_gra, WORKGROUP_SIZE), 1, 1);
    }
}

// ── Dispatch: particle physics ───────────────────────────────────────────────

pub(crate) fn dispatch_particle_sim(
    encoder: &mut wgpu::CommandEncoder,
    gpu: &GpuCompute,
) {
    // Particle sim reads from sorted buffer (current frame) and writes back in-place
    let frame = gpu.particle_frame;
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("particle_sim"), timestamp_writes: None,
    });
    pass.set_pipeline(&gpu.particle_sim_pipeline);
    pass.set_bind_group(0, &gpu.particle_sim_bgs[frame], &[]);
    pass.set_bind_group(1, &gpu.particle_sim_gra_bg, &[]);
    pass.dispatch_workgroups(dispatch_count(gpu.num_particles, WORKGROUP_SIZE), 1, 1);
}

// ── Dispatch: GRA physics ────────────────────────────────────────────────────

pub(crate) fn dispatch_gra_physics(
    encoder: &mut wgpu::CommandEncoder,
    gpu: &GpuCompute,
    readback_idx: usize,
) {
    let n = gpu.num_gra_nodes;

    // Spring forces
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gra_spring"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.gra_spring_pipeline);
        pass.set_bind_group(0, &gpu.gra_spring_data_bg, &[]);
        pass.set_bind_group(1, &gpu.gra_spring_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n, GRA_WORKGROUP_SIZE), 1, 1);
    }

    // Integrate
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gra_integrate"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.gra_integrate_pipeline);
        pass.set_bind_group(0, &gpu.gra_integrate_data_bg, &[]);
        pass.set_bind_group(1, &gpu.gra_integrate_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n, GRA_WORKGROUP_SIZE), 1, 1);
    }

    // Bbox
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
        pass.dispatch_workgroups(dispatch_count(n, GRA_WORKGROUP_SIZE), 1, 1);
    }

    // Copy for readback
    encoder.copy_buffer_to_buffer(&gpu.bbox_atomic_buf, 0, &gpu.bbox_readback_buf, 0, 16);
    let pos_copy_size = (n as u64) * 16;
    encoder.copy_buffer_to_buffer(&gpu.gra_pos_buf, 0, &gpu.gra_pos_readback_bufs[readback_idx], 0, pos_copy_size);
}

// ── Dispatch: audio ──────────────────────────────────────────────────────────

pub(crate) fn encode_audio_pass(
    encoder: &mut wgpu::CommandEncoder,
    gpu: &GpuCompute,
    staging_idx: usize,
) {
    let frame = gpu.particle_frame;
    let audio_buf_size = (CHUNK_SIZE * NUM_CHANNELS * 4) as u64;

    // Combined particle + GRA modal audio synthesis
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("audio"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.audio_pipeline);
        pass.set_bind_group(0, &gpu.audio_bgs[frame], &[]);
        pass.dispatch_workgroups(dispatch_count(CHUNK_SIZE, WORKGROUP_SIZE), 1, 1);
    }

    // Particle phase update
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_phase"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.particle_phase_pipeline);
        pass.set_bind_group(0, &gpu.particle_phase_bgs[frame], &[]);
        pass.dispatch_workgroups(dispatch_count(gpu.num_particles, WORKGROUP_SIZE), 1, 1);
    }

    // Modal phase update (GRA)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("modal_phase_update"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.modal_phase_update_pipeline);
        pass.set_bind_group(0, &gpu.modal_phase_update_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(gpu.num_gra_nodes, GRA_WORKGROUP_SIZE), 1, 1);
    }

    // Copy to staging
    encoder.copy_buffer_to_buffer(
        &gpu.audio_out_buf, 0,
        &gpu.audio_staging_bufs[staging_idx], 0,
        audio_buf_size,
    );
}

// ── Modal synthesis functions ────────────────────────────────────────────

pub(crate) fn upload_modal_coefficients(
    queue: &wgpu::Queue,
    gpu: &GpuCompute,
    coeffs_lo: &[[f32; 4]],
    coeffs_hi: &[[f32; 4]],
) {
    let order = coeffs_lo.len().min(MAX_CHEB_ORDER);
    for k in 0..order {
        queue.write_buffer(&gpu.modal_coeff_lo_bufs[k], 0, bytemuck::cast_slice(&coeffs_lo[k]));
        queue.write_buffer(&gpu.modal_coeff_hi_bufs[k], 0, bytemuck::cast_slice(&coeffs_hi[k]));
    }
}

pub(crate) fn update_modal_freqs(queue: &wgpu::Queue, gpu: &GpuCompute, freqs: &ModalFreqs) {
    queue.write_buffer(&gpu.modal_freq_buf, 0, bytemuck::bytes_of(freqs));
}

pub(crate) fn encode_modal_chebyshev(
    encoder: &mut wgpu::CommandEncoder,
    gpu: &GpuCompute,
    cheb_order: usize,
) {
    let n = gpu.num_gra_nodes;

    // Init: T_0 = signal, T_1 = W(signal), modal_amp = c0*T_0 + c1*T_1
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("modal_cheb_init"), timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.modal_cheb_init_pipeline);
        pass.set_bind_group(0, &gpu.modal_cheb_init_data_bg, &[]);
        pass.set_bind_group(1, &gpu.modal_cheb_init_params_bg, &[]);
        pass.dispatch_workgroups(dispatch_count(n, GRA_WORKGROUP_SIZE), 1, 1);
    }

    // Steps k=2..order: t_next = 2*W(t_curr) - t_prev, modal_amp += ck*t_next
    for k in 2..cheb_order {
        let bg_idx = (k - 2) % 3;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("modal_cheb_step"), timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.modal_cheb_step_pipeline);
            pass.set_bind_group(0, &gpu.modal_cheb_step_data_bgs[bg_idx], &[]);
            pass.set_bind_group(1, &gpu.modal_cheb_step_params_bgs[k], &[]);
            pass.dispatch_workgroups(dispatch_count(n, GRA_WORKGROUP_SIZE), 1, 1);
        }
    }
}

// ── Readback ─────────────────────────────────────────────────────────────────

pub(crate) fn readback_gra_positions(device: &wgpu::Device, gpu: &GpuCompute, num_nodes: usize, readback_idx: usize) -> Vec<(f32, f32)> {
    let pos_size = (num_nodes * 16) as u64;
    let buf = &gpu.gra_pos_readback_bufs[readback_idx];
    let slice = buf.slice(..pos_size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait).unwrap();
    let data = slice.get_mapped_range();
    let floats: &[[f32; 4]] = bytemuck::cast_slice(&data);
    let positions: Vec<(f32, f32)> = floats[..num_nodes].iter().map(|f| (f[0], f[1])).collect();
    drop(data);
    buf.unmap();
    positions
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}
