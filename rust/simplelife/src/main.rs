use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::VecDeque;
use std::thread::JoinHandle;

use bevy::prelude::*;
use bevy::log::info;
use bevy::render::render_graph::{self, RenderGraph, RenderLabel};
use bevy::render::render_resource::binding_types::{storage_buffer, storage_buffer_read_only, uniform_buffer};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::render::view::ViewTarget;
use bevy::ecs::query::QueryItem;
use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};

use crossbeam_channel::{self, Receiver};
use ringbuf::{traits::{Consumer, Split, Observer}, HeapRb};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

const NUM_PARTICLES: u32 = 64;
const NUM_CHANNELS: u32 = 2;
const SAMPLE_RATE: f32 = 44100.0;
const CHUNK_SIZE: u32 = 2048;
const CHUNK_FLOATS: u32 = CHUNK_SIZE * NUM_CHANNELS;
const INITIAL_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 2;
const MIN_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS;
const MAX_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 16;
const BUFFER_HISTORY_SIZE: usize = 64;
const WORKGROUP_SIZE: u32 = 64;

const COMMON_SHADER_PATH: &str = "shaders/common.wgsl";
const PARTICLE_SHADER_PATH: &str = "shaders/particle.wgsl";
const AUDIO_SHADER_PATH: &str = "shaders/audio.wgsl";
const RENDER_SHADER_PATH: &str = "shaders/render.wgsl";

// GPU-compatible types matching the shader structs
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    phase: f32,
    energy: f32,
    species: [f32; 2],
    alpha: [f32; 2],
    interaction: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
struct SimParams {
    dt: f32,
    time: f32,
    num_particles: u32,
    friction: f32,
    mass: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    volume: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
struct RenderParams {
    screen_size: [f32; 2],
    particle_size: f32,
    num_particles: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexInput {
    position: [f32; 2],
    uv: [f32; 2],
}

// Message from audio thread
struct AudioRequest {
    chunks_needed: u32,
    min_buffer_level: u32,
    current_buffer_level: u32,
}

// Simulation state (extracted to render world)
#[derive(Resource, Clone, ExtractResource)]
struct SimState {
    time: f32,
    request_threshold: u32,
    integral_error: f32,
    chunks_needed: u32,
}

// Audio resources shared between threads
struct AudioShared {
    #[allow(dead_code)]
    producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
    request_rx: Receiver<AudioRequest>,
}

#[derive(Resource)]
struct AudioResources {
    shared: Arc<Mutex<AudioShared>>,
    _thread: JoinHandle<()>,
}

// Shared audio producer for render world access
static AUDIO_PRODUCER: std::sync::OnceLock<Arc<Mutex<ringbuf::HeapProd<f32>>>> = std::sync::OnceLock::new();

// Logging counters
static COMPUTE_NODE_RUN_COUNT: AtomicU64 = AtomicU64::new(0);
static RENDER_NODE_RUN_COUNT: AtomicU64 = AtomicU64::new(0);
static AUDIO_CALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

const MAX_AUDIO_CHUNKS_PER_FRAME: usize = 4;

// Track pending audio readbacks
#[derive(Resource, Default)]
struct AudioReadbackState {
    chunks_pending: u32, // How many chunks need readback this frame
}

// Render world resources
#[derive(Resource)]
struct ParticleBuffers {
    particles: Buffer,
    sim_params: Buffer,
    audio_out: Buffer,
    audio_params: Buffer,
    render_params: Buffer,
    vertex_buffer: Buffer,
    audio_staging: Vec<Buffer>, // Multiple staging buffers for audio readback
}

#[derive(Resource)]
struct ParticleBindGroups {
    particle_compute: BindGroup,
    audio_compute: BindGroup,
    render: BindGroup,
}

#[derive(Resource)]
// Unused but kept for potential future use
#[allow(dead_code)]
struct ParticlePipelines {
    particle_compute: CachedComputePipelineId,
    audio_compute: CachedComputePipelineId,
    render: CachedRenderPipelineId,
}

// Render graph labels
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ParticleComputeLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ParticleRenderLabel;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "simplelife".to_string(),
                resolution: (800, 800).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(ParticlePlugin)
        .run();
}

struct ParticlePlugin;

impl Plugin for ParticlePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_main_world)
            .add_systems(Update, (update_simulation, handle_audio_requests))
            .add_plugins(ExtractResourcePlugin::<SimState>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(Render, prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups))
            .add_systems(Render, update_buffers.in_set(RenderSystems::Prepare))
            .add_systems(Render, poll_device.in_set(RenderSystems::Cleanup));
    }

    fn finish(&self, app: &mut App) {
        use bevy::render::render_graph::{RenderGraphExt, ViewNodeRunner};

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ParticlePipelinesHolder>();

        // Set up render graph - add compute node to main graph
        {
            let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
            render_graph.add_node(ParticleComputeLabel, ParticleComputeNode::default());
            render_graph.add_node_edge(ParticleComputeLabel, bevy::render::graph::CameraDriverLabel);
        }

        // Add render node to the 2D camera sub-graph using ViewNodeRunner
        render_app
            .add_render_graph_node::<ViewNodeRunner<ParticleRenderNode>>(Core2d, ParticleRenderLabel)
            .add_render_graph_edge(Core2d, Node2d::MainTransparentPass, ParticleRenderLabel)
            .add_render_graph_edge(Core2d, ParticleRenderLabel, Node2d::EndMainPass);
    }
}

// Temporary holder for pipeline IDs during initialization
#[derive(Resource)]
struct ParticlePipelinesHolder {
    particle_compute: CachedComputePipelineId,
    audio_compute: CachedComputePipelineId,
    render: CachedRenderPipelineId,
}

impl FromWorld for ParticlePipelinesHolder {
    fn from_world(world: &mut World) -> Self {
        info!("ParticlePipelinesHolder::from_world - Starting render world initialization");

        let render_device = world.resource::<RenderDevice>().clone();
        let render_queue = world.resource::<RenderQueue>().clone();
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();

        // Create bind group layouts
        let particle_bind_group_layout = render_device.create_bind_group_layout(
            Some("particle_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<Particle>>(false), // particles read-write
                    uniform_buffer::<SimParams>(false),     // sim params
                ),
            ),
        );

        let audio_bind_group_layout = render_device.create_bind_group_layout(
            Some("audio_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only::<Vec<Particle>>(false), // particles read-only
                    storage_buffer::<Vec<[f32; 2]>>(false),           // audio_out
                    uniform_buffer::<AudioParams>(false),             // audio params
                ),
            ),
        );

        let render_bind_group_layout = render_device.create_bind_group_layout(
            Some("render_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    storage_buffer_read_only::<Vec<Particle>>(false), // particles
                    uniform_buffer::<RenderParams>(false),            // render params
                ),
            ),
        );

        // Load shaders and create pipelines
        let particle_shader = asset_server.load(PARTICLE_SHADER_PATH);
        let audio_shader = asset_server.load(AUDIO_SHADER_PATH);
        let render_shader = asset_server.load(RENDER_SHADER_PATH);

        let particle_compute = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("particle_compute_pipeline")),
            layout: vec![particle_bind_group_layout.clone()],
            shader: particle_shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("main")),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        let audio_compute = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("audio_compute_pipeline")),
            layout: vec![audio_bind_group_layout.clone()],
            shader: audio_shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("main")),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        let render = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(Cow::from("particle_render_pipeline")),
            layout: vec![render_bind_group_layout.clone()],
            vertex: VertexState {
                shader: render_shader.clone(),
                shader_defs: vec![],
                entry_point: Some(Cow::from("vs_main")),
                buffers: vec![bevy::mesh::VertexBufferLayout::from_vertex_formats(
                    VertexStepMode::Vertex,
                    [VertexFormat::Float32x2, VertexFormat::Float32x2], // position, uv
                )],
            },
            fragment: Some(FragmentState {
                shader: render_shader,
                shader_defs: vec![],
                entry_point: Some(Cow::from("fs_main")),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 4, // Match Bevy's default MSAA setting
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        // Store layouts in world for bind group creation
        world.insert_resource(ParticleBindGroupLayouts {
            particle_compute: particle_bind_group_layout,
            audio_compute: audio_bind_group_layout,
            render: render_bind_group_layout,
        });

        // Initialize buffers
        let particles: Vec<Particle> = (0..NUM_PARTICLES)
            .map(|i| {
                let angle = (i as f32 / NUM_PARTICLES as f32) * std::f32::consts::TAU;
                let r = 0.3 + rand::random::<f32>() * 0.2;
                Particle {
                    pos: [angle.cos() * r, angle.sin() * r],
                    vel: [0.0, 0.0],
                    phase: rand::random::<f32>(),
                    energy: 0.0,
                    species: [0.0, 0.0],
                    alpha: [0.0, 0.0],
                    interaction: [0.0, 0.0],
                }
            })
            .collect();

        let particles_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let sim_params_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("sim_params"),
            size: std::mem::size_of::<SimParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let audio_buf_size = (CHUNK_FLOATS * std::mem::size_of::<f32>() as u32) as u64;
        let audio_out_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("audio_out"),
            size: audio_buf_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let audio_staging_bufs: Vec<Buffer> = (0..MAX_AUDIO_CHUNKS_PER_FRAME)
            .map(|i| {
                render_device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("audio_staging_{}", i)),
                    size: audio_buf_size,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let audio_params_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("audio_params"),
            size: std::mem::size_of::<AudioParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_params_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("render_params"),
            size: std::mem::size_of::<RenderParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let quad_vertices = [
            VertexInput { position: [-1.0, -1.0], uv: [0.0, 1.0] },
            VertexInput { position: [1.0, -1.0], uv: [1.0, 1.0] },
            VertexInput { position: [1.0, 1.0], uv: [1.0, 0.0] },
            VertexInput { position: [-1.0, -1.0], uv: [0.0, 1.0] },
            VertexInput { position: [1.0, 1.0], uv: [1.0, 0.0] },
            VertexInput { position: [-1.0, 1.0], uv: [0.0, 0.0] },
        ];

        let vertex_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("particle_quad_vertices"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: BufferUsages::VERTEX,
        });

        // Write initial audio params
        render_queue.write_buffer(
            &audio_params_buf,
            0,
            bytemuck::bytes_of(&AudioParams {
                sample_rate: SAMPLE_RATE,
                num_particles: NUM_PARTICLES,
                chunk_size: CHUNK_SIZE,
                volume: 0.8,
            }),
        );

        world.insert_resource(ParticleBuffers {
            particles: particles_buf,
            sim_params: sim_params_buf,
            audio_out: audio_out_buf,
            audio_params: audio_params_buf,
            render_params: render_params_buf,
            vertex_buffer,
            audio_staging: audio_staging_bufs,
        });

        info!("ParticlePipelinesHolder::from_world - Initialization complete, pipelines queued");

        ParticlePipelinesHolder {
            particle_compute,
            audio_compute,
            render,
        }
    }
}

#[derive(Resource)]
struct ParticleBindGroupLayouts {
    particle_compute: BindGroupLayout,
    audio_compute: BindGroupLayout,
    render: BindGroupLayout,
}

fn prepare_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    buffers: Option<Res<ParticleBuffers>>,
    layouts: Option<Res<ParticleBindGroupLayouts>>,
    pipelines: Option<Res<ParticlePipelinesHolder>>,
    existing_bind_groups: Option<Res<ParticleBindGroups>>,
) {
    // Only create once
    if existing_bind_groups.is_some() {
        return;
    }

    let (Some(buffers), Some(layouts), Some(_pipelines)) = (buffers, layouts, pipelines) else {
        return;
    };

    info!("prepare_bind_groups - Creating bind groups");

    let particle_compute_bind_group = render_device.create_bind_group(
        Some("particle_compute_bind_group"),
        &layouts.particle_compute,
        &BindGroupEntries::sequential((
            buffers.particles.as_entire_binding(),
            buffers.sim_params.as_entire_binding(),
        )),
    );

    let audio_compute_bind_group = render_device.create_bind_group(
        Some("audio_compute_bind_group"),
        &layouts.audio_compute,
        &BindGroupEntries::sequential((
            buffers.particles.as_entire_binding(),
            buffers.audio_out.as_entire_binding(),
            buffers.audio_params.as_entire_binding(),
        )),
    );

    let render_bind_group = render_device.create_bind_group(
        Some("render_bind_group"),
        &layouts.render,
        &BindGroupEntries::sequential((
            buffers.particles.as_entire_binding(),
            buffers.render_params.as_entire_binding(),
        )),
    );

    commands.insert_resource(ParticleBindGroups {
        particle_compute: particle_compute_bind_group,
        audio_compute: audio_compute_bind_group,
        render: render_bind_group,
    });
}

fn update_buffers(
    render_queue: Res<RenderQueue>,
    buffers: Option<Res<ParticleBuffers>>,
    sim_state: Option<Res<SimState>>,
) {
    let (Some(buffers), Some(sim)) = (buffers, sim_state) else {
        return;
    };

    let dt = 1.0 / 60.0;
    let sim_params = SimParams {
        dt,
        time: sim.time,
        num_particles: NUM_PARTICLES,
        friction: 0.1,
        mass: 1.0,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };
    render_queue.write_buffer(&buffers.sim_params, 0, bytemuck::bytes_of(&sim_params));

    let render_params = RenderParams {
        screen_size: [800.0, 800.0],
        particle_size: 12.0,
        num_particles: NUM_PARTICLES,
    };
    render_queue.write_buffer(&buffers.render_params, 0, bytemuck::bytes_of(&render_params));
}

fn poll_device(render_device: Res<RenderDevice>) {
    // Poll the device to process async buffer mapping callbacks
    render_device.poll(wgpu::PollType::Poll).ok();
}

// Compute node for particle simulation
#[derive(Default)]
struct ParticleComputeNode {
    state: ParticleComputeState,
}

enum ParticleComputeState {
    Loading,
    Ready,
}

impl Default for ParticleComputeState {
    fn default() -> Self {
        ParticleComputeState::Loading
    }
}

impl render_graph::Node for ParticleComputeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipelines = world.resource::<ParticlePipelinesHolder>();

        match self.state {
            ParticleComputeState::Loading => {
                let particle_state = pipeline_cache.get_compute_pipeline_state(pipelines.particle_compute);
                let audio_state = pipeline_cache.get_compute_pipeline_state(pipelines.audio_compute);

                info!("ComputeNode::update - Loading state. Particle pipeline: {:?}, Audio pipeline: {:?}",
                    particle_state, audio_state);

                if let CachedPipelineState::Ok(_) = particle_state {
                    if let CachedPipelineState::Ok(_) = audio_state {
                        info!("ComputeNode::update - Both pipelines ready! Transitioning to Ready state");
                        self.state = ParticleComputeState::Ready;
                    }
                }
            }
            ParticleComputeState::Ready => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let count = COMPUTE_NODE_RUN_COUNT.fetch_add(1, Ordering::Relaxed);

        if !matches!(self.state, ParticleComputeState::Ready) {
            if count % 60 == 0 {
                info!("ComputeNode::run - Not ready yet (frame {})", count);
            }
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipelines = world.resource::<ParticlePipelinesHolder>();
        let bind_groups = world.get_resource::<ParticleBindGroups>();
        let buffers = world.get_resource::<ParticleBuffers>();
        let sim_state = world.get_resource::<SimState>();

        let (Some(bind_groups), Some(buffers), Some(sim_state)) = (bind_groups, buffers, sim_state) else {
            if count % 60 == 0 {
                info!("ComputeNode::run - Missing resources: bind_groups={}, buffers={}, sim_state={}",
                    bind_groups.is_some(), buffers.is_some(), sim_state.is_some());
            }
            return Ok(());
        };

        let Some(particle_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.particle_compute) else {
            if count % 60 == 0 {
                info!("ComputeNode::run - Particle pipeline not ready");
            }
            return Ok(());
        };
        let Some(audio_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.audio_compute) else {
            if count % 60 == 0 {
                info!("ComputeNode::run - Audio pipeline not ready");
            }
            return Ok(());
        };

        if count % 60 == 0 {
            info!("ComputeNode::run - EXECUTING compute pass (frame {}), chunks_needed={}", count, sim_state.chunks_needed);
        }

        let encoder = render_context.command_encoder();

        // Particle simulation compute pass
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("particle_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(particle_pipeline);
            pass.set_bind_group(0, &bind_groups.particle_compute, &[]);
            pass.dispatch_workgroups((NUM_PARTICLES + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
        }

        // Audio compute passes - generate multiple chunks
        let chunks_needed = (sim_state.chunks_needed as usize).min(MAX_AUDIO_CHUNKS_PER_FRAME);
        let audio_buf_size = (CHUNK_FLOATS * std::mem::size_of::<f32>() as u32) as u64;

        for i in 0..chunks_needed {
            // Run audio compute shader
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("audio_compute_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(audio_pipeline);
                pass.set_bind_group(0, &bind_groups.audio_compute, &[]);
                pass.dispatch_workgroups((CHUNK_SIZE + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
            }

            // Copy GPU audio data to staging buffer
            encoder.copy_buffer_to_buffer(
                &buffers.audio_out,
                0,
                &buffers.audio_staging[i],
                0,
                audio_buf_size,
            );
        }

        // Spawn async tasks to read back audio data from staging buffers
        for i in 0..chunks_needed {
            let staging = buffers.audio_staging[i].clone();
            let staging_for_callback = staging.clone();
            let chunk_idx = i;

            // Map the buffer asynchronously
            staging.slice(..).map_async(bevy::render::render_resource::MapMode::Read, move |result| {
                match result {
                    Ok(()) => {
                        // Read the data in the callback
                        let data = staging_for_callback.slice(..).get_mapped_range();
                        let audio_samples: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                        drop(data);
                        staging_for_callback.unmap();

                        // Push to ring buffer
                        if let Some(producer) = AUDIO_PRODUCER.get() {
                            if let Ok(mut prod) = producer.lock() {
                                use ringbuf::traits::Producer;
                                let pushed = prod.push_slice(&audio_samples);
                                eprintln!("Audio callback chunk {}: pushed {} samples", chunk_idx, pushed);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Audio buffer map error for chunk {}: {:?}", chunk_idx, e);
                    }
                }
            });
        }

        Ok(())
    }
}

// Render node for particle rendering using ViewNode to integrate with camera
#[derive(Default)]
struct ParticleRenderNode;

impl bevy::render::render_graph::ViewNode for ParticleRenderNode {
    type ViewQuery = &'static ViewTarget;

    fn run<'w>(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        view_target: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), render_graph::NodeRunError> {
        let count = RENDER_NODE_RUN_COUNT.fetch_add(1, Ordering::Relaxed);

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipelines = world.get_resource::<ParticlePipelinesHolder>();
        let bind_groups = world.get_resource::<ParticleBindGroups>();
        let buffers = world.get_resource::<ParticleBuffers>();

        let (Some(pipelines), Some(bind_groups), Some(buffers)) = (pipelines, bind_groups, buffers) else {
            if count % 60 == 0 {
                info!("RenderNode::run - Missing resources: pipelines={}, bind_groups={}, buffers={}",
                    pipelines.is_some(), bind_groups.is_some(), buffers.is_some());
            }
            return Ok(());
        };

        let render_pipeline_state = pipeline_cache.get_render_pipeline_state(pipelines.render);
        let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipelines.render) else {
            if count % 60 == 0 {
                info!("RenderNode::run - Render pipeline not ready: {:?}", render_pipeline_state);
            }
            return Ok(());
        };

        if count % 60 == 0 {
            info!("RenderNode::run - EXECUTING render pass (frame {}), drawing {} particles", count, NUM_PARTICLES);
        }

        let color_attachment = view_target.get_color_attachment();

        let encoder = render_context.command_encoder();

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("particle_render_pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(render_pipeline);
            render_pass.set_bind_group(0, &bind_groups.render, &[]);
            render_pass.set_vertex_buffer(0, *buffers.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..NUM_PARTICLES);
        }

        Ok(())
    }
}

fn setup_main_world(mut commands: Commands) {
    info!("setup_main_world - Starting main world setup");

    // Create ring buffer for lock-free audio transfer
    let ring_buf_size = CHUNK_FLOATS as usize * 16;
    let ring_buf = HeapRb::<f32>::new(ring_buf_size);
    let (audio_producer, audio_consumer) = ring_buf.split();

    // Store producer in static for render world access
    let producer = Arc::new(Mutex::new(audio_producer));
    AUDIO_PRODUCER.set(producer.clone()).ok();

    // Start audio stream
    let (audio_request_tx, audio_request_rx) = crossbeam_channel::unbounded();

    let audio_thread = std::thread::spawn(move || {
        let host = cpal::default_host();
        let audio_device = host.default_output_device().expect("No output device");
        let config = audio_device.default_output_config().expect("No default config");

        let mut consumer = audio_consumer;
        let request_tx = audio_request_tx;
        let mut buffer_history: VecDeque<u32> = VecDeque::with_capacity(BUFFER_HISTORY_SIZE);
        let request_threshold = INITIAL_REQUEST_THRESHOLD;

        let stream = audio_device.build_output_stream(
            &config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let callback_count = AUDIO_CALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                let buffer_len = consumer.occupied_len() as u32;

                buffer_history.push_back(buffer_len);
                if buffer_history.len() > BUFFER_HISTORY_SIZE {
                    buffer_history.pop_front();
                }
                let min_buffer = buffer_history.iter().copied().min().unwrap_or(0);

                let chunks_needed = if buffer_len < request_threshold {
                    let deficit = request_threshold - buffer_len;
                    (deficit / CHUNK_FLOATS).max(1)
                } else {
                    0
                };

                if callback_count % 100 == 0 {
                    eprintln!("AudioCallback #{}: buffer_len={}, requesting {} samples, data.len={}",
                        callback_count, buffer_len, data.len(), data.len());
                }

                request_tx.send(AudioRequest {
                    chunks_needed,
                    min_buffer_level: min_buffer,
                    current_buffer_level: buffer_len,
                }).ok();

                for sample in data.iter_mut() {
                    *sample = consumer.try_pop().unwrap_or(0.0);
                }
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        ).expect("Failed to build stream");

        stream.play().expect("Failed to play stream");

        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });

    commands.insert_resource(AudioResources {
        shared: Arc::new(Mutex::new(AudioShared {
            producer,
            request_rx: audio_request_rx,
        })),
        _thread: audio_thread,
    });

    commands.insert_resource(SimState {
        time: 0.0,
        request_threshold: INITIAL_REQUEST_THRESHOLD,
        integral_error: 0.0,
        chunks_needed: 0,
    });

    // Spawn a 2D camera - our particle render node integrates with the camera's render graph
    commands.spawn(Camera2d::default());

    info!("setup_main_world - Main world setup complete, camera spawned");
}

fn update_simulation(
    time: Res<Time>,
    mut sim: ResMut<SimState>,
) {
    sim.time += time.delta_secs();
}

fn handle_audio_requests(
    mut sim: ResMut<SimState>,
    audio: Res<AudioResources>,
) {
    let shared = audio.shared.lock().unwrap();

    let mut latest_request: Option<AudioRequest> = None;
    while let Ok(request) = shared.request_rx.try_recv() {
        latest_request = Some(request);
    }

    const MAX_CHUNKS_PER_FRAME: u32 = 4;
    let total_chunks_needed = latest_request
        .as_ref()
        .map(|r| r.chunks_needed.min(MAX_CHUNKS_PER_FRAME))
        .unwrap_or(0);

    sim.chunks_needed = total_chunks_needed;

    if let Some(request) = &latest_request {
        let min_buffer = request.min_buffer_level as f32;
        let target = (CHUNK_FLOATS * 2) as f32;

        eprintln!(
            "Audio: buffer={}, min={}, threshold={}, chunks_requested={}",
            request.current_buffer_level, request.min_buffer_level,
            sim.request_threshold, total_chunks_needed
        );

        let error = target - min_buffer;
        let kp = 0.01;
        let ki = 0.002;

        sim.integral_error = (sim.integral_error + error).clamp(-5000.0, 5000.0);
        let adjustment = kp * error + ki * sim.integral_error;

        let new_threshold = ((sim.request_threshold as f32) + adjustment) as u32;
        sim.request_threshold = new_threshold.clamp(MIN_REQUEST_THRESHOLD, MAX_REQUEST_THRESHOLD);
    }
}
