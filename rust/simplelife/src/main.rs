use nannou::prelude::*;
use nannou::wgpu::BufferInitDescriptor;
use nannou_audio as audio;
use nannou_audio::Buffer;
use std::thread::{JoinHandle, current};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use crossbeam_channel::{self, Receiver};
use ringbuf::{traits::{Consumer, Producer, Split, Observer}, HeapRb};

const NUM_PARTICLES: u32 = 1;
const NUM_CHANNELS: u32 = 2;
const SAMPLE_RATE: u32 = 22050;
const CHUNK_SIZE: u32 = 2048;
const CHUNK_FLOATS: u32 = CHUNK_SIZE * NUM_CHANNELS;
const INITIAL_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 2;
const MIN_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS;
const MAX_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 4;
const BUFFER_HISTORY_SIZE: usize = 64;
const PARTICLE_SIZE: f32 = 0.05;
const PAN_SPEED: f32 = 0.2;
const ZOOM_SPEED: f32 = 0.1;
const AUDIO_LOG_INTERVAL_FRAMES: u64 = 60; // Log audio stats every ~1 second at 60fps

// Shader sources (loaded at compile time)
const PARTICLE_SHADER: &str = include_str!("shaders/particle.wgsl");
const AUDIO_SHADER: &str = include_str!("shaders/audio.wgsl");
const RENDER_SHADER: &str = include_str!("shaders/render.wgsl");
const PHASE_UPDATE_SHADER: &str = include_str!("shaders/phase_update.wgsl");

// Particle struct - must match shader layout exactly
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    dt: f32,
    time: f32,
    num_particles: u32,
    friction: f32,
    mass: f32,
    map_x0: f32,
    map_x1: f32,
    map_y0: f32,
    map_y1: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    volume: f32,
    current_x: [f32; 2],  // Current viewport x (min, max)
    current_y: [f32; 2],  // Current viewport y (min, max)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PhaseUpdateParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    map_x: [f32; 2],      // Simulation bounds x (min, max)
    map_y: [f32; 2],      // Simulation bounds y (min, max)
    current_x: [f32; 2],  // Current viewport x (min, max)
    current_y: [f32; 2],  // Current viewport y (min, max)
    particle_size: f32,
    num_particles: u32,
}

// Quad vertex for instanced rendering
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

// Unit quad vertices (will be scaled by particle size in shader)
const QUAD_VERTICES: [QuadVertex; 4] = [
    QuadVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    QuadVertex { position: [ 1.0, -1.0], uv: [1.0, 1.0] },
    QuadVertex { position: [-1.0,  1.0], uv: [0.0, 0.0] },
    QuadVertex { position: [ 1.0,  1.0], uv: [1.0, 0.0] },
];

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 1, 3, 2];

// Static vertex attributes for the quad
const QUAD_VERTEX_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];

#[derive(Clone)]
struct AudioFeedback {
    min_buffer_level: u32,  // Minimum buffer level over recent history
    current_buffer_level: u32,
}

#[derive(Clone)]
struct Compute {
    particles_buf: Arc<wgpu::Buffer>,
    sim_params_buf: Arc<wgpu::Buffer>,
    particle_bind_group: Arc<wgpu::BindGroup>,
    particle_pipeline: Arc<wgpu::ComputePipeline>,

    audio_out_buf: Arc<wgpu::Buffer>,
    // Double-buffered staging for audio readback
    audio_staging_bufs: [Arc<wgpu::Buffer>; 2],
    audio_params_buf: Arc<wgpu::Buffer>,
    audio_bind_group: Arc<wgpu::BindGroup>,
    audio_pipeline: Arc<wgpu::ComputePipeline>,

    phase_update_params_buf: Arc<wgpu::Buffer>,
    phase_update_bind_group: Arc<wgpu::BindGroup>,
    phase_update_pipeline: Arc<wgpu::ComputePipeline>,
}

#[derive(Clone)]
struct Render {
    vertex_buffer: Arc<wgpu::Buffer>,
    index_buffer: Arc<wgpu::Buffer>,
    render_params_buf: Arc<wgpu::Buffer>,
    bind_group: Arc<wgpu::BindGroup>,
    pipeline: Arc<wgpu::RenderPipeline>,
}

#[derive(Clone)]
struct Model {
    compute: Compute,
    render: Render,

    // Wrapped in Arc<Mutex> for sharing with audio callback
    audio_producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
    audio_feedback_rx: Arc<Receiver<AudioFeedback>>,
    _audio_thread: Arc<JoinHandle<()>>,

    time: f32,
    frame_count: u64,

    map_x0: f32,
    map_x1: f32,
    map_y0: f32,
    map_y1: f32,

    current_x0: f32,
    current_x1: f32,
    current_y0: f32,
    current_y1: f32,

    // PID controller state for request threshold adjustment
    request_threshold: u32,
    integral_error: f32,
    latest_feedback: Option<AudioFeedback>,
    last_buffer_level: u32,

    // Audio logging state
    last_log_frame: u64,
}

struct AudioModel {
    consumer: ringbuf::HeapCons<f32>,
    feedback_tx: crossbeam_channel::Sender<AudioFeedback>,

    // Rolling buffer of recent buffer levels for min calculation
    buffer_history: VecDeque<u32>,
}

fn main() {
    nannou::app(model).update(update).render(render).exit(exit).run();
}

fn model(app: &App) -> Model {
    // Create window
    let w_id = app
        .new_window::<Model>()
        .hdr(true)
        .size(800, 800)
        .key_pressed(key_pressed)
        .build();

    let window = app.window(w_id);
    let device = window.device();
    let queue = window.queue();

    // Create ring buffer for lock-free audio transfer
    // Size: enough for ~32 update frames worth of audio
    let ring_buf_size = CHUNK_SIZE as usize * NUM_CHANNELS as usize * 32;
    let ring_buf = HeapRb::<f32>::new(ring_buf_size);
    let (audio_producer, audio_consumer) = ring_buf.split();

    // Start audio stream on a separate thread
    let audio_host = audio::Host::new();
    let (audio_feedback_tx, audio_feedback_rx) = crossbeam_channel::unbounded();

    let audio_model = AudioModel {
        consumer: audio_consumer,
        feedback_tx: audio_feedback_tx,
        buffer_history: VecDeque::with_capacity(BUFFER_HISTORY_SIZE),
    };

    let audio_thread = std::thread::spawn(move || {
        let stream = audio_host
            .new_output_stream(audio_model)
            .render(audio_fn)
            .channels(NUM_CHANNELS as usize)
            .sample_rate(SAMPLE_RATE)
            .build()
            .unwrap();
        stream.play().unwrap();

        // Keep thread alive - stream runs in background
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });

    let map_x0 = -4.0;
    let map_x1 = 4.0;
    let map_y0 = -4.0;
    let map_y1 = 4.0;

    let current_x0 = map_x0;
    let current_x1 = map_x1;
    let current_y0 = map_y0;
    let current_y1 = map_y1;

    // Initialize particles
    let particles: Vec<Particle> = (0..NUM_PARTICLES)
        .map(|i| {
            let x = map_x0 + (map_x1 - map_x0) * random_f32();
            let y = map_y0 + (map_y1 - map_y0) * random_f32();
            Particle {
                pos: [x, y],
                vel: [0.0, 0.0],
                phase: random_f32(),
                energy: 0.0,
                species: [random_f32(), random_f32()],
                alpha: [random_f32(), random_f32()],
                interaction: [0.0; 2],
            }
        })
        .collect();

    let particles_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("particles"),
        contents: bytemuck::cast_slice(&particles),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });

    let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim_params"),
        size: std::mem::size_of::<SimParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Particle compute pipeline
    let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle_shader"),
        source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
    });

    let particle_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false)
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)
        .build(&device);

    let particles_buf_size = (NUM_PARTICLES as usize * std::mem::size_of::<Particle>()) as wgpu::BufferAddress;
    let particle_bind_group = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer::<SimParams>(&sim_params_buf, 0..1)
        .build(&device, &particle_bind_group_layout);

    let particle_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle_pl"),
        bind_group_layouts: &[&particle_bind_group_layout],
        push_constant_ranges: &[],
    });

    let particle_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("particle_pipeline"),
        layout: Some(&particle_pipeline_layout),
        module: &particle_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Audio compute pipeline
    let audio_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_out"),
        size: (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Double-buffered staging for audio readback
    let audio_staging_buf_0 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_staging_0"),
        size: (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let audio_staging_buf_1 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_staging_1"),
        size: (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let audio_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_params"),
        size: std::mem::size_of::<AudioParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let audio_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("audio_shader"),
        source: wgpu::ShaderSource::Wgsl(AUDIO_SHADER.into()),
    });

    let audio_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true) // particles read-only
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // audio_out
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)
        .build(&device);

    let audio_out_size = (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;
    let audio_bind_group = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer_bytes(&audio_out_buf, 0, std::num::NonZeroU64::new(audio_out_size))
        .buffer::<AudioParams>(&audio_params_buf, 0..1)
        .build(&device, &audio_bind_group_layout);

    let audio_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("audio_pl"),
        bind_group_layouts: &[&audio_bind_group_layout],
        push_constant_ranges: &[],
    });

    let audio_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("audio_pipeline"),
        layout: Some(&audio_pipeline_layout),
        module: &audio_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let phase_update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("phase_update_shader"),
        source: wgpu::ShaderSource::Wgsl(PHASE_UPDATE_SHADER.into()),
    });

    // Phase update params buffer
    let phase_update_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("phase_update_params"),
        size: std::mem::size_of::<PhaseUpdateParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let phase_update_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // particles read-write
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)
        .build(&device);

    let phase_update_bind_group = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer::<PhaseUpdateParams>(&phase_update_params_buf, 0..1)
        .build(&device, &phase_update_bind_group_layout);

    let phase_update_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("phase_update_pl"),
        bind_group_layouts: &[&phase_update_bind_group_layout],
        push_constant_ranges: &[],
    });

    let phase_update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("phase_update_pipeline"),
        layout: Some(&phase_update_pipeline_layout),
        module: &phase_update_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Write initial phase update params
    queue.write_buffer(
        &phase_update_params_buf,
        0,
        bytemuck::bytes_of(&PhaseUpdateParams {
            sample_rate: SAMPLE_RATE as f32,
            num_particles: NUM_PARTICLES,
            chunk_size: CHUNK_SIZE,
            _pad: 0,
        }),
    );

    // Write initial audio params
    queue.write_buffer(
        &audio_params_buf,
        0,
        bytemuck::bytes_of(&AudioParams {
            sample_rate: SAMPLE_RATE as f32,
            num_particles: NUM_PARTICLES,
            chunk_size: CHUNK_SIZE,
            volume: 0.8,
            current_x: [current_x0, current_x1],
            current_y: [current_y0, current_y1],
        }),
    );

    let particles_buf = Arc::new(particles_buf);
    let compute = Compute {
        particles_buf: particles_buf.clone(),
        sim_params_buf: Arc::new(sim_params_buf),
        particle_bind_group: Arc::new(particle_bind_group),
        particle_pipeline: Arc::new(particle_pipeline),
        audio_out_buf: Arc::new(audio_out_buf),
        audio_staging_bufs: [Arc::new(audio_staging_buf_0), Arc::new(audio_staging_buf_1)],
        audio_params_buf: Arc::new(audio_params_buf),
        audio_bind_group: Arc::new(audio_bind_group),
        audio_pipeline: Arc::new(audio_pipeline),
        phase_update_params_buf: Arc::new(phase_update_params_buf),
        phase_update_bind_group: Arc::new(phase_update_bind_group),
        phase_update_pipeline: Arc::new(phase_update_pipeline),
    };

    // Create render pipeline for GPU-based particle rendering with instancing
    let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render_shader"),
        source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
    });

    // Vertex buffer for the quad
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("quad_vertices"),
        contents: bytemuck::cast_slice(&QUAD_VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Index buffer for the quad
    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("quad_indices"),
        contents: bytemuck::cast_slice(&QUAD_INDICES),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Render params uniform buffer
    let render_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_params"),
        size: std::mem::size_of::<RenderParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Render bind group layout: particles storage + params uniform
    let render_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::VERTEX, false, true) // particles read-only
        .uniform_buffer(wgpu::ShaderStages::VERTEX, false)       // render params
        .build(&device);

    let render_bind_group = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&*particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer::<RenderParams>(&render_params_buf, 0..1)
        .build(&device, &render_bind_group_layout);

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipeline_layout"),
        bind_group_layouts: &[&render_bind_group_layout],
        push_constant_ranges: &[],
    });

    let format = Frame::TEXTURE_FORMAT;
    let msaa_samples = window.msaa_samples();

    let render_pipeline = wgpu::RenderPipelineBuilder::from_layout(&render_pipeline_layout, &render_shader)
        .vertex_entry_point("vs_main")
        .fragment_shader(&render_shader)
        .fragment_entry_point("fs_main")
        .color_format(format)
        .color_blend(wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        })
        .alpha_blend(wgpu::BlendComponent::OVER)
        .add_vertex_buffer::<QuadVertex>(&QUAD_VERTEX_ATTRS)
        .sample_count(msaa_samples)
        .primitive_topology(wgpu::PrimitiveTopology::TriangleList)
        .build(&*device);

    let render = Render {
        vertex_buffer: Arc::new(vertex_buffer),
        index_buffer: Arc::new(index_buffer),
        render_params_buf: Arc::new(render_params_buf),
        bind_group: Arc::new(render_bind_group),
        pipeline: Arc::new(render_pipeline),
    };

    Model {
        compute,
        render,
        audio_producer: Arc::new(Mutex::new(audio_producer)),
        audio_feedback_rx: Arc::new(audio_feedback_rx),
        _audio_thread: Arc::new(audio_thread),
        time: 0.0,
        frame_count: 0,
        map_x0,
        map_x1,
        map_y0,
        map_y1,
        current_x0,
        current_x1,
        current_y0,
        current_y1,
        request_threshold: INITIAL_REQUEST_THRESHOLD,
        integral_error: 0.0,
        latest_feedback: None,
        last_buffer_level: 0,  // Start at 0 to fill buffer initially
        last_log_frame: 0,
    }
}

fn exit(_app: &App, _model: Model) {
    // Audio thread will be cleaned up when Model is dropped
}

fn key_pressed(_app: &App, model: &mut Model, key: KeyCode) {
    let view_width = model.current_x1 - model.current_x0;
    let view_height = model.current_y1 - model.current_y0;
    let pan_x = view_width * PAN_SPEED;
    let pan_y = view_height * PAN_SPEED;

    match key {
        // Pan controls (WASD)
        KeyCode::KeyW => {
            model.current_y0 += pan_y;
            model.current_y1 += pan_y;
        }
        KeyCode::KeyS => {
            model.current_y0 -= pan_y;
            model.current_y1 -= pan_y;
        }
        KeyCode::KeyA => {
            model.current_x0 -= pan_x;
            model.current_x1 -= pan_x;
        }
        KeyCode::KeyD => {
            model.current_x0 += pan_x;
            model.current_x1 += pan_x;
        }
        // Zoom controls (E to zoom in, F to zoom out)
        KeyCode::KeyE => {
            let center_x = (model.current_x0 + model.current_x1) / 2.0;
            let center_y = (model.current_y0 + model.current_y1) / 2.0;
            let new_width = view_width * (1.0 - ZOOM_SPEED);
            let new_height = view_height * (1.0 - ZOOM_SPEED);
            model.current_x0 = center_x - new_width / 2.0;
            model.current_x1 = center_x + new_width / 2.0;
            model.current_y0 = center_y - new_height / 2.0;
            model.current_y1 = center_y + new_height / 2.0;
        }
        KeyCode::KeyF => {
            let center_x = (model.current_x0 + model.current_x1) / 2.0;
            let center_y = (model.current_y0 + model.current_y1) / 2.0;
            let new_width = view_width * (1.0 + ZOOM_SPEED);
            let new_height = view_height * (1.0 + ZOOM_SPEED);
            model.current_x0 = center_x - new_width / 2.0;
            model.current_x1 = center_x + new_width / 2.0;
            model.current_y0 = center_y - new_height / 2.0;
            model.current_y1 = center_y + new_height / 2.0;
        }
        // Reset view to full map
        KeyCode::KeyR => {
            model.current_x0 = model.map_x0;
            model.current_x1 = model.map_x1;
            model.current_y0 = model.map_y0;
            model.current_y1 = model.map_y1;
        }
        _ => {}
    }
}

fn update(_app: &App, model: &mut Model) {
    // Process audio feedback - use the latest min buffer level from audio thread
    let mut latest_feedback = None;
    let mut had_underrun = false;
    while let Ok(feedback) = model.audio_feedback_rx.try_recv() {
        // Detect underrun: buffer dropped to 0
        if feedback.current_buffer_level == 0 {
            had_underrun = true;
        }
        latest_feedback = Some(feedback);
    }

    // Log underrun immediately when detected
    if had_underrun {
        eprintln!("[AUDIO] Buffer underrun detected!");
    }

    // PID controller adjusts request_threshold based on buffer feedback
    if let Some(feedback) = &latest_feedback {
        let min_buffer = feedback.min_buffer_level as f32;
        let target = (CHUNK_FLOATS * 2) as f32;

        let error = target - min_buffer;
        let kp = 0.01;
        let ki = 0.002;

        model.integral_error = (model.integral_error + error).clamp(-5000.0, 5000.0);
        let adjustment = kp * error + ki * model.integral_error;

        let new_threshold = ((model.request_threshold as f32) + adjustment) as u32;
        model.request_threshold = new_threshold.clamp(MIN_REQUEST_THRESHOLD, MAX_REQUEST_THRESHOLD);
    }

    let dt = 1.0 / 60.0;
    model.time += dt;
    model.frame_count += 1;

    // Periodic logging of buffer status
    if model.frame_count - model.last_log_frame >= AUDIO_LOG_INTERVAL_FRAMES {
        if let Some(feedback) = &latest_feedback {
            eprintln!(
                "[AUDIO] Buffer: current={}, min={}, threshold={}",
                feedback.current_buffer_level,
                feedback.min_buffer_level,
                model.request_threshold
            );
        }
        model.last_log_frame = model.frame_count;
    }

    // Store latest feedback and update last_buffer_level
    if let Some(feedback) = latest_feedback {
        model.last_buffer_level = feedback.current_buffer_level;
        model.latest_feedback = Some(feedback);
    }
    // If no feedback received this frame, last_buffer_level retains its previous value
}

fn audio_fn(audio: &mut AudioModel, buffer: &mut Buffer) {
    // Fill the output buffer from ring buffer (lock-free)
    // Explicitly write to each channel as in the nannou example
    for frame in buffer.frames_mut() {
        let left = audio.consumer.try_pop().unwrap_or(0.0);
        let right = audio.consumer.try_pop().unwrap_or(0.0);
        frame[0] = left;
        frame[1] = right;
    }

    let buffer_len = audio.consumer.occupied_len() as u32;

    // Push new value, pop oldest if at capacity
    audio.buffer_history.push_back(buffer_len);
    if audio.buffer_history.len() > BUFFER_HISTORY_SIZE {
        audio.buffer_history.pop_front();
    }

    // Calculate min over the rolling window and send feedback
    let min_buffer = audio.buffer_history.iter().copied().min().unwrap_or(0);

    // Send feedback to main thread (PID controller will adjust threshold there)
    audio.feedback_tx.send(AudioFeedback {
        min_buffer_level: min_buffer,
        current_buffer_level: buffer_len,
    }).ok();
}

fn render(app: &RenderApp, model: &Model, frame: Frame) {
    let device = frame.device();

    // Update render params with current viewport
    let render_params = RenderParams {
        map_x: [model.map_x0, model.map_x1],
        map_y: [model.map_y0, model.map_y1],
        current_x: [model.current_x0, model.current_x1],
        current_y: [model.current_y0, model.current_y1],
        particle_size: PARTICLE_SIZE,
        num_particles: NUM_PARTICLES,
    };
    let render_params_bytes = bytemuck::bytes_of(&render_params);
    let new_render_params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("render-params-transfer"),
        contents: render_params_bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    // Update sim params
    let sim_params = SimParams {
        dt: 1.0 / 60.0,
        time: model.time,
        num_particles: NUM_PARTICLES,
        friction: 0.1,
        mass: 1.0,
        map_x0: model.map_x0,
        map_x1: model.map_x1,
        map_y0: model.map_y0,
        map_y1: model.map_y1,
        _pad1: 0.0,
        _pad2: 0.0,
        _pad3: 0.0,
    };
    let sim_params_bytes = bytemuck::bytes_of(&sim_params);
    let new_sim_params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("sim-params-transfer"),
        contents: sim_params_bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    // Update audio params with current viewport
    let audio_params = AudioParams {
        sample_rate: SAMPLE_RATE as f32,
        num_particles: NUM_PARTICLES,
        chunk_size: CHUNK_SIZE,
        volume: 0.8,
        current_x: [model.current_x0, model.current_x1],
        current_y: [model.current_y0, model.current_y1],
    };
    let audio_params_bytes = bytemuck::bytes_of(&audio_params);
    let new_audio_params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("audio-params-transfer"),
        contents: audio_params_bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    let audio_buf_size = (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64;

    let mut encoder = frame.command_encoder();

    // Copy new params
    encoder.copy_buffer_to_buffer(
        &new_render_params_buf,
        0,
        &*model.render.render_params_buf,
        0,
        std::mem::size_of::<RenderParams>() as u64,
    );
    encoder.copy_buffer_to_buffer(
        &new_sim_params_buf,
        0,
        &*model.compute.sim_params_buf,
        0,
        std::mem::size_of::<SimParams>() as u64,
    );
    encoder.copy_buffer_to_buffer(
        &new_audio_params_buf,
        0,
        &*model.compute.audio_params_buf,
        0,
        std::mem::size_of::<AudioParams>() as u64,
    );

    // Particle simulation pass
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&*model.compute.particle_pipeline);
        pass.set_bind_group(0, &*model.compute.particle_bind_group, &[]);
        pass.dispatch_workgroups((NUM_PARTICLES + 63) / 64, 1, 1);
    }

    let need_audio = model.last_buffer_level < model.request_threshold;

    if need_audio {
        // Audio synthesis pass (fixed chunk size)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("audio_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&*model.compute.audio_pipeline);
            pass.set_bind_group(0, &*model.compute.audio_bind_group, &[]);
            pass.dispatch_workgroups((CHUNK_SIZE + 63) / 64, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&*model.compute.phase_update_pipeline);
            pass.set_bind_group(0, &*model.compute.phase_update_bind_group, &[]);
            pass.dispatch_workgroups((NUM_PARTICLES + 63) / 64, 1, 1);
        }

        // TODO ensure buffer gets read on next frame even if audio not needed?
        // Double-buffer audio readback: write to one buffer, read from the other
        let write_idx = (model.audio_frame_count % 2) as usize;
        let read_idx = ((model.audio_frame_count + 1) % 2) as usize;

        // Copy audio to the write staging buffer
        encoder.copy_buffer_to_buffer(
            &*model.compute.audio_out_buf,
            0,
            &*model.compute.audio_staging_bufs[write_idx],
            0,
            audio_buf_size,
        );

        // After frame 1, we can start reading from the buffer that was written in the previous frame
        // The map_async callback fires when the buffer is ready - do all work inside the callback
        if model.frame_count >= 2 {
            let read_buf = model.compute.audio_staging_bufs[read_idx].clone();
            let read_buf_for_callback = read_buf.clone();
            let audio_producer = model.audio_producer.clone();
            let audio_buf_size = audio_buf_size as usize;

            read_buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    let data = read_buf_for_callback.slice(..).get_mapped_range();
                    let floats = bytemuck::cast_slice::<u8, f32>(&data[..audio_buf_size]);
                    if let Ok(mut producer) = audio_producer.lock() {
                        producer.push_slice(floats);
                    }
                    drop(data);
                    read_buf_for_callback.unmap();
                }
            });
        }
    }

    // GPU-based instanced rendering
    {
        let mut render_pass = wgpu::RenderPassBuilder::new()
            .color_attachment(frame.resolve_target_view().unwrap(), |color| color)
            .begin(&mut encoder);

        render_pass.set_pipeline(&*model.render.pipeline);
        render_pass.set_bind_group(0, &*model.render.bind_group, &[]);
        render_pass.set_vertex_buffer(0, model.render.vertex_buffer.slice(..));
        render_pass.set_index_buffer(model.render.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        // Draw all particles with instancing
        render_pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..NUM_PARTICLES);
    }
}