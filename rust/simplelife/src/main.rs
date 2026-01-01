use nannou::prelude::*;
use nannou::wgpu::BufferInitDescriptor;
use nannou_audio as audio;
use nannou_audio::Buffer;
use std::thread::JoinHandle;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use crossbeam_channel::{self, Receiver};
use ringbuf::{traits::{Consumer, Producer, Split, Observer}, HeapRb};

const NUM_PARTICLES: u32 = 64;
const NUM_CHANNELS: u32 = 2;
const SAMPLE_RATE: f32 = 44100.0;
const CHUNK_SIZE: u32 = 2048;
const CHUNK_FLOATS: u32 = CHUNK_SIZE * NUM_CHANNELS;
const INITIAL_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 2;
const MIN_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS;
const MAX_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 16;
const BUFFER_HISTORY_SIZE: usize = 64;
const PARTICLE_SIZE: f32 = 12.0;

// Shader sources (loaded at compile time)
const PARTICLE_SHADER: &str = include_str!("shaders/particle.wgsl");
const AUDIO_SHADER: &str = include_str!("shaders/audio.wgsl");
const RENDER_SHADER: &str = include_str!("shaders/render.wgsl");

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
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    volume: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    screen_size: [f32; 2],
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
    audio_staging_buf: Arc<wgpu::Buffer>,
    audio_params_buf: Arc<wgpu::Buffer>,
    audio_bind_group: Arc<wgpu::BindGroup>,
    audio_pipeline: Arc<wgpu::ComputePipeline>,
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
    window: Entity,
    compute: Compute,
    render: Render,

    // Wrapped in Arc<Mutex> for sharing with async readback task
    audio_producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
    audio_feedback_rx: Arc<Receiver<AudioFeedback>>,
    _audio_thread: Arc<JoinHandle<()>>,

    time: f32,
    read_task: Arc<Mutex<Option<Task<()>>>>,

    // PID controller state for request threshold adjustment
    request_threshold: u32,
    integral_error: f32,
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
        .primary()
        .size(800, 800)
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
            .build()
            .unwrap();
        stream.play().unwrap();

        // Keep thread alive - stream runs in background
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });

    // Initialize particles
    let particles: Vec<Particle> = (0..NUM_PARTICLES)
        .map(|i| {
            let angle = (i as f32 / NUM_PARTICLES as f32) * TAU;
            let r = 0.3 + random_f32() * 0.2;
            Particle {
                pos: [angle.cos() * r, angle.sin() * r],
                vel: [0.0, 0.0],
                phase: random_f32(),
                energy: 0.0,
                species: [0.0; 2],
                alpha: [0.0; 2],
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

    let audio_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_staging"),
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

    // Write initial audio params
    queue.write_buffer(
        &audio_params_buf,
        0,
        bytemuck::bytes_of(&AudioParams {
            sample_rate: SAMPLE_RATE,
            num_particles: NUM_PARTICLES,
            chunk_size: CHUNK_SIZE,
            volume: 0.8,
        }),
    );

    let particles_buf = Arc::new(particles_buf);
    let compute = Compute {
        particles_buf: particles_buf.clone(),
        sim_params_buf: Arc::new(sim_params_buf),
        particle_bind_group: Arc::new(particle_bind_group),
        particle_pipeline: Arc::new(particle_pipeline),
        audio_out_buf: Arc::new(audio_out_buf),
        audio_staging_buf: Arc::new(audio_staging_buf),
        audio_params_buf: Arc::new(audio_params_buf),
        audio_bind_group: Arc::new(audio_bind_group),
        audio_pipeline: Arc::new(audio_pipeline),
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
        window: w_id,
        compute,
        render,
        audio_producer: Arc::new(Mutex::new(audio_producer)),
        audio_feedback_rx: Arc::new(audio_feedback_rx),
        _audio_thread: Arc::new(audio_thread),
        time: 0.0,
        read_task: Arc::new(Mutex::new(None)),
        request_threshold: INITIAL_REQUEST_THRESHOLD,
        integral_error: 0.0,
    }
}

fn exit(_app: &App, _model: Model) {
    // Audio thread will be cleaned up when Model is dropped
}

fn update(app: &App, model: &mut Model) {
    let window = app.window(model.window);
    let device = window.device();
    let queue = window.queue();

    // Update render params with current window size
    let win_size = window.size_pixels();
    let render_params = RenderParams {
        screen_size: [win_size.x as f32, win_size.y as f32],
        particle_size: PARTICLE_SIZE,
        num_particles: NUM_PARTICLES,
    };
    queue.write_buffer(
        &*model.render.render_params_buf,
        0,
        bytemuck::bytes_of(&render_params),
    );

    // Process audio feedback - use the latest min buffer level from audio thread
    let mut latest_feedback = None;
    while let Ok(feedback) = model.audio_feedback_rx.try_recv() {
        latest_feedback = Some(feedback);
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

    // Check if previous read task completed
    {
        let mut read_task = model.read_task.lock().unwrap();
        if let Some(task) = read_task.as_mut() {
            if block_on(future::poll_once(task)).is_some() {
                *read_task = None;
            }
        }
    }

    let dt = 1.0 / 60.0;
    model.time += dt;

    // Update sim params
    let sim_params = SimParams {
        dt,
        time: model.time,
        num_particles: NUM_PARTICLES,
        friction: 0.1,
        mass: 1.0,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };
    let sim_params_bytes = bytemuck::bytes_of(&sim_params);
    let new_sim_params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("sim-params-transfer"),
        contents: sim_params_bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    // Create read buffer for audio (fixed chunk size)
    let audio_buf_size = (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64;
    let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-audio"),
        size: audio_buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute_encoder"),
    });

    // Copy new sim params
    encoder.copy_buffer_to_buffer(
        &new_sim_params_buf,
        0,
        &*model.compute.sim_params_buf,
        0,
        std::mem::size_of::<SimParams>() as u64,
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

    // Copy audio to read buffer
    encoder.copy_buffer_to_buffer(
        &*model.compute.audio_out_buf,
        0,
        &read_buffer,
        0,
        audio_buf_size,
    );

    queue.submit(Some(encoder.finish()));

    // Only spawn new read task if previous one is done and buffer needs data
    let should_request_audio = latest_feedback
        .map(|f| f.current_buffer_level < model.request_threshold)
        .unwrap_or(true);

    let task_is_none = model.read_task.lock().unwrap().is_none();
    if task_is_none && should_request_audio {
        let audio_producer_arc = model.audio_producer.clone();

        let future = async move {
            // Read audio buffer
            let audio_slice = read_buffer.slice(..);
            let (audio_tx, audio_rx) = futures::channel::oneshot::channel();
            audio_slice.map_async(wgpu::MapMode::Read, |res| {
                audio_tx.send(res).expect("Channel closed");
            });

            // Wait for audio and copy to ring buffer
            if audio_rx.await.is_ok() {
                let bytes = &audio_slice.get_mapped_range()[..];
                let floats = bytemuck::cast_slice::<u8, f32>(bytes);
                if let Ok(mut producer) = audio_producer_arc.lock() {
                    producer.push_slice(floats);
                }
            }
        };

        let thread_pool = AsyncComputeTaskPool::get();
        *model.read_task.lock().unwrap() = Some(thread_pool.spawn(future));
    }
}

fn audio_fn(audio: &mut AudioModel, buffer: &mut Buffer) {
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

    // Fill the output buffer from ring buffer (lock-free)
    for frame in buffer.frames_mut() {
        for sample in frame.iter_mut() {
            *sample = audio.consumer.try_pop().unwrap_or(0.0);
        }
    }
}

fn render(_app: &RenderApp, model: &Model, frame: Frame) {
    // GPU-based instanced rendering - no CPU roundtrip for particle positions
    let mut encoder = frame.command_encoder();

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
