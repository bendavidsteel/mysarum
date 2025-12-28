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
const TARGET_FRAME_RATE: f32 = 60.0;
// Initial chunk size: samples needed per frame with some buffer headroom
const INITIAL_CHUNK_SIZE: u32 = ((SAMPLE_RATE / TARGET_FRAME_RATE) as u32) * 2;
// Bounds for chunk size adjustment
const MIN_CHUNK_SIZE: u32 = (SAMPLE_RATE / TARGET_FRAME_RATE) as u32;
const MAX_CHUNK_SIZE: u32 = INITIAL_CHUNK_SIZE * 8;
// Safety margin: target minimum buffer level (in samples) - enough for ~2 audio callbacks
const TARGET_MIN_BUFFER: u32 = 4096;
// Number of recent buffer levels to track for min calculation
const BUFFER_HISTORY_SIZE: usize = 64;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    phase: f32,
    energy: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    dt: f32,
    time: f32,
    num_particles: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    volume: f32,
}

struct AudioFeedback {
    min_buffer_level: u32,  // Minimum buffer level over recent history
    current_buffer_level: u32,
}

struct Compute {
    particles_buf: wgpu::Buffer,
    sim_params_buf: wgpu::Buffer,
    particle_bind_group: wgpu::BindGroup,
    particle_pipeline: wgpu::ComputePipeline,

    audio_out_buf: wgpu::Buffer,
    audio_staging_buf: wgpu::Buffer,
    audio_params_buf: wgpu::Buffer,
    audio_bind_group: wgpu::BindGroup,
    audio_pipeline: wgpu::ComputePipeline,
}

struct Model {
    window: Entity,
    compute: Compute,

    // Wrapped in Arc<Mutex> for sharing with async readback task
    audio_producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
    audio_feedback_rx: Receiver<AudioFeedback>,
    _audio_thread: JoinHandle<()>,

    time: f32,
    read_task: Option<Task<()>>,
    chunk_size: u32,

    // PI controller state for chunk size adjustment
    integral_error: f32,

    // Shared particle data for rendering (updated by async readback task)
    particles: Arc<Mutex<Vec<Particle>>>,
}

struct AudioModel {
    consumer: ringbuf::HeapCons<f32>,
    feedback_tx: crossbeam_channel::Sender<AudioFeedback>,

    // Rolling buffer of recent buffer levels for min calculation
    buffer_history: VecDeque<u32>,
}

fn main() {
    nannou::app(model).update(update).exit(exit).run();
}

fn model(app: &App) -> Model {
    // Create window with view callback
    let w_id = app
        .new_window()
        .primary()
        .size(800, 800)
        .view(view)
        .build();

    let window = app.window(w_id);
    let device = window.device();
    let queue = window.queue();

    // Create ring buffer for lock-free audio transfer
    // Size: enough for ~32 update frames worth of audio
    let ring_buf_size = INITIAL_CHUNK_SIZE as usize * NUM_CHANNELS as usize * 32;
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
                _pad: [0.0; 2],
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
        size: (INITIAL_CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let audio_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_staging"),
        size: (INITIAL_CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64,
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

    let audio_out_size = (INITIAL_CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;
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
            chunk_size: INITIAL_CHUNK_SIZE,
            volume: 0.8,
        }),
    );

    let compute = Compute {
        particles_buf,
        sim_params_buf,
        particle_bind_group,
        particle_pipeline,
        audio_out_buf,
        audio_staging_buf,
        audio_params_buf,
        audio_bind_group,
        audio_pipeline,
    };

    Model {
        window: w_id,
        compute,
        audio_producer: Arc::new(Mutex::new(audio_producer)),
        audio_feedback_rx,
        _audio_thread: audio_thread,
        time: 0.0,
        read_task: None,
        chunk_size: INITIAL_CHUNK_SIZE,
        integral_error: 0.0,
        particles: Arc::new(Mutex::new(particles)),
    }
}

fn exit(_app: &App, _model: Model) {
    // Audio thread will be cleaned up when Model is dropped
}

fn update(app: &App, model: &mut Model) {
    let window = app.window(model.window);
    let device = window.device();
    let queue = window.queue();

    // Process audio feedback - use the latest min buffer level from audio thread
    let mut latest_feedback = None;
    let mut feedback_count = 0u32;
    while let Ok(feedback) = model.audio_feedback_rx.try_recv() {
        latest_feedback = Some(feedback);
        feedback_count += 1;
    }

    let mut chunk_size_changed = false;
    if let Some(feedback) = latest_feedback {
        let min_buffer = feedback.min_buffer_level as f32;
        let current_buffer = feedback.current_buffer_level;
        let target = TARGET_MIN_BUFFER as f32;

        // Log audio state (once per batch of feedback)
        eprintln!(
            "Audio: buffer={}, min_over_window={}, feedbacks={}",
            current_buffer, feedback.min_buffer_level, feedback_count
        );

        // PI controller for chunk size
        // Error: positive means buffer too low (need more samples), negative means buffer too high
        let error = target - min_buffer;

        // PI gains (tuned for stability - keep low to avoid oscillation)
        let kp = 0.01;  // Proportional gain
        let ki = 0.002; // Integral gain

        // Update integral with anti-windup clamping
        model.integral_error = (model.integral_error + error).clamp(-5000.0, 5000.0);

        // Calculate adjustment
        let adjustment = kp * error + ki * model.integral_error;

        let new_chunk_size = ((model.chunk_size as f32) + adjustment) as u32;
        let new_chunk_size = new_chunk_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);

        if new_chunk_size != model.chunk_size {
            eprintln!(
                "Chunk size: {} -> {} (min_buf={}, target={}, err={:.0}, int={:.0})",
                model.chunk_size, new_chunk_size, min_buffer as u32, TARGET_MIN_BUFFER, error, model.integral_error
            );
            model.chunk_size = new_chunk_size;
            chunk_size_changed = true;
        }
    }

    // Recreate audio buffers if chunk size changed
    if chunk_size_changed {
        let audio_buf_size = (model.chunk_size * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64;

        model.compute.audio_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("audio_out"),
            size: audio_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        model.compute.audio_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("audio_staging"),
            size: audio_buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Update audio params with new chunk size
        queue.write_buffer(
            &model.compute.audio_params_buf,
            0,
            bytemuck::bytes_of(&AudioParams {
                sample_rate: SAMPLE_RATE,
                num_particles: NUM_PARTICLES,
                chunk_size: model.chunk_size,
                volume: 0.8,
            }),
        );

        // Recreate bind group with new buffer
        let particles_buf_size = (NUM_PARTICLES as usize * std::mem::size_of::<Particle>()) as wgpu::BufferAddress;
        let audio_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
            .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true)
            .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false)
            .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)
            .build(&device);

        model.compute.audio_bind_group = wgpu::BindGroupBuilder::new()
            .buffer_bytes(&model.compute.particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
            .buffer_bytes(&model.compute.audio_out_buf, 0, std::num::NonZeroU64::new(audio_buf_size))
            .buffer::<AudioParams>(&model.compute.audio_params_buf, 0..1)
            .build(&device, &audio_bind_group_layout);
    }

    // Check if previous read task completed
    if let Some(task) = &mut model.read_task {
        if block_on(future::poll_once(task)).is_some() {
            model.read_task = None;
        }
    }

    let dt = 1.0 / 60.0;
    model.time += dt;

    // Update sim params
    let sim_params = SimParams {
        dt,
        time: model.time,
        num_particles: NUM_PARTICLES,
        _pad: 0,
    };
    let sim_params_bytes = bytemuck::bytes_of(&sim_params);
    let new_sim_params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("sim-params-transfer"),
        contents: sim_params_bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    // Create read buffer for audio
    let audio_buf_size = (model.chunk_size * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64;
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
        &model.compute.sim_params_buf,
        0,
        std::mem::size_of::<SimParams>() as u64,
    );

    // Particle simulation pass
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&model.compute.particle_pipeline);
        pass.set_bind_group(0, &model.compute.particle_bind_group, &[]);
        pass.dispatch_workgroups((NUM_PARTICLES + 63) / 64, 1, 1);
    }

    // Audio synthesis pass
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("audio_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&model.compute.audio_pipeline);
        pass.set_bind_group(0, &model.compute.audio_bind_group, &[]);
        pass.dispatch_workgroups((model.chunk_size + 63) / 64, 1, 1);
    }

    // Copy audio to read buffer
    encoder.copy_buffer_to_buffer(
        &model.compute.audio_out_buf,
        0,
        &read_buffer,
        0,
        audio_buf_size,
    );

    // Create staging buffer for particle readback
    let particles_buf_size = (NUM_PARTICLES as usize * std::mem::size_of::<Particle>()) as u64;
    let particle_read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle-read"),
        size: particles_buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &model.compute.particles_buf,
        0,
        &particle_read_buffer,
        0,
        particles_buf_size,
    );

    queue.submit(Some(encoder.finish()));

    // Only spawn new read task if previous one is done
    if model.read_task.is_none() {
        let audio_producer_arc = model.audio_producer.clone();
        let particles_arc = model.particles.clone();

        let future = async move {
            // Read audio buffer
            let audio_slice = read_buffer.slice(..);
            let (audio_tx, audio_rx) = futures::channel::oneshot::channel();
            audio_slice.map_async(wgpu::MapMode::Read, |res| {
                audio_tx.send(res).expect("Channel closed");
            });

            // Read particle buffer
            let particle_slice = particle_read_buffer.slice(..);
            let (particle_tx, particle_rx) = futures::channel::oneshot::channel();
            particle_slice.map_async(wgpu::MapMode::Read, |res| {
                particle_tx.send(res).expect("Channel closed");
            });

            // Wait for audio and copy to ring buffer
            if let Ok(_) = audio_rx.await {
                let bytes = &audio_slice.get_mapped_range()[..];
                let floats = bytemuck::cast_slice::<u8, f32>(bytes);
                if let Ok(mut producer) = audio_producer_arc.lock() {
                    let pushed = producer.push_slice(floats);
                    eprintln!("GPU: produced {} samples (pushed {})", floats.len(), pushed);
                }
            }

            // Wait for particles and copy to shared vec
            if let Ok(_) = particle_rx.await {
                let bytes = &particle_slice.get_mapped_range()[..];
                let particles: &[Particle] = bytemuck::cast_slice(bytes);
                if let Ok(mut shared_particles) = particles_arc.lock() {
                    shared_particles.copy_from_slice(particles);
                }
            }
        };

        let thread_pool = AsyncComputeTaskPool::get();
        model.read_task = Some(thread_pool.spawn(future));
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

fn view(app: &App, model: &Model) {
    let draw = app.draw();
    draw.background().color(BLACK);

    let win = app.window_rect();
    let scale = win.w().min(win.h()) * 0.9;

    // Draw particles from actual GPU-computed positions
    if let Ok(particles) = model.particles.lock() {
        for (i, p) in particles.iter().enumerate() {
            let x = p.pos[0] * scale;
            let y = p.pos[1] * scale;
            let hue = i as f32 / NUM_PARTICLES as f32;
            let energy_brightness = 0.4 + p.energy.min(1.0) * 0.6;
            draw.ellipse()
                .x_y(x, y)
                .w_h(6.0, 6.0)
                .hsla(hue, 0.7, energy_brightness, 0.9);
        }
    }
}

const PARTICLE_SHADER: &str = r#"
struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    phase: f32,
    energy: f32,
    _pad: vec2<f32>,
}

struct SimParams {
    dt: f32,
    time: f32,
    num_particles: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// Simple Lenia-inspired kernel
fn kernel(r: f32, mu: f32, sigma: f32) -> f32 {
    return exp(-pow((r - mu) / sigma, 2.0) / 2.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.num_particles {
        return;
    }

    var p = particles[id.x];

    // Compute field from other particles
    var field = 0.0;
    var grad = vec2<f32>(0.0);

    for (var i = 0u; i < params.num_particles; i++) {
        if i == id.x {
            continue;
        }
        let other = particles[i];
        let diff = other.pos - p.pos;
        let dist = length(diff);

        if dist > 0.001 && dist < 0.5 {
            let k = kernel(dist, 0.15, 0.05);
            field += k;
            grad += normalize(diff) * k;
        }
    }

    // Growth function (Lenia-style)
    let growth = kernel(field, 0.3, 0.1) * 2.0 - 1.0;

    // Update velocity based on growth gradient
    p.vel += grad * growth * params.dt * 0.5;
    p.vel *= 0.98; // Damping

    // Update position
    p.pos += p.vel * params.dt;

    // Boundary wrap
    p.pos = fract(p.pos + 1.0) - 0.5;

    // Update phase (for audio synthesis)
    p.phase = fract(p.phase + params.dt * (1.0 + field * 2.0));

    // Store energy for audio
    p.energy = length(p.vel) + abs(growth) * 0.5;

    particles[id.x] = p;
}
"#;

const AUDIO_SHADER: &str = r#"
struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    phase: f32,
    energy: f32,
    _pad: vec2<f32>,
}

struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    volume: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> audio_out: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: AudioParams;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.chunk_size {
        return;
    }

    let sample_idx = id.x;
    let t = f32(sample_idx) / params.sample_rate;

    var left = 0.0;
    var right = 0.0;
    var total_energy = 0.0;

    // Sum contributions from all particles
    for (var i = 0u; i < params.num_particles; i++) {
        let p = particles[i];

        // Base frequency derived from particle position
        let base_freq = 100.0 + (p.pos.x + 0.5) * 400.0;

        // Phase accumulation with particle's stored phase
        let phase = TAU * base_freq * t + p.phase * TAU;

        // Simple sine with harmonics, modulated by energy
        let osc = sin(phase) + 0.5 * sin(phase * 2.0) + 0.25 * sin(phase * 3.0);

        // Amplitude from particle energy (boosted for audibility)
        let amp = p.energy * 0.5 + 0.002;  // base amplitude even with low energy

        // Stereo pan based on x position
        let pan = p.pos.x + 0.5; // 0 to 1

        left += osc * amp * (1.0 - pan);
        right += osc * amp * pan;
        total_energy += p.energy;
    }

    // Normalize by particle count and apply volume
    let norm = 1.0 / f32(params.num_particles);
    audio_out[sample_idx] = vec2<f32>(left, right) * norm * params.volume;
}
"#;
