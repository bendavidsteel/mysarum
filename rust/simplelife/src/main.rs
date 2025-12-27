use nannou::prelude::*;
use nannou::wgpu::BufferInitDescriptor;
use nannou_audio as audio;
use nannou_audio::Buffer;
use std::thread::JoinHandle;
use std::sync::Mutex;
use crossbeam_channel::{self, Receiver};
use ringbuf::{traits::{Consumer, Producer, Split, Observer}, HeapRb};

const NUM_PARTICLES: u32 = 512;
const NUM_CHANNELS: u32 = 2;
const SAMPLE_RATE: f32 = 44100.0;
const TARGET_FRAME_RATE: f32 = 60.0;
// Initial chunk size: samples needed per frame with some buffer headroom
const INITIAL_CHUNK_SIZE: u32 = ((SAMPLE_RATE / TARGET_FRAME_RATE) as u32) * 2;
// Bounds for chunk size adjustment
const MIN_CHUNK_SIZE: u32 = (SAMPLE_RATE / TARGET_FRAME_RATE) as u32;
const MAX_CHUNK_SIZE: u32 = INITIAL_CHUNK_SIZE * 8;

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

enum AudioCommand {
    Exit,
}

struct AudioFeedback {
    ideal_chunk_size: u32,
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

    // Wrapped in Mutex to satisfy Sync requirement (only accessed from main thread)
    audio_producer: Mutex<ringbuf::HeapProd<f32>>,
    audio_feedback_rx: Receiver<AudioFeedback>,
    _audio_thread: JoinHandle<()>,

    time: f32,
    read_task: Option<Task<Vec<f32>>>,
    chunk_size: u32,
}

struct AudioModel {
    consumer: ringbuf::HeapCons<f32>,
    feedback_tx: crossbeam_channel::Sender<AudioFeedback>,
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
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        audio_producer: Mutex::new(audio_producer),
        audio_feedback_rx,
        _audio_thread: audio_thread,
        time: 0.0,
        read_task: None,
        chunk_size: INITIAL_CHUNK_SIZE,
    }
}

fn exit(_app: &App, _model: Model) {
    // Audio thread will be cleaned up when Model is dropped
}

fn update(app: &App, model: &mut Model) {
    let window = app.window(model.window);
    let device = window.device();
    let queue = window.queue();

    // Process audio feedback - use the latest ideal chunk size from audio thread
    let mut latest_feedback = None;
    while let Ok(feedback) = model.audio_feedback_rx.try_recv() {
        latest_feedback = Some(feedback);
    }

    let mut chunk_size_changed = false;
    if let Some(feedback) = latest_feedback {
        let ideal = feedback.ideal_chunk_size;
        let current = model.chunk_size;

        // Only adjust if we're more than 20% off from ideal (hysteresis)
        let ratio = ideal as f32 / current as f32;
        if ratio < 0.8 || ratio > 1.2 {
            // Blend towards ideal (smooth adjustment)
            let new_chunk_size = ((current as f32 * 0.7) + (ideal as f32 * 0.3)) as u32;
            let new_chunk_size = new_chunk_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);

            if new_chunk_size != current {
                eprintln!("Chunk size adjusted: {} -> {} (ideal: {})", current, new_chunk_size, ideal);
                model.chunk_size = new_chunk_size;
                chunk_size_changed = true;
            }
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
        if let Some(samples) = block_on(future::poll_once(task)) {
            let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            // Push directly to ring buffer - no copying through channel
            let pushed = model.audio_producer.lock().unwrap().push_slice(&samples);
            eprintln!("GPU: produced {} samples (pushed {}), max={:.4}", samples.len(), pushed, max_val);
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

    queue.submit(Some(encoder.finish()));

    // Only spawn new read task if previous one is done
    if model.read_task.is_none() {
        let future = async move {
            let slice = read_buffer.slice(..);
            let (tx, rx) = futures::channel::oneshot::channel();
            slice.map_async(wgpu::MapMode::Read, |res| {
                tx.send(res).expect("Channel closed");
            });
            if let Ok(_) = rx.await {
                let bytes = &slice.get_mapped_range()[..];
                let floats = bytemuck::cast_slice::<u8, f32>(bytes);
                floats.to_vec()
            } else {
                vec![]
            }
        };

        let thread_pool = AsyncComputeTaskPool::get();
        model.read_task = Some(thread_pool.spawn(future));
    }
}

fn audio_fn(audio: &mut AudioModel, buffer: &mut Buffer) {
    static mut FRAME_COUNT: u32 = 0;

    let buffer_len = audio.consumer.occupied_len();
    let callback_samples = buffer.len() * NUM_CHANNELS as usize;

    // Calculate how many samples we need per GPU update frame
    // Audio callbacks run at SAMPLE_RATE/buffer_frames Hz, updates run at TARGET_FRAME_RATE Hz
    // So we need (callback_rate / update_rate) * callback_samples per update
    let buffer_frames = buffer.len() as f32 / NUM_CHANNELS as f32;
    let callback_rate = SAMPLE_RATE / buffer_frames;
    let callbacks_per_update = callback_rate / TARGET_FRAME_RATE;
    let base_chunk_size = (callback_samples as f32 * callbacks_per_update) as usize;

    // Target buffer: enough for ~4 update frames worth of audio
    let target_buffer = base_chunk_size * 4;

    let ideal_chunk_size = if buffer_len < target_buffer {
        // Buffer is low - increase chunk size to catch up
        let deficit = target_buffer - buffer_len;
        base_chunk_size + deficit / 4
    } else {
        // Buffer is high - reduce chunk size to drain excess
        let excess = buffer_len - target_buffer;
        base_chunk_size.saturating_sub(excess / 8)
    };

    // Clamp to bounds and send
    let ideal_chunk_size = (ideal_chunk_size as u32).clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
    audio.feedback_tx.send(AudioFeedback { ideal_chunk_size }).ok();

    unsafe {
        FRAME_COUNT += 1;
        if FRAME_COUNT % 100 == 0 {
            eprintln!("Audio callback: buffer has {}, target {}, ideal chunk {}", buffer_len, target_buffer, ideal_chunk_size);
        }
    }

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

    // Draw particles as a visual indicator
    let t = model.time;
    for i in 0..NUM_PARTICLES {
        let angle = (i as f32 / NUM_PARTICLES as f32) * TAU + t * 0.5;
        let r = 200.0 + (t * 2.0 + i as f32 * 0.1).sin() * 50.0;
        let x = angle.cos() * r;
        let y = angle.sin() * r;
        draw.ellipse()
            .x_y(x, y)
            .w_h(4.0, 4.0)
            .hsla(i as f32 / NUM_PARTICLES as f32, 0.7, 0.6, 0.8);
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
