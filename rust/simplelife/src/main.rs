use nannou::prelude::*;
use nannou::wgpu::BufferInitDescriptor;
use nannou_audio as audio;
use nannou_audio::Buffer;
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use std::collections::VecDeque;

const NUM_PARTICLES: u32 = 512;
const AUDIO_CHUNK_SIZE: u32 = 8192;  // Larger chunks to reduce gaps

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
    PushSamples(Vec<f32>),
    Exit,
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

    audio_tx: Sender<AudioCommand>,
    _audio_thread: JoinHandle<()>,

    time: f32,
    read_task: Option<Task<Vec<f32>>>,
}

struct AudioModel {
    buffer: VecDeque<f32>,
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

    // Start audio stream on a separate thread
    let audio_host = audio::Host::new();
    let audio_model = AudioModel {
        buffer: VecDeque::with_capacity(AUDIO_CHUNK_SIZE as usize * 32),
    };

    let (audio_tx, audio_rx) = mpsc::channel();
    let audio_thread = std::thread::spawn(move || {
        let stream = audio_host
            .new_output_stream(audio_model)
            .render(audio_fn)
            .build()
            .unwrap();
        stream.play().unwrap();

        // Process commands from main thread
        loop {
            match audio_rx.recv() {
                Ok(AudioCommand::PushSamples(samples)) => {
                    let len = samples.len();
                    stream.send(move |audio| {
                        audio.buffer.extend(samples);
                        eprintln!("Audio thread: pushed {} samples, buffer now {}", len, audio.buffer.len());
                    }).ok();
                }
                Ok(AudioCommand::Exit) | Err(_) => {
                    stream.pause().ok();
                    break;
                }
            }
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
        size: (AUDIO_CHUNK_SIZE * 2 * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let audio_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_staging"),
        size: (AUDIO_CHUNK_SIZE * 2 * std::mem::size_of::<f32>() as u32) as u64,
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

    let audio_out_size = (AUDIO_CHUNK_SIZE * 2 * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;
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
            sample_rate: 44100.0,
            num_particles: NUM_PARTICLES,
            chunk_size: AUDIO_CHUNK_SIZE,
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
        audio_tx,
        _audio_thread: audio_thread,
        time: 0.0,
        read_task: None,
    }
}

fn exit(_app: &App, model: Model) {
    model.audio_tx.send(AudioCommand::Exit).ok();
}

fn update(app: &App, model: &mut Model) {
    let window = app.window(model.window);
    let device = window.device();
    let queue = window.queue();

    // Check if previous read task completed
    if let Some(task) = &mut model.read_task {
        if let Some(samples) = block_on(future::poll_once(task)) {
            let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            eprintln!("GPU: sending {} samples, max={:.4}", samples.len(), max_val);
            model.audio_tx.send(AudioCommand::PushSamples(samples)).ok();
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
    let audio_buf_size = (AUDIO_CHUNK_SIZE * 2 * std::mem::size_of::<f32>() as u32) as u64;
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
        pass.dispatch_workgroups((AUDIO_CHUNK_SIZE + 63) / 64, 1, 1);
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
    unsafe {
        FRAME_COUNT += 1;
        if FRAME_COUNT % 100 == 0 {
            eprintln!("Audio callback: buffer has {}, needs {}", audio.buffer.len(), buffer.len());
        }
    }

    // Fill the output buffer from our local buffer
    for frame in buffer.frames_mut() {
        for sample in frame.iter_mut() {
            *sample = audio.buffer.pop_front().unwrap_or(0.0);
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
