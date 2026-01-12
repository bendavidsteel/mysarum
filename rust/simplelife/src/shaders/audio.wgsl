// Particle struct - shared definition
struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    phase: f32,
    energy: f32,
    species: vec2<f32>,
    alpha: vec2<f32>,
    interaction: vec2<f32>,
}

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    volume: f32,
    current_x: vec2<f32>,  // Current viewport x (min, max)
    current_y: vec2<f32>,  // Current viewport y (min, max)
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> audio_out: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: AudioParams;

// Compute base frequency for a particle
fn compute_frequency(p: Particle) -> f32 {
    return 100.0 + 500.0 * p.species.x;
}

// Compute amplitude for a particle
fn compute_amplitude(p: Particle) -> f32 {
    return p.energy * 0.5 + 0.002;  // base amplitude even with low energy
}

// Compute oscillator value at a given phase
fn compute_oscillator(phase: f32) -> f32 {
    // Simple sine with harmonics
    return sin(phase);// + 0.5 * sin(phase * 2.0) + 0.25 * sin(phase * 3.0);
}

// Compute the instantaneous phase for a particle at a given time offset
fn compute_phase(p: Particle, t: f32) -> f32 {
    let freq = compute_frequency(p);
    return TAU * freq * t + p.phase;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.chunk_size {
        return;
    }

    let sample_idx = id.x;
    let t = f32(sample_idx) / params.sample_rate;

    var left = 0.0;
    var right = 0.0;
    var visible_count = 0u;

    // Viewport bounds
    let view_left = params.current_x.x;
    let view_right = params.current_x.y;
    let view_bottom = params.current_y.x;
    let view_top = params.current_y.y;
    let view_width = view_right - view_left;
    let view_height = view_top - view_bottom;

    // Sum contributions from visible particles only
    for (var i = 0u; i < params.num_particles; i++) {
        let p = particles[i];

        // Skip particles outside the viewport
        if (p.pos.x < view_left || p.pos.x > view_right ||
            p.pos.y < view_bottom || p.pos.y > view_top) {
            continue;
        }

        visible_count += 1u;

        // Normalize position within viewport (0 to 1)
        let norm_x = (p.pos.x - view_left) / view_width;
        let norm_y = (p.pos.y - view_bottom) / view_height;

        // Compute waveform
        let phase = compute_phase(p, t);
        let osc = compute_oscillator(phase);
        let amp = compute_amplitude(p);

        // Stereo pan based on x position within viewport
        left += osc * amp * (1.0 - norm_x);
        right += osc * amp * norm_x;
    }

    // Normalize by visible particle count and apply volume
    if (visible_count > 0u) {
        let norm = 1.0 / f32(visible_count);
        audio_out[sample_idx] = vec2<f32>(left, right) * norm * params.volume;
    } else {
        audio_out[sample_idx] = vec2<f32>(0.0, 0.0);
    }
}