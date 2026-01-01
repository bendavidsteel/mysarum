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
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> audio_out: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: AudioParams;

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