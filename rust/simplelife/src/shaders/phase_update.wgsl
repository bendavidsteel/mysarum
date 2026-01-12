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

const TAU: f32 = 6.28318530718;

struct PhaseUpdateParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: PhaseUpdateParams;

// ============================================================================
// Waveform functions - MUST match audio.wgsl exactly
// ============================================================================

// Compute base frequency for a particle
fn compute_frequency(p: Particle) -> f32 {
    return 100.0 + 500.0 * p.species.x;
}

// Compute amplitude for a particle
fn compute_amplitude(p: Particle) -> f32 {
    return p.energy * 0.5 + 0.002;
}

// Compute oscillator value at a given phase
fn compute_oscillator(phase: f32) -> f32 {
    return sin(phase);// + 0.5 * sin(phase * 2.0) + 0.25 * sin(phase * 3.0);
}

// Compute the instantaneous phase for a particle at a given time offset
fn compute_phase(p: Particle, t: f32) -> f32 {
    let freq = compute_frequency(p);
    return TAU * freq * t + p.phase;
}

// ============================================================================

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_particles) { return; }

    let p = particles[id.x];
    let chunk_duration = f32(params.chunk_size) / params.sample_rate;

    // Advance phase by the same amount the audio shader would have used
    let end_phase = compute_phase(p, chunk_duration);
    particles[id.x].phase = end_phase % TAU;
}