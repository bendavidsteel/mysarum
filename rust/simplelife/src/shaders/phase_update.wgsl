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

fn compute_frequency(p: Particle) -> f32 {
    // Base frequency derived from particle position (same as audio shader)
    return 100.0 + (p.pos.x + 0.5) * 400.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_particles) { return; }

    let freq = compute_frequency(particles[id.x]);
    let phase_advance = f32(params.chunk_size) * freq / params.sample_rate * TAU;
    particles[id.x].phase = (particles[id.x].phase + phase_advance) % TAU;
}