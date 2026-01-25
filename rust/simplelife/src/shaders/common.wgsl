// Shared definitions for particle and audio shaders

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

// Spatial binning helpers
struct BinInfo {
    binIndex: u32,
    binX: u32,
    binY: u32,
}

fn getBinInfo(pos: vec2<f32>, map_x0: f32, map_y0: f32, bin_size: f32, num_bins_x: u32, num_bins_y: u32) -> BinInfo {
    let binX = u32((pos.x - map_x0) / bin_size);
    let binY = u32((pos.y - map_y0) / bin_size);
    let clampedX = clamp(binX, 0u, num_bins_x - 1u);
    let clampedY = clamp(binY, 0u, num_bins_y - 1u);
    return BinInfo(clampedY * num_bins_x + clampedX, clampedX, clampedY);
}

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
    bin_size: f32,
    num_bins_x: u32,
    num_bins_y: u32,
    radius: f32,
    collision_radius: f32,
    collision_strength: f32,
    max_force_strength: f32,
    copy_radius: f32,
    copy_cos_sim_threshold: f32,
    copy_probability: f32,
    _pad: f32
}

// Waveform functions for audio synthesis

fn compute_frequency(p: Particle, t: f32) -> f32 {
    let base_freq = 100.0 + tanh(p.energy) * 10.0;
    return 100.0 + p.species.x * sin(TAU * base_freq * 2.0 * t) + p.species.y * cos(TAU * base_freq * 3.0 * t);
}

fn compute_amplitude(p: Particle) -> f32 {
    let speed = length(p.vel);
    return -tanh(p.energy) * 0.1 + 0.8 + 0.1 * speed;
}

fn compute_oscillator(phase: f32) -> f32 {
    return sin(phase);
}

fn compute_phase(p: Particle, t: f32) -> f32 {
    let freq = compute_frequency(p, t);
    return p.phase + TAU * freq * t;
}