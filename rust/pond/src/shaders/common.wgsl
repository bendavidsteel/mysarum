// ── Shared types for pond (particle life + GRA) ──────────────────────────────

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    phase: f32,
    energy: f32,
    species: vec2<f32>,
    alpha: vec2<f32>,
    interaction: vec2<f32>,
    amp_phase: f32,
    _pad: f32,
}

struct SimParams {
    // World
    world_half: f32,
    dt: f32,
    time: f32,

    // Particle counts & physics
    num_particles: u32,
    particle_friction: f32,
    particle_mass: f32,
    particle_radius: f32,
    particle_collision_radius: f32,
    particle_collision_strength: f32,
    particle_max_force: f32,
    particle_copy_radius: f32,
    particle_copy_cos_sim: f32,
    particle_copy_prob: f32,

    // Particle spatial hash
    p_bin_size: f32,
    p_num_bins_x: u32,
    p_num_bins_y: u32,

    // GRA counts & physics
    num_gra_nodes: u32,
    num_gra_connections: u32,
    gra_spring_length: f32,
    gra_spring_stiffness: f32,
    gra_damping: f32,
    gra_max_velocity: f32,

    // GRA spatial hash (for particle→GRA repulsion lookups)
    g_bin_size: f32,
    g_num_bins_x: u32,
    g_num_bins_y: u32,

    // Particle↔GRA repulsion
    gra_repulsion_radius: f32,
    gra_repulsion_strength: f32,

    // Pre-computed friction: pow(0.5, dt / particle_friction)
    particle_friction_mu: f32,

    current_strength: f32,
    _pad0: u32,
}

struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    num_gra_nodes: u32,
    chunk_size: u32,
    volume: f32,
    current_x: vec2<f32>,
    current_y: vec2<f32>,
    max_speed: f32,
    energy_scale: f32,
    gra_max_speed: f32,
    // particle bin info for spatial audio
    p_map_x0: f32,
    p_map_y0: f32,
    p_bin_size: f32,
    p_num_bins_x: u32,
    p_num_bins_y: u32,
    // GRA bin info for spatial audio
    g_bin_size: f32,
    g_num_bins_x: u32,
    g_num_bins_y: u32,
    world_half_audio: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

struct RenderUniforms {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    particle_size: f32,
    gra_node_radius: f32,
    num_particles: u32,
    num_gra_nodes: u32,
    num_gra_connections: u32,
    window_aspect: f32,
    world_half: f32,
    max_speed: f32,
    energy_scale: f32,
    current_strength: f32,
    time: f32,
    _pad0: u32,
}

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const EPSILON: f32 = 1e-6;

// ── Spatial binning helpers ──────────────────────────────────────────────────

fn get_bin_index(pos: vec2<f32>, world_half: f32, bin_size: f32, num_bins_x: u32, num_bins_y: u32) -> u32 {
    let map_min = -world_half;
    let bx = clamp(u32((pos.x - map_min) / bin_size), 0u, num_bins_x - 1u);
    let by = clamp(u32((pos.y - map_min) / bin_size), 0u, num_bins_y - 1u);
    return by * num_bins_x + bx;
}

fn get_bin_xy(pos: vec2<f32>, world_half: f32, bin_size: f32, num_bins_x: u32, num_bins_y: u32) -> vec2<u32> {
    let map_min = -world_half;
    let bx = clamp(u32((pos.x - map_min) / bin_size), 0u, num_bins_x - 1u);
    let by = clamp(u32((pos.y - map_min) / bin_size), 0u, num_bins_y - 1u);
    return vec2(bx, by);
}

// ── Toroidal wrapping ────────────────────────────────────────────────────────

fn wrap_delta_1d(d: f32, world_half: f32) -> f32 {
    let size = world_half * 2.0;
    if d > world_half { return d - size; }
    if d < -world_half { return d + size; }
    return d;
}

fn wrap_pos(p: vec2<f32>, world_half: f32) -> vec2<f32> {
    let size = world_half * 2.0;
    return ((p + vec2(world_half)) % vec2(size) + vec2(size)) % vec2(size) - vec2(world_half);
}

// ── Audio helpers (particle) ─────────────────────────────────────────────────

fn normalize_energy(energy: f32, energy_scale: f32) -> f32 {
    return clamp(energy / energy_scale, -1.0, 1.0);
}

fn particle_frequency(p: Particle, t: f32, energy_scale: f32) -> f32 {
    let energy_normalized = -1.0 * normalize_energy(p.energy, energy_scale) + 1.0;
    let base_freq = 80.0 + pow(energy_normalized, 3.0) * 200.0;
    let mod1 = smoothstep(0.0, 1.0, p.species.x);
    let mod2 = smoothstep(0.0, 1.0, p.species.y);
    let mod3 = smoothstep(0.0, 1.0, -p.species.x);
    let mod4 = smoothstep(0.0, 1.0, -p.species.y);
    return base_freq + mod1 * sin(TAU * base_freq * 2.0 * t) + mod2 * sin(TAU * base_freq * 3.0 * t) + mod3 * sin(TAU * base_freq * 5.0 * t) + mod4 * sin(TAU * base_freq * 7.0 * t);
}

fn particle_amplitude(p: Particle, max_speed: f32, energy_scale: f32) -> f32 {
    let speed = length(p.vel);
    let speed_normalized = clamp(speed / max(max_speed, 0.01), 0.0, 1.0);
    let energy_normalized = normalize_energy(p.energy, energy_scale);
    let energy_contrib = (1.0 - energy_normalized) * 0.5;
    let speed_contrib = speed_normalized;
    return clamp(0.05 + energy_contrib * 0.35 + speed_contrib * 0.6, 0.0, 1.0);
}

fn particle_phase(p: Particle, t: f32, energy_scale: f32) -> f32 {
    let freq = particle_frequency(p, t, energy_scale);
    return p.phase + TAU * freq * t;
}

// E: Species-driven AM — magnitude controls rate, species.y controls attack shape
fn particle_amp_phase(p: Particle, t: f32, energy_scale: f32) -> f32 {
    let energy_normalized = (1.0 - normalize_energy(p.energy, energy_scale)) * 0.5;
    let mag = length(p.species);
    var lfo_freq = 0.1 + mag * 4.0;  // 0.1–~4 Hz based on species magnitude
    lfo_freq += energy_normalized * 0.5;
    lfo_freq = min(lfo_freq, 8.0);
    return p.amp_phase + TAU * lfo_freq * t;
}

// E: Shaped AM envelope — species.y controls sharpness (soft pulse ↔ percussive)
fn particle_amp_envelope(p: Particle, amp_phase: f32) -> f32 {
    let sharpness = 1.0 + smoothstep(0.0, 1.0, p.species.y) * 4.0; // exponent 1–5
    let raw = sin(amp_phase);
    let shaped = pow(abs(raw), 1.0 / sharpness) * sign(raw);
    return 0.2 + 0.8 * shaped;
}

// ── Audio helpers (GRA node) ─────────────────────────────────────────────────

fn gra_frequency(color: vec3<f32>, t: f32) -> f32 {
    let base_freq = 60.0 + color.r * 40.0 + color.g * 30.0 + color.b * 20.0;
    let mod1 = color.r * sin(TAU * base_freq * 2.0 * t);
    let mod2 = color.g * sin(TAU * base_freq * 3.0 * t);
    let mod3 = color.b * sin(TAU * base_freq * 5.0 * t);
    return base_freq + mod1 * 8.0 + mod2 * 5.0 + mod3 * 3.0;
}

fn gra_amplitude(speed: f32, max_speed: f32, neighbor_hint: f32) -> f32 {
    let speed_norm = clamp(speed / max(max_speed, 0.01), 0.0, 1.0);
    return clamp(0.08 + speed_norm * 0.7 + neighbor_hint * 0.2, 0.0, 1.0);
}

// ── Gradient noise for current field ────────────────────────────────────────

fn grad_hash(p: vec2<f32>) -> vec2<f32> {
    let k = vec2(0.3183099, 0.3678794);
    var pp = p * k + k.yx;
    return -1.0 + 2.0 * fract(16.0 * k * fract(pp.x * pp.y * (pp.x + pp.y)));
}

fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(
        mix(dot(grad_hash(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
            dot(grad_hash(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
        mix(dot(grad_hash(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
            dot(grad_hash(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x),
        u.y
    );
}

// ── Ocean current ───────────────────────────────────────────────────────────

fn current_radial_profile(r: f32) -> f32 {
    // 0 at centre, peak at r=0.75, ~0.25 at r=1.0
    return smoothstep(0.0, 0.75, r) * mix(1.0, 0.25, smoothstep(0.75, 1.0, r));
}

fn current_at_pos(pos: vec2<f32>, world_half: f32, time: f32, strength: f32) -> vec2<f32> {
    let r = length(pos) / world_half;
    if r < 0.001 {
        return vec2(0.0);
    }

    let profile = current_radial_profile(r);

    // Base tangent (counter-clockwise)
    let tangent = normalize(vec2(-pos.y, pos.x));

    // Coarse, slowly evolving Perlin noise
    let ns = 0.3;
    let t = time * 0.05;
    let angle_noise = gradient_noise(pos * ns + vec2(t, 0.0)) * 6.4;
    let str_noise = gradient_noise(pos * ns + vec2(0.0, t + 43.0)) * 2.8;

    // Rotate tangent by noise angle
    let ca = cos(angle_noise);
    let sa = sin(angle_noise);
    let dir = vec2(tangent.x * ca - tangent.y * sa, tangent.x * sa + tangent.y * ca);

    return dir * profile * strength * (1.0 + str_noise);
}

fn current_magnitude_at_pos(pos: vec2<f32>, world_half: f32, time: f32, strength: f32) -> f32 {
    let r = length(pos) / world_half;
    let profile = current_radial_profile(r);
    let ns = 0.3;
    let t = time * 0.05;
    let str_noise = gradient_noise(pos * ns + vec2(0.0, t + 43.0)) * 2.8;
    return profile * strength * (1.0 + str_noise);
}
