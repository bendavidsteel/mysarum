// Particle rendering shader with instanced quads
// Standalone file (not concatenated after common.wgsl)

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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) core_hue: f32,
    @location(2) edge_hue: f32,
    @location(3) energy: f32,
    @location(4) vel: vec2<f32>,
    @location(5) alpha: vec2<f32>,
    @location(6) particle_id: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// ── Color helpers ───────────────────────────────────────────────────────────

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - abs(h6 % 2.0 - 1.0));
    let m = l - c * 0.5;
    var rgb: vec3<f32>;
    if h6 < 1.0 { rgb = vec3(c, x, 0.0); }
    else if h6 < 2.0 { rgb = vec3(x, c, 0.0); }
    else if h6 < 3.0 { rgb = vec3(0.0, c, x); }
    else if h6 < 4.0 { rgb = vec3(0.0, x, c); }
    else if h6 < 5.0 { rgb = vec3(x, 0.0, c); }
    else { rgb = vec3(c, 0.0, x); }
    return rgb + vec3(m);
}

// ── Procedural noise (replaces blue noise texture) ─────────────────────────

fn hash_f32(x: f32) -> vec2<f32> {
    let s = vec2(x * 127.1, x * 311.7);
    return fract(sin(s) * 43758.5453123);
}

fn hash2d(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise with smooth interpolation
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2d(i);
    let b = hash2d(i + vec2(1.0, 0.0));
    let c = hash2d(i + vec2(0.0, 1.0));
    let d = hash2d(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ── Vertex shader ───────────────────────────────────────────────────────────

@vertex
fn vs_main(
    in: VertexInput,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    var out: VertexOutput;

    if instance >= uniforms.num_particles {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        out.uv = vec2(0.0);
        out.core_hue = 0.0;
        out.edge_hue = 0.0;
        out.energy = 0.0;
        out.vel = vec2(0.0);
        out.alpha = vec2(0.0);
        out.particle_id = 0.0;
        return out;
    }

    let p = particles[instance];

    // Orthographic projection: world pos -> NDC
    let range_x = uniforms.max_x - uniforms.min_x;
    let range_y = uniforms.max_y - uniforms.min_y;
    let cx = 2.0 * (p.pos.x - uniforms.min_x) / range_x - 1.0;
    let cy = 2.0 * (p.pos.y - uniforms.min_y) / range_y - 1.0;
    let center = vec2(cx, cy);

    // Quad size in clip space (3x for bloom halo, matching reference)
    let world_size = uniforms.particle_size * 3.0;
    let size_x = world_size / range_x * 2.0;
    let size_y = world_size / range_y * 2.0;

    out.clip_position = vec4(center + in.position * vec2(size_x, size_y), 0.0, 1.0);
    out.uv = in.uv;

    // Hues from species (matching reference: map [-1,1] to [0,1])
    out.core_hue = (p.species.x + 1.0) * 0.5;
    out.edge_hue = (p.species.y + 1.0) * 0.5;

    // Normalised energy
    out.energy = p.energy;

    out.vel = p.vel;
    out.alpha = p.alpha;
    out.particle_id = f32(instance);

    return out;
}

// ── Fragment shader (matching reference render detail) ──────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let centered_uv = in.uv - vec2(0.5);

    // Unstretched distance for circular core
    let dist = length(centered_uv) * 2.0;

    // Velocity-based glow stretching (trail behind particle)
    let speed = length(in.vel);
    let safe_speed = max(speed, 0.0001);
    let vel_dir = select(vec2(0.0, 1.0), in.vel / safe_speed, speed > 0.001);
    let speed_norm = min(speed / uniforms.max_speed, 1.0);

    // How much is this pixel behind the particle (opposite to velocity)?
    let behind_amount = dot(centered_uv, -vel_dir);

    // Stretch factor: pixels behind the particle expand glow backward
    let stretch = 1.0 + max(behind_amount, 0.0) * speed_norm * 10.0;

    // Stretched distance for glow trail
    let glow_dist = dist / stretch;

    if glow_dist > 1.0 { discard; }

    // Energy-driven parameters
    let scaled_energy = tanh(in.energy);
    let core_radius = 0.2 + 0.1 * scaled_energy;
    let energy_brightness = 0.8 + 0.2 * scaled_energy;

    // Smooth Gaussian core
    let core_sharpness = 8.0 + 4.0 * scaled_energy;
    let core_falloff = dist / core_radius;
    let core = exp(-core_sharpness * core_falloff * core_falloff);

    // Membrane ring at core boundary
    let membrane_width = 0.15;
    let membrane = smoothstep(core_radius - membrane_width, core_radius, dist)
                  * smoothstep(core_radius + membrane_width, core_radius, dist);

    // Glow with velocity stretching
    let glow = pow(1.0 - glow_dist, 2.0) * (0.5 + 0.2 * scaled_energy);

    // Procedural noise for organic texture (replaces blue noise texture)
    let noise_offset = hash_f32(in.particle_id);
    let noise_uv = in.uv * 8.0 + noise_offset * 64.0;
    let noise_val = value_noise(noise_uv);

    // Noise modulation: visible variation in brightness and alpha
    let noise_strength = 0.5 + 0.2 * abs(in.alpha.x);
    let noise_mod = 1.0 - noise_strength + noise_strength * noise_val;

    // Dual hue: interpolate from core color to edge color based on distance
    let hue_blend = smoothstep(0.0, core_radius * 2.0, dist);
    let hue_t = clamp(hue_blend + in.alpha.y * 0.2, 0.0, 1.0);
    let core_rgb = hsl_to_rgb(in.core_hue, 0.9, energy_brightness * 0.5);
    let edge_rgb = hsl_to_rgb(in.edge_hue, 0.7, energy_brightness * 0.6);
    let rgb = mix(core_rgb, edge_rgb, hue_t);

    // Apply noise only inside the core/membrane ring, leave outer glow clean
    let inner = (membrane * 0.4 + core * 0.2) * noise_mod;
    let combined = inner + glow;
    let alpha = combined * 0.9;

    return vec4(rgb * combined, alpha);
}
