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

struct SimParams {
    dt: f32,
    time: f32,
    num_particles: u32,
    friction: f32,
    mass: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.num_particles {
        return;
    }

    var p = particles[id.x];

    let force = vec2<f32>(sin(params.time), cos(params.time)) * 10.0;

    let mu = pow(0.5, params.dt / params.friction);
    p.vel *= mu;

    p.vel += force * params.dt / params.mass;

    // Update position based on velocity
    p.pos += p.vel * params.dt;

    // Wrap positions to stay in [-1, 1] range
    p.pos = (p.pos + 1.0) % 2.0 - 1.0;
    // Handle negative modulo correctly
    if (p.pos.x < -1.0) { p.pos.x += 2.0; }
    if (p.pos.y < -1.0) { p.pos.y += 2.0; }

    // Update energy based on velocity magnitude
    let speed = length(p.vel);
    p.energy = clamp(speed * 0.5, 0.2, 1.0);

    particles[id.x] = p;
}