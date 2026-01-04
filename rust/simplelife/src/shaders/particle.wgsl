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
    map_x0: f32,
    map_x1: f32,
    map_y0: f32,
    map_y1: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.num_particles {
        return;
    }

    var p = particles[id.x];

    let force = vec2<f32>(sin(params.time), cos(params.time)) * 0.001;

    let mu = pow(0.5, params.dt / params.friction);
    p.vel *= mu;

    p.vel += force * params.dt / params.mass;

    // Update position based on velocity
    p.pos += p.vel * params.dt;

    // Wrap positions to stay in [-1, 1] range
    let map_min = vec2<f32>(params.map_x0, params.map_y0);
    let map_max = vec2<f32>(params.map_x1, params.map_y1);
    let map_size = map_max - map_min;
    p.pos -= floor((p.pos - map_min) / map_size) * map_size;

    // Update energy based on velocity magnitude
    let speed = length(p.vel);
    p.energy = clamp(speed * 0.5, 0.2, 1.0);

    particles[id.x] = p;
}