#define_import_path simplelife::common

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