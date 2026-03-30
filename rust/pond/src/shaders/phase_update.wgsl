// Particle phase advancement after audio chunk generation
// Concatenated after common.wgsl

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> audio_params: AudioParams;

@compute @workgroup_size(256)
fn update_particle_phase(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= audio_params.num_particles {
        return;
    }

    let energy_scale   = audio_params.energy_scale;
    let chunk_duration = f32(audio_params.chunk_size) / audio_params.sample_rate;

    var p = particles[id.x];

    p.phase     = particle_phase(p, chunk_duration, energy_scale) % TAU;
    p.amp_phase = particle_amp_phase(p, chunk_duration, energy_scale) % TAU;

    particles[id.x] = p;
}
