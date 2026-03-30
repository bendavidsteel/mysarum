// GRA node phase advancement after audio chunk generation
// Concatenated after common.wgsl

@group(0) @binding(0) var<storage, read_write> gra_audio: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gra_state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gra_vel: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> audio_params: AudioParams;

@compute @workgroup_size(64)
fn update_gra_phase(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= audio_params.num_gra_nodes {
        return;
    }

    let chunk_duration = f32(audio_params.chunk_size) / audio_params.sample_rate;

    let color = gra_state[id.x].xyz;

    let old_phase     = gra_audio[id.x].x;
    let old_amp_phase = gra_audio[id.x].y;

    let base_freq = 60.0 + color.r * 40.0 + color.g * 30.0 + color.b * 20.0;
    let new_phase = (old_phase + TAU * base_freq * chunk_duration) % TAU;

    let lfo_freq      = 0.5 + color.r * 2.0 + color.b * 1.5;
    let new_amp_phase = (old_amp_phase + TAU * lfo_freq * chunk_duration) % TAU;

    gra_audio[id.x] = vec4<f32>(new_phase, new_amp_phase, 0.0, 0.0);
}
