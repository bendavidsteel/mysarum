// Modal synthesis audio shader for GRA nodes.
// For each visible node, sum 8 modal sinusoids weighted by Chebyshev bandpass amplitudes.
// Produces stereo output with viewport-based spatial filtering.
// This shader is standalone (not concatenated with common.wgsl).

struct ModalAudioParams {
    sample_rate: f32,
    num_gra_nodes: u32,
    chunk_size: u32,
    volume: f32,
    current_x: vec2<f32>,
    current_y: vec2<f32>,
    max_speed: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ModalFreqs {
    lo: vec4<f32>,  // natural frequencies for bands 0-3
    hi: vec4<f32>,  // natural frequencies for bands 4-7
}

@group(0) @binding(0) var<storage, read> gra_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gra_vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> modal_amp: array<vec4<f32>>;    // 2 per node
@group(0) @binding(3) var<storage, read> modal_phase: array<vec4<f32>>;  // 2 per node
@group(0) @binding(4) var<storage, read_write> audio_out: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> params: ModalAudioParams;
@group(0) @binding(6) var<uniform> freqs: ModalFreqs;

const TAU: f32 = 6.283185307;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.chunk_size { return; }

    let sample_idx = id.x;
    let t = f32(sample_idx) / params.sample_rate;

    var left = 0.0;
    var right = 0.0;
    var total_weight = 0.0;

    // Viewport bounds
    let view_left = params.current_x.x;
    let view_right = params.current_x.y;
    let view_bottom = params.current_y.x;
    let view_top = params.current_y.y;
    let view_width = view_right - view_left;
    let view_height = view_top - view_bottom;
    let margin_x = view_width * 0.1;
    let margin_y = view_height * 0.1;

    for (var i = 0u; i < params.num_gra_nodes; i++) {
        let pos = gra_pos[i];
        if pos.w < 0.5 { continue; }  // inactive node

        let px = pos.x;
        let py = pos.y;

        // Soft edge gain (viewport proximity)
        let gain_x = clamp(
            min((px - (view_left - margin_x)) / margin_x,
                ((view_right + margin_x) - px) / margin_x),
            0.0, 1.0);
        let gain_y = clamp(
            min((py - (view_bottom - margin_y)) / margin_y,
                ((view_top + margin_y) - py) / margin_y),
            0.0, 1.0);
        let edge_gain = gain_x * gain_y;
        if edge_gain <= 0.0 { continue; }

        // Stereo position
        let norm_x = clamp((px - view_left) / view_width, 0.0, 1.0);

        // Speed-based excitation: faster nodes resonate louder
        let speed = length(gra_vel[i].xy);
        let excitation = 0.2 + 0.8 * clamp(speed / max(params.max_speed, 0.01), 0.0, 1.0);

        // Read modal amplitudes and phases (8 bands in 2 vec4 each)
        let amp_lo = modal_amp[i * 2u];
        let amp_hi = modal_amp[i * 2u + 1u];
        let phase_lo = modal_phase[i * 2u];
        let phase_hi = modal_phase[i * 2u + 1u];

        var node_signal = 0.0;

        // Bands 0-3: low-frequency modes
        for (var b = 0u; b < 4u; b++) {
            let phase = phase_lo[b] + TAU * freqs.lo[b] * t;
            node_signal += abs(amp_lo[b]) * sin(phase);
        }
        // Bands 4-7: high-frequency modes
        for (var b = 0u; b < 4u; b++) {
            let phase = phase_hi[b] + TAU * freqs.hi[b] * t;
            node_signal += abs(amp_hi[b]) * sin(phase);
        }

        let weighted = excitation * edge_gain;
        node_signal *= weighted;
        total_weight += weighted;

        // Stereo panning
        left += node_signal * (1.0 - norm_x);
        right += node_signal * norm_x;
    }

    // Normalize by total contribution weight
    let norm = select(1.0 / total_weight, 1.0, total_weight < 0.001);

    audio_out[sample_idx] = vec2<f32>(left, right) * norm * params.volume;
}
