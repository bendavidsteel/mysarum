// Modal synthesis audio shader for GRA nodes.
// For each visible node, sum 8 modal sinusoids weighted by Chebyshev bandpass amplitudes.
// Produces stereo output with viewport-based spatial filtering.
// Uses spatially-binned GRA positions for efficient viewport culling.
// This shader is standalone (not concatenated with common.wgsl).

struct ModalAudioParams {
    sample_rate: f32,
    num_gra_nodes: u32,
    chunk_size: u32,
    volume: f32,
    current_x: vec2<f32>,
    current_y: vec2<f32>,
    max_speed: f32,
    g_bin_size: f32,
    g_num_bins_x: u32,
    g_num_bins_y: u32,
    world_half: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ModalFreqs {
    lo: vec4<f32>,  // natural frequencies for bands 0-3
    hi: vec4<f32>,  // natural frequencies for bands 4-7
}

@group(0) @binding(0) var<storage, read> gra_sorted_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> g_bin_offset: array<u32>;
@group(0) @binding(2) var<storage, read> gra_vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> modal_amp: array<vec4<f32>>;    // 2 per node
@group(0) @binding(4) var<storage, read> modal_phase: array<vec4<f32>>;  // 2 per node
@group(0) @binding(5) var<storage, read_write> audio_out: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> params: ModalAudioParams;
@group(0) @binding(7) var<uniform> freqs: ModalFreqs;

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

    // Extended bounds (viewport + 10% margin)
    let ext_x0 = view_left - margin_x;
    let ext_x1 = view_right + margin_x;
    let ext_y0 = view_bottom - margin_y;
    let ext_y1 = view_top + margin_y;

    // Bin range overlapping the extended viewport
    let map_min = -params.world_half;
    let bin_size = params.g_bin_size;
    let num_bins_x = params.g_num_bins_x;
    let num_bins_y = params.g_num_bins_y;

    let bx_min = clamp(u32(max(ext_x0 - map_min, 0.0) / bin_size), 0u, num_bins_x - 1u);
    let bx_max = clamp(u32(max(ext_x1 - map_min, 0.0) / bin_size), 0u, num_bins_x - 1u);
    let by_min = clamp(u32(max(ext_y0 - map_min, 0.0) / bin_size), 0u, num_bins_y - 1u);
    let by_max = clamp(u32(max(ext_y1 - map_min, 0.0) / bin_size), 0u, num_bins_y - 1u);

    for (var by = by_min; by <= by_max; by++) {
        for (var bx = bx_min; bx <= bx_max; bx++) {
            let bin_idx = by * num_bins_x + bx;
            let start = g_bin_offset[bin_idx];
            let end = g_bin_offset[bin_idx + 1u];

            for (var j = start; j < end; j++) {
                let sorted = gra_sorted_pos[j];
                if sorted.w < 0.5 { continue; }  // inactive node

                let px = sorted.x;
                let py = sorted.y;
                let orig_idx = u32(sorted.z);  // original node index stored by sort shader

                // Soft edge gain (viewport proximity)
                let gain_x = clamp(
                    min((px - ext_x0) / margin_x,
                        (ext_x1 - px) / margin_x),
                    0.0, 1.0);
                let gain_y = clamp(
                    min((py - ext_y0) / margin_y,
                        (ext_y1 - py) / margin_y),
                    0.0, 1.0);
                let edge_gain = gain_x * gain_y;
                if edge_gain <= 0.0 { continue; }

                // Stereo position
                let norm_x = clamp((px - view_left) / view_width, 0.0, 1.0);

                // Speed-based excitation: faster nodes resonate louder
                let speed = length(gra_vel[orig_idx].xy);
                let excitation = 0.2 + 0.8 * clamp(speed / max(params.max_speed, 0.01), 0.0, 1.0);

                // Read modal amplitudes and phases (8 bands in 2 vec4 each)
                let amp_lo = modal_amp[orig_idx * 2u];
                let amp_hi = modal_amp[orig_idx * 2u + 1u];
                let phase_lo = modal_phase[orig_idx * 2u];
                let phase_hi = modal_phase[orig_idx * 2u + 1u];

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
        }
    }

    // Normalize by total contribution weight
    let norm = select(1.0 / total_weight, 1.0, total_weight < 0.001);

    audio_out[sample_idx] = vec2<f32>(left, right) * norm * params.volume;
}
