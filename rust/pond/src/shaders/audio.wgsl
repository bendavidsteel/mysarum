// ── Combined audio synthesis (particles + GRA modal) ────────────────────────
// Concatenated after common.wgsl — all types and helpers are available.
// Part 1: Particle FM synthesis via spatial bin lookup
// Part 2: GRA modal synthesis (8 Chebyshev bandpass modes) via spatial bin lookup

struct ModalFreqs {
    lo: vec4<f32>,  // natural frequencies for bands 0-3
    hi: vec4<f32>,  // natural frequencies for bands 4-7
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> particle_bin_offset: array<u32>;
@group(0) @binding(2) var<storage, read> gra_sorted_pos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> g_bin_offset: array<u32>;
@group(0) @binding(4) var<storage, read> gra_vel: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> modal_amp: array<vec4<f32>>;    // 2 per node
@group(0) @binding(6) var<storage, read> modal_phase: array<vec4<f32>>;  // 2 per node
@group(0) @binding(7) var<storage, read_write> audio_out: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> audio_params: AudioParams;
@group(0) @binding(9) var<uniform> modal_freqs: ModalFreqs;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample_idx = id.x;
    if sample_idx >= audio_params.chunk_size {
        return;
    }

    let sample_rate  = audio_params.sample_rate;
    let volume       = audio_params.volume;
    let max_speed    = audio_params.max_speed;
    let energy_scale = audio_params.energy_scale;

    let t = f32(sample_idx) / sample_rate;

    // Separate accumulators for particle and GRA audio
    var p_left  = 0.0;
    var p_right = 0.0;
    var p_amp   = 0.0;  // for sqrt normalization

    var g_left   = 0.0;
    var g_right  = 0.0;
    var g_weight = 0.0;  // for weight normalization

    // Viewport bounds
    let vp_x0 = audio_params.current_x.x;
    let vp_x1 = audio_params.current_x.y;
    let vp_y0 = audio_params.current_y.x;
    let vp_y1 = audio_params.current_y.y;

    let vp_w = vp_x1 - vp_x0;
    let vp_h = vp_y1 - vp_y0;
    let margin_x = vp_w * 0.1;
    let margin_y = vp_h * 0.1;

    // Extended bounds (viewport + 10% margin)
    let ext_x0 = vp_x0 - margin_x;
    let ext_x1 = vp_x1 + margin_x;
    let ext_y0 = vp_y0 - margin_y;
    let ext_y1 = vp_y1 + margin_y;

    // ── PART 1: Particle audio (spatial bin lookup) ─────────────────────────

    let p_bin_size   = audio_params.p_bin_size;
    let p_num_bins_x = audio_params.p_num_bins_x;
    let p_num_bins_y = audio_params.p_num_bins_y;
    let map_x0       = audio_params.p_map_x0;
    let map_y0       = audio_params.p_map_y0;

    let pbx_min = clamp(u32(max(ext_x0 - map_x0, 0.0) / p_bin_size), 0u, p_num_bins_x - 1u);
    let pbx_max = clamp(u32(max(ext_x1 - map_x0, 0.0) / p_bin_size), 0u, p_num_bins_x - 1u);
    let pby_min = clamp(u32(max(ext_y0 - map_y0, 0.0) / p_bin_size), 0u, p_num_bins_y - 1u);
    let pby_max = clamp(u32(max(ext_y1 - map_y0, 0.0) / p_bin_size), 0u, p_num_bins_y - 1u);

    for (var by = pby_min; by <= pby_max; by++) {
        for (var bx = pbx_min; bx <= pbx_max; bx++) {
            let bin_idx = by * p_num_bins_x + bx;
            let start = particle_bin_offset[bin_idx];
            let end   = particle_bin_offset[bin_idx + 1u];

            for (var pi = start; pi < end; pi++) {
                let p = particles[pi];
                let px = p.pos.x;
                let py = p.pos.y;

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

                let norm_x = clamp((px - vp_x0) / vp_w, 0.0, 1.0);

                let phase = particle_phase(p, t, energy_scale);
                let osc   = sin(phase);

                let amp     = particle_amplitude(p, max_speed, energy_scale);
                let amp_mod = particle_amp_envelope(p, particle_amp_phase(p, t, energy_scale));
                let amp_final = amp * amp_mod * edge_gain;

                p_amp   += amp * edge_gain;
                p_left  += osc * amp_final * (1.0 - norm_x);
                p_right += osc * amp_final * norm_x;
            }
        }
    }

    // ── PART 2: GRA modal audio (spatial bin lookup) ────────────────────────

    let g_bin_size   = audio_params.g_bin_size;
    let g_num_bins_x = audio_params.g_num_bins_x;
    let g_num_bins_y = audio_params.g_num_bins_y;
    let g_map_min    = -audio_params.world_half_audio;

    let gbx_min = clamp(u32(max(ext_x0 - g_map_min, 0.0) / g_bin_size), 0u, g_num_bins_x - 1u);
    let gbx_max = clamp(u32(max(ext_x1 - g_map_min, 0.0) / g_bin_size), 0u, g_num_bins_x - 1u);
    let gby_min = clamp(u32(max(ext_y0 - g_map_min, 0.0) / g_bin_size), 0u, g_num_bins_y - 1u);
    let gby_max = clamp(u32(max(ext_y1 - g_map_min, 0.0) / g_bin_size), 0u, g_num_bins_y - 1u);

    for (var by = gby_min; by <= gby_max; by++) {
        for (var bx = gbx_min; bx <= gbx_max; bx++) {
            let bin_idx = by * g_num_bins_x + bx;
            let start = g_bin_offset[bin_idx];
            let end = g_bin_offset[bin_idx + 1u];

            for (var j = start; j < end; j++) {
                let sorted = gra_sorted_pos[j];
                if sorted.w < 0.5 { continue; }

                let px = sorted.x;
                let py = sorted.y;
                let orig_idx = u32(sorted.z);

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

                let norm_x = clamp((px - vp_x0) / vp_w, 0.0, 1.0);

                // Speed-based excitation
                let speed = length(gra_vel[orig_idx].xy);
                let excitation = 0.5 + 0.5 * clamp(speed / max(audio_params.gra_max_speed, 0.01), 0.0, 1.0);

                // 8 modal bands (2 vec4 each)
                let amp_lo = modal_amp[orig_idx * 2u];
                let amp_hi = modal_amp[orig_idx * 2u + 1u];
                let phase_lo = modal_phase[orig_idx * 2u];
                let phase_hi = modal_phase[orig_idx * 2u + 1u];

                var node_signal = 0.0;
                for (var b = 0u; b < 4u; b++) {
                    let phase = phase_lo[b] + TAU * modal_freqs.lo[b] * t;
                    let s = sin(phase);
                    node_signal += abs(amp_lo[b]) * (s + 0.3 * sin(2.0 * phase) + 0.15 * sin(4.0 * phase));
                }
                for (var b = 0u; b < 4u; b++) {
                    let phase = phase_hi[b] + TAU * modal_freqs.hi[b] * t;
                    let s = sin(phase);
                    node_signal += abs(amp_hi[b]) * (s + 0.3 * sin(2.0 * phase) + 0.15 * sin(4.0 * phase));
                }

                let weighted = excitation * edge_gain;
                node_signal *= weighted;
                g_weight += weighted;

                g_left  += node_signal * (1.0 - norm_x);
                g_right += node_signal * norm_x;
            }
        }
    }

    // ── Normalize and mix ───────────────────────────────────────────────────

    // Particle: sqrt(N) normalization for random-phase oscillators
    let p_norm = select(1.0 / sqrt(p_amp), 0.0, p_amp <= 0.0);

    // GRA modal: weight normalization
    let g_norm = select(1.0 / sqrt(g_weight), 0.0, g_weight < 0.001);

    let left  = p_left * p_norm + g_left * g_norm;
    let right = p_right * p_norm + g_right * g_norm;

    audio_out[sample_idx] = vec2<f32>(left, right) * volume;
}
