// ── Combined audio synthesis (particles + GRA nodes) ────────────────────────
// Concatenated after common.wgsl — all types and helpers are available.

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> particle_bin_offset: array<u32>;
@group(0) @binding(2) var<storage, read> gra_pos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gra_vel: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> gra_state: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> gra_audio: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> audio_out: array<vec2<f32>>;
@group(0) @binding(7) var<uniform> audio_params: AudioParams;

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

    var left        = 0.0;
    var right       = 0.0;
    var visible_amp = 0.0;

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

    let bin_size    = audio_params.p_bin_size;
    let num_bins_x  = audio_params.p_num_bins_x;
    let num_bins_y  = audio_params.p_num_bins_y;
    let map_x0      = audio_params.p_map_x0;
    let map_y0      = audio_params.p_map_y0;

    // Bin range overlapping the extended viewport
    let bx_min = clamp(u32(max(ext_x0 - map_x0, 0.0) / bin_size), 0u, num_bins_x - 1u);
    let bx_max = clamp(u32(max(ext_x1 - map_x0, 0.0) / bin_size), 0u, num_bins_x - 1u);
    let by_min = clamp(u32(max(ext_y0 - map_y0, 0.0) / bin_size), 0u, num_bins_y - 1u);
    let by_max = clamp(u32(max(ext_y1 - map_y0, 0.0) / bin_size), 0u, num_bins_y - 1u);

    for (var by = by_min; by <= by_max; by++) {
        for (var bx = bx_min; bx <= bx_max; bx++) {
            let bin_idx = by * num_bins_x + bx;
            let start = particle_bin_offset[bin_idx];
            let end   = particle_bin_offset[bin_idx + 1u];

            for (var pi = start; pi < end; pi++) {
                let p = particles[pi];
                let px = p.pos.x;
                let py = p.pos.y;

                // Soft edge gain: 1.0 inside viewport, linear ramp in margin zone, 0 outside
                var edge_gain = 1.0;
                if px < vp_x0 {
                    edge_gain *= clamp((px - ext_x0) / margin_x, 0.0, 1.0);
                } else if px > vp_x1 {
                    edge_gain *= clamp((ext_x1 - px) / margin_x, 0.0, 1.0);
                }
                if py < vp_y0 {
                    edge_gain *= clamp((py - ext_y0) / margin_y, 0.0, 1.0);
                } else if py > vp_y1 {
                    edge_gain *= clamp((ext_y1 - py) / margin_y, 0.0, 1.0);
                }

                if edge_gain <= 0.0 {
                    continue;
                }

                let norm_x = clamp((px - vp_x0) / vp_w, 0.0, 1.0);

                let phase = particle_phase(p, t, energy_scale);
                let osc   = sin(phase);

                let amp     = particle_amplitude(p, max_speed, energy_scale);
                let amp_mod = 0.2 + 0.8 * sin(particle_amp_phase(p, t, energy_scale));
                let amp_final = amp * amp_mod * edge_gain;

                visible_amp += amp * edge_gain;
                left  += osc * amp_final * (1.0 - norm_x);
                right += osc * amp_final * norm_x;
            }
        }
    }

    // ── PART 2: GRA node audio (iterate all nodes) ──────────────────────────

    let gra_max_speed = audio_params.gra_max_speed;

    for (var i = 0u; i < audio_params.num_gra_nodes; i++) {
        let pos         = gra_pos[i];
        let vel         = gra_vel[i];
        let state       = gra_state[i];
        let audio_state = gra_audio[i];

        // pos.w < 0.5 means inactive node
        if pos.w < 0.5 {
            continue;
        }

        let nx = pos.x;
        let ny = pos.y;

        // Soft edge gain
        var edge_gain = 1.0;
        if nx < vp_x0 {
            edge_gain *= clamp((nx - ext_x0) / margin_x, 0.0, 1.0);
        } else if nx > vp_x1 {
            edge_gain *= clamp((ext_x1 - nx) / margin_x, 0.0, 1.0);
        }
        if ny < vp_y0 {
            edge_gain *= clamp((ny - ext_y0) / margin_y, 0.0, 1.0);
        } else if ny > vp_y1 {
            edge_gain *= clamp((ext_y1 - ny) / margin_y, 0.0, 1.0);
        }

        if edge_gain <= 0.0 {
            continue;
        }

        let norm_x = clamp((nx - vp_x0) / vp_w, 0.0, 1.0);

        let color = state.xyz;
        let speed = length(vel.xy);

        let freq  = gra_frequency(color, t);
        let phase = audio_state.x + TAU * freq * t;
        let osc   = sin(phase);

        let amp = gra_amplitude(speed, gra_max_speed, color.g * 0.5);

        let lfo_freq = 0.5 + color.r * 2.0 + color.b * 1.5;
        let amp_mod  = 0.3 + 0.7 * sin(audio_state.y + TAU * lfo_freq * t);
        let amp_final = amp * amp_mod * edge_gain;

        visible_amp += amp * edge_gain;
        left  += osc * amp_final * (1.0 - norm_x);
        right += osc * amp_final * norm_x;
    }

    // ── Normalize and write output ──────────────────────────────────────────
    // Use sqrt(visible_amp) since N random-phase oscillators produce
    // RMS amplitude proportional to sqrt(N), not N.

    var norm = 1.0;
    if visible_amp > 0.0 {
        norm = 1.0 / sqrt(visible_amp);
    }

    audio_out[sample_idx] = vec2<f32>(left, right) * norm * volume;
}
