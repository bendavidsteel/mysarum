struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    volume: f32,
    current_x: vec2<f32>,  // Current viewport x (min, max)
    current_y: vec2<f32>,  // Current viewport y (min, max)
    map_x0: f32,
    map_y0: f32,
    bin_size: f32,
    num_bins_x: u32,
    num_bins_y: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> audio_out: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: AudioParams;
@group(0) @binding(3) var<storage, read> bin_offset: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.chunk_size {
        return;
    }

    let sample_idx = id.x;
    let t = f32(sample_idx) / params.sample_rate;

    var left = 0.0;
    var right = 0.0;
    var visible_count = 0u;

    // Viewport bounds
    let view_left = params.current_x.x;
    let view_right = params.current_x.y;
    let view_bottom = params.current_y.x;
    let view_top = params.current_y.y;
    let view_width = view_right - view_left;
    let view_height = view_top - view_bottom;

    // Calculate which bins overlap the viewport
    let min_bin_x = u32(max(0.0, (view_left - params.map_x0) / params.bin_size));
    let max_bin_x = min(params.num_bins_x - 1u, u32((view_right - params.map_x0) / params.bin_size));
    let min_bin_y = u32(max(0.0, (view_bottom - params.map_y0) / params.bin_size));
    let max_bin_y = min(params.num_bins_y - 1u, u32((view_top - params.map_y0) / params.bin_size));

    // Iterate over bins that overlap the viewport
    for (var bin_y = min_bin_y; bin_y <= max_bin_y; bin_y++) {
        for (var bin_x = min_bin_x; bin_x <= max_bin_x; bin_x++) {
            let bin_idx = bin_y * params.num_bins_x + bin_x;
            let start_idx = bin_offset[bin_idx];
            let end_idx = bin_offset[bin_idx + 1];

            // Process particles in this bin
            for (var i = start_idx; i < end_idx; i++) {
                let p = particles[i];

                // Double-check particle is in viewport (bin may extend beyond viewport edge)
                if (p.pos.x < view_left || p.pos.x > view_right ||
                    p.pos.y < view_bottom || p.pos.y > view_top) {
                    continue;
                }

                visible_count += 1u;

                // Normalize position within viewport (0 to 1)
                let norm_x = (p.pos.x - view_left) / view_width;
                let norm_y = (p.pos.y - view_bottom) / view_height;

                // Compute waveform
                let phase = compute_phase(p, t);
                let osc = compute_oscillator(phase);
                let amp = compute_amplitude(p);

                // Stereo pan based on x position within viewport
                left += osc * amp * (1.0 - norm_x);
                right += osc * amp * norm_x;
            }
        }
    }

    // add compression
    let compression_threshold = 0.8;

    // Normalize by visible particle count and apply volume
    if (visible_count > 0u) {
        let norm = 1.0 / log(f32(visible_count) + 1.0);
        audio_out[sample_idx] = vec2<f32>(left, right) * norm * params.volume;
    } else {
        audio_out[sample_idx] = vec2<f32>(0.0, 0.0);
    }
}
