// Modal phase update: advance 8 phase oscillators per node after each audio chunk.
// Concatenated after common.wgsl — AudioParams is available.

struct ModalFreqs {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> modal_phase: array<vec4<f32>>;  // 2 per node
@group(0) @binding(1) var<uniform> params: AudioParams;
@group(0) @binding(2) var<uniform> freqs: ModalFreqs;
@group(0) @binding(3) var<storage, read> gra_state: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.num_gra_nodes { return; }

    let chunk_dur = f32(params.chunk_size) / params.sample_rate;
    let freq_scale = gra_state[id.x].w;

    // Advance phases for bands 0-3
    var p_lo = modal_phase[id.x * 2u];
    p_lo = (p_lo + TAU * freqs.lo * freq_scale * chunk_dur) % TAU;
    modal_phase[id.x * 2u] = p_lo;

    // Advance phases for bands 4-7
    var p_hi = modal_phase[id.x * 2u + 1u];
    p_hi = (p_hi + TAU * freqs.hi * freq_scale * chunk_dur) % TAU;
    modal_phase[id.x * 2u + 1u] = p_hi;
}
