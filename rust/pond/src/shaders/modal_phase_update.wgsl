// Modal phase update: advance 8 phase oscillators per node after each audio chunk.
// Standalone shader (not concatenated with common.wgsl).

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
    lo: vec4<f32>,
    hi: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> modal_phase: array<vec4<f32>>;  // 2 per node
@group(0) @binding(1) var<uniform> params: ModalAudioParams;
@group(0) @binding(2) var<uniform> freqs: ModalFreqs;

const TAU: f32 = 6.283185307;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.num_gra_nodes { return; }

    let chunk_dur = f32(params.chunk_size) / params.sample_rate;

    // Advance phases for bands 0-3
    var p_lo = modal_phase[id.x * 2u];
    p_lo = (p_lo + TAU * freqs.lo * chunk_dur) % TAU;
    modal_phase[id.x * 2u] = p_lo;

    // Advance phases for bands 4-7
    var p_hi = modal_phase[id.x * 2u + 1u];
    p_hi = (p_hi + TAU * freqs.hi * chunk_dur) % TAU;
    modal_phase[id.x * 2u + 1u] = p_hi;
}
