@group(0) @binding(0) var<storage, read> vertex_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sorted_idx: array<u32>;
@group(0) @binding(2) var<storage, read> bin_offset: array<u32>;
@group(0) @binding(3) var<storage, read_write> bin_counter: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn clear_counters(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&bin_counter) {
        return;
    }
    atomicStore(&bin_counter[id.x], 0u);
}

@compute @workgroup_size(64)
fn sort_vertices(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices {
        return;
    }
    let pos = vertex_pos[id.x];
    if pos.w < 0.0 { return; }
    let info = get_bin_info(pos.x, pos.y, params);
    let offset = bin_offset[info.bin_index] + atomicAdd(&bin_counter[info.bin_index], 1u);
    sorted_idx[offset] = id.x;
}
