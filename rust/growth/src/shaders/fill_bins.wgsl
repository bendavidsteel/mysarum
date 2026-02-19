// Vertex positions: xyz + w (w > 0 = active)
@group(0) @binding(0) var<storage, read> vertex_pos: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: SimParams;
@group(2) @binding(0) var<storage, read_write> bin_size: array<atomic<u32>>;

@compute @workgroup_size(64)
fn clear_bins(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&bin_size) {
        return;
    }
    atomicStore(&bin_size[id.x], 0u);
}

@compute @workgroup_size(64)
fn fill_bins(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_vertices {
        return;
    }
    let pos = vertex_pos[id.x];
    if pos.w < 0.0 { return; }
    let info = get_bin_info(pos.x, pos.y, params);
    atomicAdd(&bin_size[info.bin_index + 1u], 1u);
}
