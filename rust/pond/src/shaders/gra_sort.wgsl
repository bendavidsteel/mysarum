// Sort GRA node positions into spatially-binned order using counting sort
// Concatenated after common.wgsl

@group(0) @binding(0) var<storage, read> gra_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> gra_sorted_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bin_offset: array<u32>;
@group(0) @binding(3) var<storage, read_write> bin_size: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn clear_bins(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&bin_size)) {
        return;
    }
    atomicStore(&bin_size[id.x], 0u);
}

@compute @workgroup_size(256)
fn sort_nodes(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_gra_nodes) {
        return;
    }
    let node = gra_pos[id.x];
    // Only sort active nodes
    if (node.w < 0.5) {
        return;
    }
    let binIndex = get_bin_index(
        node.xy,
        params.world_half,
        params.g_bin_size,
        params.g_num_bins_x,
        params.g_num_bins_y,
    );
    let slot = atomicAdd(&bin_size[binIndex], 1u);
    let newIndex = bin_offset[binIndex] + slot;
    gra_sorted_pos[newIndex] = node;
}
