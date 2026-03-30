// GRA node spatial hash: clear bins and count active nodes per bin
// Concatenated after common.wgsl

@group(0) @binding(0) var<storage, read> gra_pos: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: SimParams;
@group(2) @binding(0) var<storage, read_write> bin_size: array<atomic<u32>>;

@compute @workgroup_size(256)
fn clear_bins(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&bin_size)) {
        return;
    }
    atomicStore(&bin_size[id.x], 0u);
}

@compute @workgroup_size(256)
fn fill_bins(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_gra_nodes) {
        return;
    }
    let node = gra_pos[id.x];
    // Only count active nodes
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
    // +1 offset: bin_size[0] stays 0 so prefix sum produces correct offsets
    atomicAdd(&bin_size[binIndex + 1u], 1u);
}
