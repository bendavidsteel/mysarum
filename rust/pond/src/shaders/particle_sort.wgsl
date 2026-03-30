// Sort particles into spatially-binned order using counting sort
// Concatenated after common.wgsl

@group(0) @binding(0) var<storage, read> source: array<Particle>;
@group(0) @binding(1) var<storage, read_write> destination: array<Particle>;
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
fn sort_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_particles) {
        return;
    }
    let p = source[id.x];
    let binIndex = get_bin_index(
        p.pos,
        params.world_half,
        params.p_bin_size,
        params.p_num_bins_x,
        params.p_num_bins_y,
    );
    let slot = atomicAdd(&bin_size[binIndex], 1u);
    let newIndex = bin_offset[binIndex] + slot;
    destination[newIndex] = p;
}
