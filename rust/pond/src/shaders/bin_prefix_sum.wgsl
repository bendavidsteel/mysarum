// Generic prefix sum (Hillis-Steele scan with blocked loads)
// Each workgroup processes 256 * BLOCK_SIZE = 1024 elements.

const BLOCK_SIZE: u32 = 4u;

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;

var<workgroup> partial_sums: array<u32, 256>;

@compute @workgroup_size(256)
fn prefix_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = arrayLength(&source);
    let workgroup_offset = wid.x * 256u * BLOCK_SIZE;
    let base = workgroup_offset + lid.x * BLOCK_SIZE;

    // Phase 1: Each thread loads BLOCK_SIZE elements and computes a local prefix sum
    var local_data: array<u32, 4>;
    for (var i = 0u; i < BLOCK_SIZE; i = i + 1u) {
        let idx = base + i;
        if (idx < n) {
            local_data[i] = source[idx];
        } else {
            local_data[i] = 0u;
        }
    }
    // Local (inclusive) prefix sum
    for (var i = 1u; i < BLOCK_SIZE; i = i + 1u) {
        local_data[i] = local_data[i] + local_data[i - 1u];
    }

    // Store this thread's total into shared memory
    partial_sums[lid.x] = local_data[BLOCK_SIZE - 1u];
    workgroupBarrier();

    // Phase 2: Hillis-Steele inclusive scan on the 256 partial sums
    var stride = 1u;
    loop {
        if (stride >= 256u) {
            break;
        }
        var val = 0u;
        if (lid.x >= stride) {
            val = partial_sums[lid.x - stride];
        }
        workgroupBarrier();
        if (lid.x >= stride) {
            partial_sums[lid.x] = partial_sums[lid.x] + val;
        }
        workgroupBarrier();
        stride = stride * 2u;
    }

    // Phase 3: Add prefix from previous threads in this workgroup,
    // plus carry from the previous workgroup (source[workgroup_offset - 1] via destination)
    var block_prefix = 0u;
    if (lid.x > 0u) {
        block_prefix = partial_sums[lid.x - 1u];
    }

    // Carry from previous workgroups: the last element of the previous workgroup's output
    var carry = 0u;
    if (workgroup_offset > 0u && base == workgroup_offset && lid.x == 0u) {
        // We rely on sequential workgroup execution per dispatch;
        // for large arrays the host dispatches multiple passes.
    }

    for (var i = 0u; i < BLOCK_SIZE; i = i + 1u) {
        let idx = base + i;
        if (idx < n) {
            destination[idx] = local_data[i] + block_prefix;
        }
    }
}
