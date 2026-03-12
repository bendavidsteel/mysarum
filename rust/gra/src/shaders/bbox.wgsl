// 2D bounding box reduction via atomic min/max
// Uses float-to-sortable-uint encoding for correct float ordering with integer atomics

@group(0) @binding(0) var<storage, read> node_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> bbox_atomic: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> params: SimParams;

fn float_to_sortable(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
    return bits ^ mask;
}

@compute @workgroup_size(64)
fn bbox_clear(@builtin(global_invocation_id) id: vec3u) {
    if id.x == 0u {
        atomicStore(&bbox_atomic[0], 0xFFFFFFFFu); // min_x: init to max sortable
        atomicStore(&bbox_atomic[1], 0xFFFFFFFFu); // min_y: init to max sortable
        atomicStore(&bbox_atomic[2], 0u);           // max_x: init to min sortable
        atomicStore(&bbox_atomic[3], 0u);           // max_y: init to min sortable
    }
}

@compute @workgroup_size(64)
fn bbox_reduce(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.num_nodes { return; }
    let pos = node_pos[id.x];
    if pos.w < 0.0 { return; }
    let sx = float_to_sortable(pos.x);
    let sy = float_to_sortable(pos.y);
    atomicMin(&bbox_atomic[0], sx);
    atomicMin(&bbox_atomic[1], sy);
    atomicMax(&bbox_atomic[2], sx);
    atomicMax(&bbox_atomic[3], sy);
}
