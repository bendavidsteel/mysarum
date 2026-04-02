// Background shader — visualises ocean current strength as dark blue
// Concatenated after common.wgsl — all types + current functions available.

@group(0) @binding(0) var<uniform> uniforms: RenderUniforms;

struct BgVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> BgVertexOutput {
    // Fullscreen triangle (3 verts cover the screen)
    var pos = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0),
    );
    var out: BgVertexOutput;
    out.clip_position = vec4(pos[vi], 0.0, 1.0);

    // Map clip space → world space using the render viewport
    let ndc = pos[vi];
    let wx = mix(uniforms.min_x, uniforms.max_x, (ndc.x + 1.0) * 0.5);
    let wy = mix(uniforms.min_y, uniforms.max_y, (ndc.y + 1.0) * 0.5);
    out.world_pos = vec2(wx, wy);

    return out;
}

@fragment
fn fs_main(in: BgVertexOutput) -> @location(0) vec4<f32> {
    let mag = current_magnitude_at_pos(
        in.world_pos, uniforms.world_half, uniforms.time, uniforms.current_strength
    );

    // Normalise: current_strength is the peak force, magnitude can exceed it slightly
    // due to noise, so we clamp after dividing
    let norm = clamp(mag / max(uniforms.current_strength, 0.001), 0.0, 1.0);

    // Black → dark blue
    let blue = vec3(0.0, 0.02, 0.08) * norm;

    return vec4(blue, 1.0);
}
