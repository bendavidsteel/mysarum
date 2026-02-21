struct RenderUniforms {
    view_proj: mat4x4<f32>,
    center: vec4<f32>,     // xyz = mesh center, w = scale
    light: vec4<f32>,      // xyz = direction, w = ambient
    render_mode: vec4<f32>, // x = mode (0=lit+wire, 1=normal map), yzw unused
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> states: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: RenderUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) bary: vec3<f32>,
    @location(2) state: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex_id = indices[vertex_index];
    let pos = positions[vertex_id];
    let state = states[vertex_id];

    let centered = vec3<f32>(
        (pos.x - uniforms.center.x) * uniforms.center.w,
        (pos.y - uniforms.center.y) * uniforms.center.w,
        pos.z * uniforms.center.w,
    );

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(centered, 1.0);
    out.world_position = centered;

    // Barycentric coordinates per triangle vertex
    let local_idx = vertex_index % 3u;
    var bary = vec3<f32>(0.0);
    if local_idx == 0u { bary.x = 1.0; }
    else if local_idx == 1u { bary.y = 1.0; }
    else { bary.z = 1.0; }
    out.bary = bary;
    out.state = state;

    return out;
}

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h, h, h) + k) * 6.0 - 3.0);
    return v * mix(vec3<f32>(1.0), clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(s));
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    // Face normal from screen-space derivatives of world position
    let ddx_pos = dpdx(in.world_position);
    let ddy_pos = dpdy(in.world_position);
    var normal = normalize(cross(ddx_pos, ddy_pos));
    if !front_facing {
        normal = -normal;
    }

    let bary = in.bary;
    let state = in.state;
    let render_mode = u32(uniforms.render_mode.x);

    var final_color: vec3<f32>;

    if render_mode == 1u {
        // Normal map visualization
        final_color = normal * 0.5 + 0.5;
    } else {
        // Default: lit + state color + wireframe
        let hue = mix(180.0 / 360.0, 250.0 / 360.0, state);
        let base_color = hsv2rgb(hue, 0.8, 0.9);

        let light_dir = normalize(uniforms.light.xyz);
        let n_dot_l = abs(dot(normal, light_dir));
        let ambient = uniforms.light.w;
        let brightness = ambient + (1.0 - ambient) * n_dot_l;

        let lit_color = base_color * brightness;

        // Wireframe
        let edge_width = 1.5;
        let d = min(bary.x, min(bary.y, bary.z));
        let edge_fw = fwidth(d);
        let edge = 1.0 - smoothstep(0.0, edge_fw * edge_width, d);

        let wire_color = vec3<f32>(1.0, 1.0, 1.0);
        final_color = mix(lit_color, wire_color, edge * 0.6);
    }

    return vec4<f32>(final_color, 1.0);
}
