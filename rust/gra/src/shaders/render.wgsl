// Shared bindings for both node and line rendering
@group(0) @binding(0) var<storage, read> node_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> node_state: array<f32>;
@group(0) @binding(2) var<storage, read> connections: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: RenderUniforms;

struct RenderUniforms {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    node_radius: f32,
    num_nodes: u32,
    num_connections: u32,
    window_aspect: f32,
}

fn world_to_clip(wx: f32, wy: f32) -> vec2<f32> {
    let range_x = uniforms.max_x - uniforms.min_x;
    let range_y = uniforms.max_y - uniforms.min_y;
    let cx = 2.0 * (wx - uniforms.min_x) / range_x - 1.0;
    let cy = 2.0 * (wy - uniforms.min_y) / range_y - 1.0;
    return vec2(cx, cy);
}

// ── Node rendering (instanced quads with bloom) ────────────────────────────

struct NodeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) state: f32,
}

@vertex
fn vs_node(
    @location(0) quad_pos: vec2<f32>,
    @location(1) quad_uv: vec2<f32>,
    @builtin(instance_index) instance: u32,
) -> NodeVertexOutput {
    var out: NodeVertexOutput;

    if instance >= uniforms.num_nodes {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        out.uv = vec2(0.0);
        out.state = 0.0;
        return out;
    }

    let pos = node_pos[instance];
    if pos.w < 0.0 {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        out.uv = vec2(0.0);
        out.state = 0.0;
        return out;
    }

    let center = world_to_clip(pos.x, pos.y);
    let range_x = uniforms.max_x - uniforms.min_x;
    let range_y = uniforms.max_y - uniforms.min_y;
    // Use uniform world-space radius, then correct for aspect ratio
    let world_size = uniforms.node_radius * 2.0 * 8.0;
    let size_x = world_size / range_x;
    let size_y = world_size / range_x * uniforms.window_aspect;

    out.clip_position = vec4(center + quad_pos * vec2(size_x, size_y), 0.0, 1.0);
    out.uv = quad_uv;
    out.state = node_state[instance];
    return out;
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - abs(h6 % 2.0 - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    if h6 < 1.0 { rgb = vec3(c, x, 0.0); }
    else if h6 < 2.0 { rgb = vec3(x, c, 0.0); }
    else if h6 < 3.0 { rgb = vec3(0.0, c, x); }
    else if h6 < 4.0 { rgb = vec3(0.0, x, c); }
    else if h6 < 5.0 { rgb = vec3(x, 0.0, c); }
    else { rgb = vec3(c, 0.0, x); }
    return rgb + vec3(m);
}

// Bloom pass (additive blending)
@fragment
fn fs_node(in: NodeVertexOutput) -> @location(0) vec4<f32> {
    let centered = in.uv - vec2(0.5);
    let dist = length(centered) * 2.0;
    if dist > 1.0 { discard; }

    // Color from state (hue 180-250 degrees)
    let hue = mix(180.0, 250.0, in.state) / 360.0;
    let rgb = hsv_to_rgb(hue, 1.0, 1.0);

    // Soft bloom falloff
    let glow = pow(1.0 - dist, 2.0) * 0.6;
    return vec4(rgb * glow, glow);
}

// Opaque core pass (standard alpha blending, drawn on top)
@vertex
fn vs_core(
    @location(0) quad_pos: vec2<f32>,
    @location(1) quad_uv: vec2<f32>,
    @builtin(instance_index) instance: u32,
) -> NodeVertexOutput {
    var out: NodeVertexOutput;

    if instance >= uniforms.num_nodes {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        out.uv = vec2(0.0);
        out.state = 0.0;
        return out;
    }

    let pos = node_pos[instance];
    if pos.w < 0.0 {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        out.uv = vec2(0.0);
        out.state = 0.0;
        return out;
    }

    let center = world_to_clip(pos.x, pos.y);
    let range_x = uniforms.max_x - uniforms.min_x;
    // Core is smaller than bloom quad
    let world_size = uniforms.node_radius * 2.0;
    let size_x = world_size / range_x;
    let size_y = world_size / range_x * uniforms.window_aspect;

    out.clip_position = vec4(center + quad_pos * vec2(size_x, size_y), 0.0, 1.0);
    out.uv = quad_uv;
    out.state = node_state[instance];
    return out;
}

@fragment
fn fs_core(in: NodeVertexOutput) -> @location(0) vec4<f32> {
    let centered = in.uv - vec2(0.5);
    let dist = length(centered) * 2.0;

    // Anti-aliased circle
    let alpha = smoothstep(1.0, 0.85, dist);
    if alpha < 0.001 { discard; }

    // Color from state (hue 180-250 degrees), brighter for core
    let hue = mix(180.0, 250.0, in.state) / 360.0;
    let rgb = hsv_to_rgb(hue, 0.7, 1.0);

    return vec4(rgb, alpha);
}

// ── Line rendering (connection edges) ──────────────────────────────────────

struct LineVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_line(@builtin(vertex_index) vertex_index: u32) -> LineVertexOutput {
    var out: LineVertexOutput;

    let conn_idx = vertex_index / 2u;
    let end_idx = vertex_index % 2u;

    if conn_idx >= uniforms.num_connections {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let from_idx = connections[conn_idx * 2u];
    let to_idx = connections[conn_idx * 2u + 1u];
    let node_idx = select(to_idx, from_idx, end_idx == 0u);
    let pos = node_pos[node_idx];

    out.clip_position = vec4(world_to_clip(pos.x, pos.y), 0.0, 1.0);
    return out;
}

@fragment
fn fs_line(in: LineVertexOutput) -> @location(0) vec4<f32> {
    return vec4(0.0, 0.5, 0.64, 1.0);
}
