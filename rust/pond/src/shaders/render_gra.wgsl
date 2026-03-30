// GRA node and edge rendering
// Standalone file (not concatenated after common.wgsl)

struct RenderUniforms {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    particle_size: f32,
    gra_node_radius: f32,
    num_particles: u32,
    num_gra_nodes: u32,
    num_gra_connections: u32,
    window_aspect: f32,
    world_half: f32,
    max_speed: f32,
    energy_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> gra_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gra_state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> connections: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: RenderUniforms;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn world_to_clip(wx: f32, wy: f32) -> vec2<f32> {
    let range_x = uniforms.max_x - uniforms.min_x;
    let range_y = uniforms.max_y - uniforms.min_y;
    let cx = 2.0 * (wx - uniforms.min_x) / range_x - 1.0;
    let cy = 2.0 * (wy - uniforms.min_y) / range_y - 1.0;
    return vec2(cx, cy);
}

fn wrap_delta(d: f32) -> f32 {
    let size = uniforms.world_half * 2.0;
    if d > uniforms.world_half { return d - size; }
    if d < -uniforms.world_half { return d + size; }
    return d;
}

// ── EDGE rendering (quad strips with Gaussian falloff) ─────────────────────

struct LineVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) perp: f32,  // -1..+1 across the edge width
}

// 12 vertices per connection: 2 segments × 6 verts (two triangles each).
// Segment 0: from_pos → from_pos + delta  (ghost of to near from)
// Segment 1: to_pos   → to_pos - delta    (ghost of from near to)

@vertex
fn vs_line(@builtin(vertex_index) vertex_index: u32) -> LineVertexOutput {
    var out: LineVertexOutput;

    let conn_idx = vertex_index / 12u;
    let local = vertex_index % 12u;

    if conn_idx >= uniforms.num_gra_connections {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        out.perp = 0.0;
        return out;
    }

    let from_idx = connections[conn_idx * 2u];
    let to_idx = connections[conn_idx * 2u + 1u];
    let from_pos = gra_pos[from_idx].xy;
    let to_pos = gra_pos[to_idx].xy;

    let dx = wrap_delta(to_pos.x - from_pos.x);
    let dy = wrap_delta(to_pos.y - from_pos.y);
    let delta = vec2(dx, dy);

    // Which segment (0 or 1) and which of the 6 verts within the quad
    let seg = local / 6u;
    let quad_vert = local % 6u;

    // Segment endpoints in world space
    var seg_start: vec2<f32>;
    var seg_end: vec2<f32>;
    if seg == 0u {
        seg_start = from_pos;
        seg_end = from_pos + delta;
    } else {
        seg_start = to_pos;
        seg_end = to_pos - delta;
    }

    // Convert to clip space
    let clip_start = world_to_clip(seg_start.x, seg_start.y);
    let clip_end = world_to_clip(seg_end.x, seg_end.y);

    // Edge direction and perpendicular in clip space
    let edge_dir = clip_end - clip_start;
    let edge_len = length(edge_dir);
    let half_width = uniforms.gra_node_radius * 0.4 / (uniforms.max_x - uniforms.min_x) * 2.0;

    if edge_len < 0.00001 {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        out.perp = 0.0;
        return out;
    }

    let dir = edge_dir / edge_len;
    let perp = vec2(-dir.y, dir.x) * half_width;

    // Quad corners: 0=start+perp, 1=start-perp, 2=end+perp, 3=end-perp
    // Two triangles: (0,1,2) and (2,1,3)
    var corner: u32;
    switch quad_vert {
        case 0u: { corner = 0u; }
        case 1u: { corner = 1u; }
        case 2u: { corner = 2u; }
        case 3u: { corner = 2u; }
        case 4u: { corner = 1u; }
        default: { corner = 3u; }
    }

    let along = select(clip_start, clip_end, corner >= 2u);
    let side = select(1.0, -1.0, (corner & 1u) != 0u);
    out.clip_position = vec4(along + perp * side, 0.0, 1.0);
    out.perp = side;
    return out;
}

@fragment
fn fs_line(in: LineVertexOutput) -> @location(0) vec4<f32> {
    let d = abs(in.perp);
    let alpha = exp(-4.0 * d * d);
    let color = vec3(0.0, 0.5, 0.64);
    return vec4(color * alpha, alpha);
}

// ── NODE GLOW (instanced quads, additive blending) ──────────────────────────

struct NodeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec3<f32>,
}

@vertex
fn vs_node(
    @location(0) quad_pos: vec2<f32>,
    @location(1) quad_uv: vec2<f32>,
    @builtin(instance_index) instance: u32,
) -> NodeVertexOutput {
    var out: NodeVertexOutput;

    if instance >= uniforms.num_gra_nodes {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        out.uv = vec2(0.0);
        out.color = vec3(0.0);
        return out;
    }

    let pos = gra_pos[instance];
    if pos.w < 0.0 {
        out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
        out.uv = vec2(0.0);
        out.color = vec3(0.0);
        return out;
    }

    let center = world_to_clip(pos.x, pos.y);
    let range_x = uniforms.max_x - uniforms.min_x;
    let range_y = uniforms.max_y - uniforms.min_y;
    let world_size = uniforms.gra_node_radius * 3.0;
    let size_x = world_size / range_x * 2.0;
    let size_y = world_size / range_y * 2.0;

    out.clip_position = vec4(center + quad_pos * vec2(size_x, size_y), 0.0, 1.0);
    out.uv = quad_uv;
    out.color = gra_state[instance].xyz;
    return out;
}

@fragment
fn fs_node(in: NodeVertexOutput) -> @location(0) vec4<f32> {
    let centered = in.uv - vec2(0.5);
    let dist = length(centered) * 2.0;
    if dist > 1.0 { discard; }

    // Gaussian core + soft glow (matching particle style)
    let core_radius = 0.25;
    let core_falloff = dist / core_radius;
    let core = exp(-8.0 * core_falloff * core_falloff);
    let glow = pow(1.0 - dist, 2.0) * 0.5;
    let brightness = core + glow;

    return vec4(in.color * brightness, brightness);
}

// Core entry points kept as no-op stubs (pipeline references them)

@vertex
fn vs_core(
    @location(0) quad_pos: vec2<f32>,
    @location(1) quad_uv: vec2<f32>,
    @builtin(instance_index) instance: u32,
) -> NodeVertexOutput {
    var out: NodeVertexOutput;
    out.clip_position = vec4(0.0, 0.0, -2.0, 1.0);
    out.uv = vec2(0.0);
    out.color = vec3(0.0);
    return out;
}

@fragment
fn fs_core(in: NodeVertexOutput) -> @location(0) vec4<f32> {
    discard;
}
