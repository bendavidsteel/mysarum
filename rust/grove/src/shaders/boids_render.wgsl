// Instanced billboard points for the boids. Additive blend.

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv:    vec2<f32>,
    @location(1) alpha: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32,
           @builtin(instance_index) inst: u32) -> VsOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = quad[vid];
    let p = particles[inst];

    var clip = u.view_proj * vec4<f32>(p.pos.xyz, 1.0);
    let size = u.render_params.w;       // NDC half-size
    let aspect = u.activity.w;
    clip.x += corner.x * size * clip.w / aspect;
    clip.y += corner.y * size * clip.w;

    var out: VsOut;
    out.clip = clip;
    out.uv = corner;
    out.alpha = p.color.a;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    if (d > 1.0) { discard; }
    let glow = pow(1.0 - d, 2.0);
    let a = clamp(in.alpha, 0.0, 1.0) * glow * 0.5;
    return vec4<f32>(vec3<f32>(0.0, 0.9, 0.9) * a, a);
}
