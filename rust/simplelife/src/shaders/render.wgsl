// Render shader for GPU-side particle rendering
// Uses instancing to draw a quad per particle, fragment shader draws the particle

struct VertexInput {
    @location(0) position: vec2<f32>,  // Quad vertex position (-1 to 1)
    @location(1) uv: vec2<f32>,        // UV coordinates (0 to 1)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) hue: f32,
    @location(2) energy: f32,
}

struct RenderParams {
    map_x: vec2<f32>,
    map_y: vec2<f32>,
    current_x: vec2<f32>,
    current_y: vec2<f32>,
    particle_size: f32,
    num_particles: u32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: RenderParams;

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let particle = particles[instance];

    // Orthographic projection: map world coords to clip space [-1, 1]
    // current_x.x = left, current_x.y = right
    // current_y.x = bottom, current_y.y = top
    let view_width = params.current_x.y - params.current_x.x;
    let view_height = params.current_y.y - params.current_y.x;

    // Transform particle position from world space to normalized device coordinates
    let ndc_x = 2.0 * (particle.pos.x - params.current_x.x) / view_width - 1.0;
    let ndc_y = 2.0 * (particle.pos.y - params.current_y.x) / view_height - 1.0;

    // Scale particle size relative to view (particle_size is in world units)
    let size_ndc_x = params.particle_size * 2.0 / view_width;
    let size_ndc_y = params.particle_size * 2.0 / view_height;

    // Offset vertex by particle position in NDC
    let pos = vec2<f32>(
        ndc_x + vertex.position.x * size_ndc_x,
        ndc_y + vertex.position.y * size_ndc_y
    );

    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vertex.uv;
    out.hue = particle.species.x;
    out.energy = particle.energy;

    return out;
}

// HSL to RGB conversion
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - abs(h6 % 2.0 - 1.0));
    let m = l - c * 0.5;

    var rgb: vec3<f32>;
    if (h6 < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h6 < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h6 < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h6 < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h6 < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + vec3<f32>(m);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance from center of quad (UV is 0-1, center is 0.5)
    let centered_uv = in.uv - vec2<f32>(0.5);
    let dist = length(centered_uv) * 2.0;  // Distance from center, 0 at center, 1 at edge

    // Discard pixels outside the circle
    if (dist > 1.0) {
        discard;
    }

    // Soft edge with glow
    let core_radius = 0.6;
    let core = smoothstep(core_radius, 0.0, dist);
    let glow = smoothstep(1.0, core_radius, dist);

    // Energy affects brightness
    let energy_brightness = 0.4 + min(in.energy, 1.0) * 0.6;

    // Convert HSL to RGB
    let rgb = hsl_to_rgb(in.hue, 0.7, energy_brightness * 0.5);

    // Combine core and glow
    let alpha = core + glow * 0.5;

    return vec4<f32>(rgb * alpha, alpha * 0.9);
}