// Shared declarations prefixed to every grove shader (compute + render).
// Contains only struct definitions and pure helpers — no bindings, no entry
// points — so it can be concatenated ahead of any specialised shader.

const PI:  f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// ── Uniforms shared by the render pipelines ─────────────────────────────────
struct Uniforms {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos:       vec4<f32>,   // xyz + pad
    world_dim:     vec4<f32>,   // sim world W,H,D + pad
    vol_min:       vec4<f32>,   // physarum volume world-space min + pad
    vol_max:       vec4<f32>,   // physarum volume world-space max + pad
    vol_res:       vec4<f32>,   // trail texel resolution x,y,z + pad
    misc:          vec4<f32>,   // time, wind_strength, wind_x, wind_y
    activity:      vec4<f32>,   // boid, physarum, tree, aspect
    render_params: vec4<f32>,   // raycast_quality, density, threshold, point_size
}

// ── Uniforms shared by the compute pipelines ────────────────────────────────
struct ComputeParams {
    vol_res:   vec4<f32>,   // physarum volume res (x,y,z)
    world_res: vec4<f32>,   // boids world res (x,y,z); w = num_agents
    timing:    vec4<f32>,   // time, delta, wind_strength, brightness
    wind:      vec4<f32>,   // wind.x, wind.y, activity, _
    boid_a:    vec4<f32>,   // attraction, attractionMax, alignment, alignmentMax
    boid_b:    vec4<f32>,   // repulsion, repulsionMax, maxSpeed, randomStrength
    boid_c:    vec4<f32>,   // fov, kuramotoStrength, kuramotoMax, _
    phys:      vec4<f32>,   // trailWeight, sensorAngle, sensorOffset, moveSpeed
    phys2:     vec4<f32>,   // turnSpeed, diffuseRate, decayRate, _
    blur_dir:  vec4<f32>,   // blur dir (x,y,z)
}

// ── Simulation data structs (std430 storage layout) ─────────────────────────
struct Particle {
    pos:   vec4<f32>,
    vel:   vec4<f32>,
    attr:  vec4<f32>,   // x: natural freq, y: phase
    color: vec4<f32>,   // rgb + alpha (brightness)
}

struct Agent {
    pos:   vec4<f32>,
    vel:   vec4<f32>,
    attr:  vec4<f32>,   // x: species idx
    state: vec4<f32>,   // x: maxSense, y: avgSense, z: turnAmount
}

struct Segment {
    p0:    vec4<f32>,   // xyz + width at p0
    p1:    vec4<f32>,   // xyz + width at p1
    color: vec4<f32>,
}

// ── Bob Jenkins one-at-a-time hash → [0,1) floats ───────────────────────────
fn hash_u(x: u32) -> u32 {
    var v = x;
    v += v << 10u; v ^= v >> 6u;
    v += v << 3u;  v ^= v >> 11u;
    v += v << 15u;
    return v;
}
fn hash2(v: vec2<u32>) -> u32 { return hash_u(v.x ^ hash_u(v.y)); }
fn hash3(v: vec3<u32>) -> u32 { return hash_u(v.x ^ hash_u(v.y) ^ hash_u(v.z)); }
fn hash4(v: vec4<u32>) -> u32 { return hash_u(v.x ^ hash_u(v.y) ^ hash_u(v.z) ^ hash_u(v.w)); }

fn float_construct(m_in: u32) -> f32 {
    let m = (m_in & 0x007FFFFFu) | 0x3F800000u;
    return bitcast<f32>(m) - 1.0;
}
fn random1(x: f32)      -> f32 { return float_construct(hash_u(bitcast<u32>(x))); }
fn random4(v: vec4<f32>) -> f32 { return float_construct(hash4(bitcast<vec4<u32>>(v))); }

// Rotate `from` a fraction `frac` of the way toward `to`.
fn turn_toward(from_v: vec3<f32>, to_v: vec3<f32>, frac: f32) -> vec3<f32> {
    let dp = dot(from_v, to_v);
    let axis = normalize(cross(from_v, to_v));
    let perp = normalize(cross(axis, from_v));
    var angle = acos(clamp(dp / (length(from_v) * length(to_v)), -1.0, 1.0));
    angle *= frac;
    return from_v * cos(angle) + perp * sin(angle);
}

// ── 3D simplex noise (from node.frag) ───────────────────────────────────────
fn random3(c: vec3<f32>) -> vec3<f32> {
    let j = 4096.0 * sin(dot(c, vec3<f32>(17.0, 59.4, 15.0)));
    var r: vec3<f32>;
    r.z = fract(512.0 * j);
    let j1 = j * 0.125;
    r.x = fract(512.0 * j1);
    let j2 = j1 * 0.125;
    r.y = fract(512.0 * j2);
    return r - 0.5;
}

fn simplex3d(p: vec3<f32>) -> f32 {
    let F3 = 0.3333333;
    let G3 = 0.1666667;
    let s = floor(p + dot(p, vec3<f32>(F3)));
    let x = p - s + dot(s, vec3<f32>(G3));
    let e = step(vec3<f32>(0.0), x - x.yzx);
    let i1 = e * (1.0 - e.zxy);
    let i2 = 1.0 - e.zxy * (1.0 - e);
    let x1 = x - i1 + G3;
    let x2 = x - i2 + 2.0 * G3;
    let x3 = x - 1.0 + 3.0 * G3;
    var w = vec4<f32>(dot(x, x), dot(x1, x1), dot(x2, x2), dot(x3, x3));
    w = max(vec4<f32>(0.6) - w, vec4<f32>(0.0));
    var d = vec4<f32>(
        dot(random3(s),      x),
        dot(random3(s + i1), x1),
        dot(random3(s + i2), x2),
        dot(random3(s + 1.0), x3),
    );
    w = w * w;
    w = w * w;
    d = d * w;
    return dot(d, vec4<f32>(52.0));
}

fn simplex3d_fractal(m: vec3<f32>) -> f32 {
    let rot1 = mat3x3<f32>(-0.37, 0.36, 0.85, -0.14, -0.93, 0.34, 0.92, 0.01, 0.4);
    let rot2 = mat3x3<f32>(-0.55, -0.39, 0.74, 0.33, -0.91, -0.24, 0.77, 0.12, 0.63);
    let rot3 = mat3x3<f32>(-0.71, 0.52, -0.47, -0.08, -0.72, -0.68, -0.7, -0.45, 0.56);
    return 0.5333333 * simplex3d(m * rot1)
         + 0.2666667 * simplex3d(2.0 * m * rot2)
         + 0.1333333 * simplex3d(4.0 * m * rot3)
         + 0.0666667 * simplex3d(8.0 * m);
}
