// Volumetric raymarch of the physarum trail volume — port of raycast.frag.
// Fullscreen triangle; world rays reconstructed from inv_view_proj; front-to-back
// compositing of the trail volume. Screen-blended over the scene.

@group(0) @binding(0) var trail: texture_3d<f32>;
@group(0) @binding(1) var samp:  sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VsOut;
    out.clip = vec4<f32>(pos[vid], 0.0, 1.0);
    out.ndc = pos[vid];
    return out;
}

fn world_at(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let p = u.inv_view_proj * vec4<f32>(ndc.x, ndc.y, z, 1.0);
    return p.xyz / p.w;
}

// returns vec2(t_near, t_far); t_far < t_near means miss
fn intersect_box(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / rd;
    let t0 = (bmin - ro) * inv;
    let t1 = (bmax - ro) * inv;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let tn = max(max(tmin.x, tmin.y), tmin.z);
    let tf = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(tn, tf);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ro = u.cam_pos.xyz;
    let far = world_at(in.ndc, 1.0);
    let rd = normalize(far - ro);

    let bmin = u.vol_min.xyz;
    let bmax = u.vol_max.xyz;
    let hit = intersect_box(ro, rd, bmin, bmax);
    var tn = max(hit.x, 0.0);
    let tf = hit.y;
    if (tf <= tn) { discard; }

    let quality   = u.render_params.x;
    let density   = u.render_params.y;
    let threshold = u.render_params.z;

    let span = bmax - bmin;
    let length_world = tf - tn;
    let steps = clamp(i32(length_world * quality * 0.1), 1, 256);
    let dt = (tf - tn) / f32(steps);
    let a_scale = density / max(quality, 0.01);

    let species = vec4<f32>(0.35, 0.95, 1.0, 1.0);

    // jitter start to reduce banding
    let jitter = fract(sin(dot(in.clip.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    var t = tn + dt * jitter;

    var col = vec4<f32>(0.0);
    for (var i = 0; i < steps; i = i + 1) {
        let wpos = ro + rd * t;
        let uvw = (wpos - bmin) / span;
        let s = textureSampleLevel(trail, samp, uvw, 0.0);
        let total = s.r + s.g + s.b + s.a;
        if (total > threshold) {
            var tc = s.r * species;
            tc = min(tc, vec4<f32>(1.0));
            let one_minus_a = 1.0 - col.a;
            tc *= a_scale;
            col = vec4<f32>(mix(col.rgb, tc.rgb * tc.a, one_minus_a),
                            col.a + tc.a * one_minus_a);
            if (col.a > 1e-4) { col = vec4<f32>(col.rgb / col.a, col.a); }
            if (col.a >= 1.0) { break; }
        }
        t += dt;
    }

    return vec4<f32>(col.rgb * col.a, col.a);
}
