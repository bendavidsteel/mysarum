// Branch segments rendered as analytic rounded-cone (capsule) impostors.
//
// Each segment emits a small camera-facing quad that bounds the capsule
// p0..p1 (radii = widths). The fragment shader casts the view ray, intersects
// the true rounded cone, writes correct depth, and shades it. Because adjacent
// segments share endpoints + matching widths (see Trees::segments), the
// depth-correct capsules overlap into one seamless, continuous tube — no joint
// gaps, rounded caps. Material is graded trunk(brown)->twig(pale) by radius.

@group(0) @binding(0) var<storage, read> segments: array<Segment>;
@group(0) @binding(1) var<uniform> u: Uniforms;

const MIN_R:    f32 = 0.3;   // floor on branch radius
const QUAD_PAD: f32 = 1.4;   // grow the bounding quad so the capsule never clips

struct VsOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0)                 world: vec3<f32>,
    @location(1) @interpolate(flat) pa:    vec3<f32>,
    @location(2) @interpolate(flat) pb:    vec3<f32>,
    @location(3) @interpolate(flat) radii: vec2<f32>,
}

struct FsOut {
    @location(0)            color: vec4<f32>,
    @builtin(frag_depth)    depth: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32,
           @builtin(instance_index) inst: u32) -> VsOut {
    let s = segments[inst];
    let p0 = s.p0.xyz;
    let p1 = s.p1.xyz;
    let r0 = max(s.p0.w, MIN_R);     // true capsule radii (passed to FS)
    let r1 = max(s.p1.w, MIN_R);
    let q0 = r0 * QUAD_PAD;          // padded radii for the bounding quad
    let q1 = r1 * QUAD_PAD;

    let mid      = (p0 + p1) * 0.5;
    let axis_dir = normalize(p1 - p0 + vec3<f32>(1e-5, 0.0, 0.0));
    let view_dir = normalize(u.cam_pos.xyz - mid);

    var side = cross(axis_dir, view_dir);
    if (length(side) < 1e-4) {
        // viewing nearly down the axis: any vector perpendicular to view works
        side = cross(view_dir, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(1e-4, 0.0, 0.0);
    }
    side = normalize(side);

    // extend ends along the axis by the radius so the round caps are covered
    let e0 = p0 - axis_dir * q0;
    let e1 = p1 + axis_dir * q1;

    var idx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let k = idx[vid];
    var world: vec3<f32>;
    if      (k == 0u) { world = e0 + side * q0; }
    else if (k == 1u) { world = e0 - side * q0; }
    else if (k == 2u) { world = e1 + side * q1; }
    else              { world = e1 - side * q1; }

    var out: VsOut;
    out.clip  = u.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.pa    = p0;
    out.pb    = p1;
    out.radii = vec2<f32>(r0, r1);
    return out;
}

// iq's rounded-cone signed distance (https://iquilezles.org/articles/distfunctions/).
fn sd_round_cone(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r1: f32, r2: f32) -> f32 {
    let ba  = b - a;
    let l2  = dot(ba, ba);
    let rr  = r1 - r2;
    let a2  = l2 - rr * rr;
    let il2 = 1.0 / l2;
    let pa  = p - a;
    let y   = dot(pa, ba);
    let z   = y - l2;
    let xp  = pa * l2 - ba * y;
    let x2  = dot(xp, xp);
    let y2  = y * y * l2;
    let z2  = z * z * l2;
    let k   = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) { return sqrt(x2 + z2) * il2 - r2; }
    if (sign(y) * a2 * y2 < k) { return sqrt(x2 + y2) * il2 - r1; }
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

fn cone_normal(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
    let h = 0.002 * max(max(r1, r2), 0.1);
    let kx = vec3<f32>( 1.0, -1.0, -1.0);
    let ky = vec3<f32>(-1.0, -1.0,  1.0);
    let kz = vec3<f32>(-1.0,  1.0, -1.0);
    let kw = vec3<f32>( 1.0,  1.0,  1.0);
    return normalize(
        kx * sd_round_cone(p + kx * h, a, b, r1, r2) +
        ky * sd_round_cone(p + ky * h, a, b, r1, r2) +
        kz * sd_round_cone(p + kz * h, a, b, r1, r2) +
        kw * sd_round_cone(p + kw * h, a, b, r1, r2));
}

// iq's rounded-cone ray intersection. Returns distance along rd, or -1 on miss.
fn cone_intersect(ro: vec3<f32>, rd: vec3<f32>,
                  pa: vec3<f32>, pb: vec3<f32>, ra: f32, rb: f32) -> f32 {
    let ba = pb - pa;
    let oa = ro - pa;
    let ob = ro - pb;
    let rr = ra - rb;
    let m0 = dot(ba, ba);
    let m1 = dot(ba, oa);
    let m2 = dot(ba, rd);
    let m3 = dot(rd, oa);
    let m5 = dot(oa, oa);
    let m6 = dot(ob, rd);
    let m7 = dot(ob, ob);

    let d2 = m0 - rr * rr;
    let k2 = d2 - m2 * m2;
    let k1 = d2 * m3 - m1 * m2 + m2 * rr * ra;
    let k0 = d2 * m5 - m1 * m1 + m1 * rr * ra * 2.0 - m0 * ra * ra;

    let h = k1 * k1 - k0 * k2;
    if (h >= 0.0 && abs(k2) > 1e-6) {
        let t = (-sqrt(h) - k1) / k2;
        let y = m1 - ra * rr + t * m2;
        if (y > 0.0 && y < d2) { return t; }
    }

    // spherical caps
    let h1 = m3 * m3 - m5 + ra * ra;
    let h2 = m6 * m6 - m7 + rb * rb;
    if (max(h1, h2) < 0.0) { return -1.0; }

    var r = 1e20;
    if (h1 > 0.0) { r = -m3 - sqrt(h1); }
    if (h2 > 0.0) {
        let t = -m6 - sqrt(h2);
        if (t < r) { r = t; }
    }
    return r;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let ro = u.cam_pos.xyz;
    let rd = normalize(in.world - ro);
    let r0 = in.radii.x;
    let r1 = in.radii.y;

    let t = cone_intersect(ro, rd, in.pa, in.pb, r0, r1);
    if (t <= 0.0) { discard; }

    let hit    = ro + rd * t;
    let normal = cone_normal(hit, in.pa, in.pb, r0, r1);

    // ── correct depth so capsules occlude one another / the scene ────────────
    let clip = u.view_proj * vec4<f32>(hit, 1.0);

    // ── material: grade trunk(brown) -> twig(pale) by local radius ───────────
    let ba   = in.pb - in.pa;
    let seg  = clamp(dot(hit - in.pa, ba) / max(dot(ba, ba), 1e-5), 0.0, 1.0);
    let r    = mix(r0, r1, seg);
    let tt   = clamp((r - MIN_R) / 3.0, 0.0, 1.0);
    let trunk_col = vec3<f32>(0.30, 0.20, 0.12);
    let twig_col  = vec3<f32>(0.52, 0.58, 0.42);
    var base = mix(twig_col, trunk_col, tt);

    // subtle bark texture from the shared simplex field
    var sp = hit / u.world_dim.xyz;
    sp.z += u.misc.x * 0.1;
    let n = simplex3d_fractal(sp * 8.0 + 8.0);
    base *= 0.82 + 0.18 * n;

    // ── lighting: lambert + ambient + view rim ───────────────────────────────
    let light = normalize(vec3<f32>(0.4, 0.9, 0.35));
    let ndl   = max(dot(normal, light), 0.0);
    let amb   = 0.35;
    let rim   = pow(1.0 - max(dot(normal, -rd), 0.0), 2.0) * 0.35;
    let col   = base * (amb + 0.75 * ndl) + rim * vec3<f32>(0.55, 0.65, 0.55);

    var out: FsOut;
    out.color = vec4<f32>(col, 1.0);
    out.depth = clip.z / clip.w;
    return out;
}
