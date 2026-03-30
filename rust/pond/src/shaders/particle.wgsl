// ── Particle life compute (concatenated after common.wgsl) ──────────────────

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read> particle_bin_offset: array<u32>;

@group(1) @binding(0) var<storage, read> gra_sorted_pos: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> gra_bin_offset: array<u32>;

// ── Hash / random ───────────────────────────────────────────────────────────

fn hash(x: u32) -> u32 {
    var u = x;
    u += (u << 10u);
    u ^= (u >>  6u);
    u += (u <<  3u);
    u ^= (u >> 11u);
    u += (u << 15u);
    return u;
}

fn random(f: f32) -> f32 {
    let mantissaMask = 0x007FFFFFu;
    let one          = 0x3F800000u;
    var h = hash(bitcast<u32>(f));
    h &= mantissaMask;
    h |= one;
    return bitcast<f32>(h) - 1.0;
}

fn random2(v: vec2f) -> f32 { return random(v.x + v.y * 57.0); }
fn random3(v: vec3f) -> f32 { return random(v.x + v.y * 57.0 + v.z * 113.0); }
fn random4(v: vec4f) -> f32 { return random(v.x + v.y * 57.0 + v.z * 113.0 + v.w * 197.0); }

// ── Main compute ────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.num_particles {
        return;
    }

    var particle = particles[id.x];

    let bin_xy = get_bin_xy(
        particle.pos,
        params.world_half,
        params.p_bin_size,
        params.p_num_bins_x,
        params.p_num_bins_y,
    );

    let binXMin = i32(bin_xy.x) - 1;
    let binYMin = i32(bin_xy.y) - 1;
    let binXMax = i32(bin_xy.x) + 1;
    let binYMax = i32(bin_xy.y) + 1;

    var totalForce = vec2f(0.0, 0.0);
    var totalEnergy = 0.0;
    var totalInteraction = vec2f(0.0, 0.0);

    // Random bin offset for species copying (pick one random neighbour bin)
    let copyBinXRand = random3(vec3f(params.time, particle.pos.x, particle.pos.y));
    let copyBinYRand = random4(vec4f(params.time, particle.pos.y, particle.pos.x, particle.vel.x));
    let copyBinXOffset = i32(floor(copyBinXRand * 3.0)) - 1;
    let copyBinYOffset = i32(floor(copyBinYRand * 3.0)) - 1;

    // ── Particle–particle interactions ──────────────────────────────────────

    for (var binX = binXMin; binX <= binXMax; binX += 1) {
        for (var binY = binYMin; binY <= binYMax; binY += 1) {
            let realBinX = u32((binX + i32(params.p_num_bins_x)) % i32(params.p_num_bins_x));
            let realBinY = u32((binY + i32(params.p_num_bins_y)) % i32(params.p_num_bins_y));

            let binIndex = realBinY * params.p_num_bins_x + realBinX;
            let binStart = particle_bin_offset[binIndex];
            let binEnd   = particle_bin_offset[binIndex + 1u];

            for (var j = binStart; j < binEnd; j += 1u) {
                let other = particles[j];

                var r = other.pos - particle.pos;
                r.x = wrap_delta_1d(r.x, params.world_half);
                r.y = wrap_delta_1d(r.y, params.world_half);

                let d = length(r);
                if d > 0.0 && d < params.particle_radius {
                    let n = r / d;

                    let alpha_scalar = dot(particle.alpha, other.species) * params.particle_max_force;
                    let forceRegion = params.particle_radius - params.particle_collision_radius;
                    let r_mid = params.particle_collision_radius + forceRegion * 0.5;

                    var f = 0.0;
                    var e = 0.0;
                    if d < params.particle_collision_radius {
                        let dd = params.particle_collision_radius - d;
                        f = dd * -params.particle_collision_strength;
                        e = 0.5 * (f * dd - alpha_scalar * forceRegion);
                    } else if d < r_mid {
                        let dd = d - params.particle_collision_radius;
                        f = 2.0 * alpha_scalar * dd / forceRegion;
                        e = 0.5 * (f * dd - alpha_scalar * forceRegion);
                    } else {
                        let dm = d - params.particle_radius;
                        f = -2.0 * alpha_scalar * dm / forceRegion;
                        e = 0.5 * f * dm;
                    }

                    totalForce += f * n;
                    totalEnergy += e;
                    totalInteraction += 0.01 * (other.species - particle.species);
                }

                // Species copying: only check in the randomly chosen neighbour bin
                if binX == i32(bin_xy.x) + copyBinXOffset && binY == i32(bin_xy.y) + copyBinYOffset {
                    if d < params.particle_copy_radius {
                        let speciesCosSim = dot(particle.species, other.species)
                            / (length(particle.species) * length(other.species) + EPSILON);
                        if speciesCosSim > params.particle_copy_cos_sim {
                            let p = random3(vec3f(params.time, particle.pos.x, other.pos.x));
                            if p < params.particle_copy_prob {
                                particle.species = other.species;
                                particle.alpha = other.alpha;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Particle–GRA node repulsion ─────────────────────────────────────────

    let gra_bin_xy = get_bin_xy(
        particle.pos,
        params.world_half,
        params.g_bin_size,
        params.g_num_bins_x,
        params.g_num_bins_y,
    );

    let gBinXMin = i32(gra_bin_xy.x) - 1;
    let gBinYMin = i32(gra_bin_xy.y) - 1;
    let gBinXMax = i32(gra_bin_xy.x) + 1;
    let gBinYMax = i32(gra_bin_xy.y) + 1;

    for (var gx = gBinXMin; gx <= gBinXMax; gx += 1) {
        for (var gy = gBinYMin; gy <= gBinYMax; gy += 1) {
            let realGX = u32((gx + i32(params.g_num_bins_x)) % i32(params.g_num_bins_x));
            let realGY = u32((gy + i32(params.g_num_bins_y)) % i32(params.g_num_bins_y));

            let gBinIndex = realGY * params.g_num_bins_x + realGX;
            let gBinStart = gra_bin_offset[gBinIndex];
            let gBinEnd   = gra_bin_offset[gBinIndex + 1u];

            for (var j = gBinStart; j < gBinEnd; j += 1u) {
                let gNode = gra_sorted_pos[j];

                // Skip inactive nodes
                if gNode.w < 0.5 {
                    continue;
                }

                var r = gNode.xy - particle.pos;
                r.x = wrap_delta_1d(r.x, params.world_half);
                r.y = wrap_delta_1d(r.y, params.world_half);

                let d = length(r);
                if d > 0.0 && d < params.gra_repulsion_radius {
                    let n = r / d;
                    let f = (params.gra_repulsion_radius - d) * -params.gra_repulsion_strength;
                    totalForce += f * n;
                }
            }
        }
    }

    // ── Integrate ───────────────────────────────────────────────────────────

    // Friction
    let mu = pow(0.5, params.dt / params.particle_friction);
    particle.vel *= mu;

    // Apply force
    particle.vel += totalForce * params.dt / params.particle_mass;

    // Update position
    particle.pos += particle.vel * params.dt;

    // Toroidal wrap
    particle.pos = wrap_pos(particle.pos, params.world_half);

    // Store accumulated energy & interaction
    particle.energy = totalEnergy;
    particle.interaction = totalInteraction;

    particles[id.x] = particle;
}
