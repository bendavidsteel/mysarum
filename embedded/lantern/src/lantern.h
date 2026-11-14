// The lantern's animation programs, shared by the firmware (main.cpp) and the browser simulator
// (sim/). Include after something that provides Arduino's millis() and TWO_PI, esp_random(), and
// FastLED's CRGB, CHSV and dim8_video: the real headers on the ESP32, sim/host.h in the browser.
#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <utility>

#include "colormaps.h"
#include "lantern_graph.h"
#include "params.h"

constexpr int OUTPUTS = 2;
constexpr int LEDS = 45;  // per output
constexpr int MAX_COMETS = 4;
constexpr int MAX_RECOVERY = 30;
static_assert(GRAPH_NODES == OUTPUTS * LEDS, "re-run tools/export_graph.py for this LED count");

struct Params {
#define X(type, name, value, lo, hi) type name = value;
    LANTERN_PARAMS(X)
#undef X
};
inline Params params;

struct ParamInfo {
    const char *name;
    bool is_int;
    float lo, hi;
    void *value;
};

inline constexpr bool is_int_type(int *) { return true; }
inline constexpr bool is_int_type(float *) { return false; }

const ParamInfo PARAM_INFO[] = {
#define X(type, name, value, lo, hi) {#name, is_int_type((type *)nullptr), lo, hi, &params.name},
    LANTERN_PARAMS(X)
#undef X
};
constexpr int PARAM_COUNT = sizeof(PARAM_INFO) / sizeof(PARAM_INFO[0]);

inline float get_param(int i) {
    const ParamInfo &p = PARAM_INFO[i];
    return p.is_int ? *(int *)p.value : *(float *)p.value;
}

inline void set_param(int i, float value) {
    const ParamInfo &p = PARAM_INFO[i];
    value = fminf(fmaxf(value, p.lo), p.hi);
    if (p.is_int) *(int *)p.value = lroundf(value);
    else *(float *)p.value = value;
}

// Returns false for an unknown name.
inline bool set_param(const char *name, float value) {
    for (int i = 0; i < PARAM_COUNT; i++)
        if (!strcmp(PARAM_INFO[i].name, name)) return set_param(i, value), true;
    return false;
}

inline float frand() { return (float)esp_random() / 4294967296.0f; }
inline float frand(float lo, float hi) { return lo + (hi - lo) * frand(); }
// Whole number from lo to hi inclusive.
inline int irand(int lo, int hi) { return lo + (int)(frand() * (hi - lo + 1)); }

struct Comet {
    float pos, vel;
    CRGB color;
    bool alive;
};

// The "mood" of a strip; re-rolled every so often so the behaviour keeps changing.
struct Mood {
    float decay, sparkle_rate, comet_rate, hue_spread, hue_wander, saturation;
    uint32_t frames_left;

    static Mood roll() {
        const Params &p = params;
        return {frand(p.mood_decay_lo, p.mood_decay_hi),
                frand(p.mood_sparkle_lo, p.mood_sparkle_hi),
                frand(p.mood_comet_lo, p.mood_comet_hi),
                frand(p.mood_spread_lo, p.mood_spread_hi),
                frand(p.mood_wander_lo, p.mood_wander_hi),
                frand(p.mood_saturation_lo, p.mood_saturation_hi),
                (uint32_t)frand(p.mood_frames_lo, p.mood_frames_hi)};
    }
};

struct Strip {
    float px[LEDS][3] = {};
    Comet comets[MAX_COMETS] = {};
    float hue = frand();
    Mood mood = Mood::roll();

    CRGB color() {
        float h = hue + frand(-mood.hue_spread, mood.hue_spread);
        h -= floorf(h);
        return CHSV(h * 255, mood.saturation * 255, 255);
    }

    void add(int i, CRGB c, float w) {
        if (i < 0 || i >= LEDS) return;
        for (int k = 0; k < 3; k++) px[i][k] = fminf(px[i][k] + c[k] / 255.0f * w, 1.0f);
    }

    void step() {
        if (mood.frames_left == 0) mood = Mood::roll();
        mood.frames_left--;
        hue += frand(-1, 1) * mood.hue_wander;

        for (auto &p : px)
            for (auto &v : p) v *= mood.decay;

        if (frand() < mood.sparkle_rate) add(frand() * LEDS, color(), frand(0.4, 1.0));

        if (frand() < mood.comet_rate) {
            for (auto &c : comets) {
                if (c.alive) continue;
                bool forward = frand() < 0.5;
                float speed = frand(params.comet_speed_lo, params.comet_speed_hi);
                c = {forward ? 0.0f : LEDS - 1.0f, speed * (forward ? 1 : -1), color(), true};
                break;
            }
        }

        for (auto &c : comets) {
            if (!c.alive) continue;
            c.pos += c.vel;
            if (c.pos < 0 || c.pos > LEDS - 1) {
                c.alive = false;
                continue;
            }
            // Split between the two neighbouring pixels so slow comets glide.
            int i = (int)c.pos;
            float f = c.pos - i;
            add(i, c.color, 1 - f);
            add(i + 1, c.color, f);
        }
    }

    void render(CRGB *out) {
        for (int i = 0; i < LEDS; i++)
            out[i] = CRGB(dim8_video(px[i][0] * 255), dim8_video(px[i][1] * 255), dim8_video(px[i][2] * 255));
    }
};

// Turns per-LED values from any of the automata into colour. Each LED moves from the value it was
// showing to the newest one at a constant rate, is blended with its neighbours on the graph, and
// is looked up in a colour map, or taken directly as red, green and blue. Wrapping values (a
// place in a cycle) live in [0, 1), move the short way round, and dim where neighbours disagree,
// as at the core of a spiral.
struct Look {
    int map;
    bool wrap;
    float blend;       // weight of the neighbours, relative to the LED itself
    bool rgb = false;  // three values per LED in [0, 1], shown as colour without a map
};

struct Display {
    float from[GRAPH_NODES][3] = {}, to[GRAPH_NODES][3] = {}, shown[GRAPH_NODES][3] = {};
    uint32_t start = 0, duration = 1;
    Look look = {TWILIGHT, false, 0};

    void push(const float *values, uint32_t duration_ms, Look l) {
        float rows[GRAPH_NODES][3] = {};
        for (int i = 0; i < GRAPH_NODES; i++) rows[i][0] = values[i];
        push(rows, duration_ms, l);
    }

    void push(const float values[][3], uint32_t duration_ms, Look l) {
        memcpy(from, shown, sizeof(from));
        memcpy(to, values, sizeof(to));
        start = millis(), duration = duration_ms > 1 ? duration_ms : 1, look = l;
    }

    void render(CRGB out[OUTPUTS][LEDS]) {
        const int channels = look.rgb ? 3 : 1;
        const bool wrap = look.wrap && !look.rgb;
        float progress = fminf((float)(millis() - start) / duration, 1);
        float x[GRAPH_NODES], y[GRAPH_NODES];
        for (int i = 0; i < GRAPH_NODES; i++) {
            for (int k = 0; k < channels; k++) {
                float d = to[i][k] - from[i][k];
                if (wrap) d -= roundf(d);
                shown[i][k] = from[i][k] + d * progress;
            }
            if (wrap) {
                shown[i][0] -= floorf(shown[i][0]);
                x[i] = cosf(TWO_PI * shown[i][0]), y[i] = sinf(TWO_PI * shown[i][0]);
            }
        }
        int map_index = params.map_override >= 0 ? params.map_override : look.map;
        const Colormap &map = COLORMAP[map_index < COLORMAPS ? map_index : 0];
        for (int i = 0; i < GRAPH_NODES; i++) {
            float v[3] = {shown[i][0], shown[i][1], shown[i][2]}, level = 1;
            if (look.blend > 0) {
                float sx = wrap ? x[i] : 0, sy = wrap ? y[i] : 0, total = 1;
                for (int e = GRAPH_OFFSETS[i]; e < GRAPH_OFFSETS[i + 1]; e++) {
                    int n = GRAPH_NEIGHBOURS[e];
                    float w = look.blend * GRAPH_WEIGHTS[e];
                    if (wrap) sx += w * x[n], sy += w * y[n];
                    else
                        for (int k = 0; k < channels; k++) v[k] += w * shown[n][k];
                    total += w;
                }
                if (wrap) {
                    v[0] = atan2f(sy, sx) / TWO_PI;
                    level = 0.3f + 0.7f * sqrtf(sx * sx + sy * sy) / total;
                } else {
                    for (int k = 0; k < channels; k++) v[k] /= total;
                }
            }
            CRGB c;
            if (look.rgb)
                for (int k = 0; k < 3; k++) c[k] = dim8_video(255 * fminf(fmaxf(v[k], 0), 1));
            else
                c = lookup(map, v[0]);
            if (level < 1) c.nscale8_video(255 * level);
            if (params.single_channel) c = c.b > c.r ? CRGB(0, 0, c.b) : CRGB(c.r, 0, 0);
            out[i / LEDS][i % LEDS] = c;
        }
    }

    static CRGB lookup(const Colormap &map, float t) {
        float f = map.cyclic ? (t - floorf(t)) * COLORMAP_STOPS : fminf(fmaxf(t, 0), 1) * (COLORMAP_STOPS - 1);
        int last = map.cyclic ? COLORMAP_STOPS - 1 : COLORMAP_STOPS - 2;
        int k = (int)f < last ? (int)f : last;
        f -= k;
        const uint8_t *a = map.stops[k], *b = map.stops[(k + 1) % COLORMAP_STOPS];
        return CRGB(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]));
    }
};

// Greenberg-Hastings excitable medium on the mapped neighbour graph: a resting LED fires when the
// distance-weighted share of its firing neighbours passes a threshold, then recovers for a few
// steps. Rare spontaneous firings seed new waves; the rule parameters drift randomly over time.
// Shown as 1 when firing, falling away through recovery to 0 at rest.
struct Excitable {
    uint8_t state[GRAPH_NODES] = {};  // 0 resting, 1 firing, 2..recovery recovering
    float threshold, spontaneous;
    uint32_t step_ms, steps_left = 0, last_step = 0;

    static Look look() { return {params.ex_map, false, params.ex_blend}; }

    void reroll() {
        const Params &p = params;
        threshold = frand(p.ex_threshold_lo, p.ex_threshold_hi);
        spontaneous = frand(p.ex_spontaneous_lo, p.ex_spontaneous_hi);
        step_ms = frand(p.ex_step_ms_lo, p.ex_step_ms_hi);
        steps_left = frand(p.ex_steps_lo, p.ex_steps_hi);
    }

    void step() {
        if (steps_left == 0) reroll();
        steps_left--;
        const int recovery = params.ex_recovery;
        uint8_t next[GRAPH_NODES];
        for (int i = 0; i < GRAPH_NODES; i++) {
            if (state[i] == 0) {
                float firing = 0, total = 0;
                for (int e = GRAPH_OFFSETS[i]; e < GRAPH_OFFSETS[i + 1]; e++) {
                    total += GRAPH_WEIGHTS[e];
                    if (state[GRAPH_NEIGHBOURS[e]] == 1) firing += GRAPH_WEIGHTS[e];
                }
                next[i] = (firing >= threshold * total || frand() < spontaneous) ? 1 : 0;
            } else {
                next[i] = state[i] >= recovery ? 0 : state[i] + 1;
            }
        }
        memcpy(state, next, sizeof(state));
    }

    void update(Display &display) {
        if (steps_left == 0) reroll();
        if (millis() - last_step < step_ms) return;
        last_step = millis();
        step();
        const int recovery = params.ex_recovery;
        float values[GRAPH_NODES];
        for (int i = 0; i < GRAPH_NODES; i++)
            values[i] = state[i] == 1   ? 1.0f
                        : state[i] >= 2 ? fmaxf(0.6f * (recovery - state[i]) / (recovery - 2), 0)
                                        : 0.0f;
        display.push(values, step_ms, look());
    }
};

// Gray-Scott reaction-diffusion on a sphere mesh fitted to the lantern, read out at each LED as a
// smooth blend of nearby vertices. Drifts between a chaotic regime and one of wandering spots,
// and reseeds gradually if the pattern dies out.
struct GrayScott {
    float buf[4][SPHERE_VERTS];
    float *u = buf[0], *v = buf[1], *next_u = buf[2], *next_v = buf[3];
    float regime = 0, target_regime = 0;  // 0 chaotic, 1 spots
    float sample[GRAPH_NODES];
    float range_lo = 0, range_hi = 0.35f;
    uint32_t last_reseed = 0;
    // RAM copies of the mesh tables, which are slow to read from flash.
    uint16_t offsets[SPHERE_VERTS + 1], neighbours[SPHERE_EDGES];
    float inv_degree[SPHERE_VERTS];

    static Look look() { return {params.gs_map, false, params.gs_blend}; }

    GrayScott() {
        memcpy(offsets, SPHERE_OFFSETS, sizeof(offsets));
        memcpy(neighbours, SPHERE_NEIGHBOURS, sizeof(neighbours));
        for (int i = 0; i < SPHERE_VERTS; i++) {
            u[i] = 1, v[i] = 0;
            inv_degree[i] = 1.0f / (offsets[i + 1] - offsets[i]);
        }
    }

    // Copying would leave u and v pointing into the old buffers.
    GrayScott(const GrayScott &) = delete;
    GrayScott &operator=(const GrayScott &) = delete;

    void drop() {
        const float *c = SPHERE_POS[(int)(frand() * SPHERE_VERTS)];
        for (int i = 0; i < SPHERE_VERTS; i++) {
            const float *p = SPHERE_POS[i];
            if (p[0] * c[0] + p[1] * c[1] + p[2] * c[2] > params.gs_drop_size) u[i] = 0.5f, v[i] = 0.25f;
        }
    }

    void step() {
        const Params &p = params;
        float feed = p.gs_feed_chaos + (p.gs_feed_spots - p.gs_feed_chaos) * regime;
        float kill = p.gs_kill_chaos + (p.gs_kill_spots - p.gs_kill_chaos) * regime;
        for (int i = 0; i < SPHERE_VERTS; i++) {
            float su = 0, sv = 0;
            for (int e = offsets[i]; e < offsets[i + 1]; e++) {
                su += u[neighbours[e]];
                sv += v[neighbours[e]];
            }
            float uvv = u[i] * v[i] * v[i];
            next_u[i] = u[i] + p.gs_du * (su * inv_degree[i] - u[i]) - uvv + feed * (1 - u[i]);
            next_v[i] = v[i] + p.gs_dv * (sv * inv_degree[i] - v[i]) + uvv - (feed + kill) * v[i];
        }
        std::swap(u, next_u);
        std::swap(v, next_v);
    }

    void update(Display &display) {
        if (frand() < params.gs_regime_change) target_regime = frand();
        regime += (target_regime - regime) * params.gs_regime_rate;
        float peak = 0;
        for (int i = 0; i < SPHERE_VERTS; i++) peak = fmaxf(peak, v[i]);
        // Once the pattern has died, reseed one spot at a time so it regrows from a point.
        if (peak < 0.02f && millis() - last_reseed > (uint32_t)params.gs_reseed_ms) {
            drop();
            last_reseed = millis();
        } else if (frand() < params.gs_drop_chance) {
            drop();
        }
        for (int s = 0; s < params.gs_steps_per_frame; s++) step();

        float lo = 1e9f, hi = -1e9f;
        for (int i = 0; i < GRAPH_NODES; i++) {
            float x = 0;
            for (int e = SAMPLE_OFFSETS[i]; e < SAMPLE_OFFSETS[i + 1]; e++) x += SAMPLE_WEIGHTS[e] * v[SAMPLE_VERTS[e]];
            sample[i] = x;
            lo = fminf(lo, x), hi = fmaxf(hi, x);
        }
        // Stretch to the current range of the pattern, tracked slowly so the overall level doesn't flicker.
        range_lo += (lo - range_lo) * 0.01f;
        range_hi += (fmaxf(hi, range_lo + params.gs_contrast) - range_hi) * 0.01f;
        for (int i = 0; i < GRAPH_NODES; i++) {
            float t = fminf(fmaxf((sample[i] - range_lo) / (range_hi - range_lo), 0), 1);
            sample[i] = t * t * (3 - 2 * t);
        }
        display.push(sample, 50, look());
    }
};

// Cyclic cellular automaton on the mapped neighbour graph: a cell advances to the next of a few
// cyclic states when enough of its neighbours (by distance weight) are already in that state,
// which settles into waves and spirals, shown as each LED's place in the cycle. Small random
// patches are rescrambled now and then, and whenever the pattern freezes.
struct Cyclic {
    static constexpr int FROZEN_STEPS = 20;
    uint8_t state[GRAPH_NODES], states = 5;
    float threshold;
    uint32_t step_ms, steps_left = 0, last_step = 0, frozen = 0;

    static Look look() { return {params.cy_map, true, params.cy_blend}; }

    Cyclic() {
        for (auto &s : state) s = frand() * states;
    }

    void reroll() {
        const Params &p = params;
        states = irand(p.cy_states_lo, p.cy_states_hi);
        for (auto &s : state) s %= states;
        threshold = frand(p.cy_threshold_lo, p.cy_threshold_hi);
        step_ms = frand(p.cy_step_ms_lo, p.cy_step_ms_hi);
        steps_left = frand(p.cy_steps_lo, p.cy_steps_hi);
    }

    void scramble() {
        int c = frand() * GRAPH_NODES;
        state[c] = frand() * states;
        for (int e = GRAPH_OFFSETS[c]; e < GRAPH_OFFSETS[c + 1]; e++) state[GRAPH_NEIGHBOURS[e]] = frand() * states;
    }

    void step() {
        if (steps_left == 0) reroll();
        steps_left--;
        uint8_t next[GRAPH_NODES];
        int advanced = 0;
        for (int i = 0; i < GRAPH_NODES; i++) {
            uint8_t succ = (state[i] + 1) % states;
            float ahead = 0, total = 0;
            for (int e = GRAPH_OFFSETS[i]; e < GRAPH_OFFSETS[i + 1]; e++) {
                total += GRAPH_WEIGHTS[e];
                if (state[GRAPH_NEIGHBOURS[e]] == succ) ahead += GRAPH_WEIGHTS[e];
            }
            bool advance = ahead > 0 && ahead >= threshold * total;
            next[i] = advance ? succ : state[i];
            advanced += advance;
        }
        memcpy(state, next, sizeof(state));
        frozen = advanced ? 0 : frozen + 1;
        if (frozen % FROZEN_STEPS == 1 || frand() < params.cy_scramble) scramble();
    }

    void update(Display &display) {
        if (steps_left == 0) reroll();
        if (millis() - last_step < step_ms) return;
        last_step = millis();
        step();
        float values[GRAPH_NODES];
        for (int i = 0; i < GRAPH_NODES; i++) values[i] = (float)state[i] / states;
        display.push(values, step_ms, look());
    }
};

// Neural cellular automaton in the style of muNCA: each LED holds a few channels and updates them
// with one linear layer over isotropic features of its neighbourhood (its own state, the graph
// Laplacian and the size of the local gradient, plus absolute values), so no direction on the
// lantern is favoured. Updates pass through tanh with a slight pull towards zero, which keeps
// the channels off their limits. The weights take a Metropolis walk: every few seconds a small
// Ornstein-Uhlenbeck step (nca_step of their spread) is proposed, run, and kept or undone by how
// its fitness compares with the last kept weights. Fitness is the energy of the shown channels in a middle band of spatial
// and of temporal frequencies, so contrast at a few LEDs' scale and motion over fractions of a
// second count, while speckle, flicker, flatness and stillness don't. When the pattern goes
// flat, a small patch is scrambled; if it goes flat again soon after, the weights take one
// unscored step, so they never jump.
// Shown either as channel 0 on a colour map, or as channels 0-2 directly as red, green and blue,
// so the colours evolve with the weights.
struct Nca {
    static constexpr int C = 8, F = 5 * C;
    static constexpr uint32_t FLAT_STEPS = 25, RESEED_MS = 2000, TOOK_STEPS = 150;
    float state[GRAPH_NODES][C];
    float w[C][F], b[C], kept[C][F];
    float fitness_sum = 0, kept_fitness = NAN;  // NAN until the current weights have been scored
    float fast[GRAPH_NODES][3] = {}, slow[GRAPH_NODES][3] = {};  // moving averages of the shown channels
    float edge_dir[GRAPH_EDGES][3];  // unit direction of each edge in the tangent plane at its source
    uint32_t last_step = 0, last_seed = 0, flat_steps = 0, seed_steps = 0, eval_steps = 0;

    static Look look() { return {params.nca_map, false, params.nca_blend, params.nca_rgb != 0}; }
    // Standard deviation of the weights: the gain over sqrt(F).
    static float w_std() { return params.nca_gain / sqrtf(F); }

    Nca() {
        for (int i = 0; i < GRAPH_NODES; i++) {
            const float *p = GRAPH_POS[i];
            float r = sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            for (int e = GRAPH_OFFSETS[i]; e < GRAPH_OFFSETS[i + 1]; e++) {
                const float *q = GRAPH_POS[GRAPH_NEIGHBOURS[e]];
                float d[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
                float radial = (d[0] * p[0] + d[1] * p[1] + d[2] * p[2]) / (r * r), len = 0;
                for (int k = 0; k < 3; k++) d[k] -= radial * p[k], len += d[k] * d[k];
                for (int k = 0; k < 3; k++) edge_dir[e][k] = d[k] / sqrtf(len);
            }
            for (auto &x : state[i]) x = frand(-0.5, 0.5);
        }
        redraw();
    }

    static float gauss() { return sqrtf(-2 * logf(1 - frand())) * cosf(TWO_PI * frand()); }

    void redraw() {
        for (auto &row : w)
            for (auto &x : row) x = w_std() * gauss();
        for (auto &x : b) x = 0.02f * gauss();
        kept_fitness = NAN, fitness_sum = 0, eval_steps = 0;
    }

    // Keeps or undoes the weights just scored, then proposes the next step from the kept ones.
    void evaluate() {
        float fitness = logf(fitness_sum / eval_steps + 1e-6f);
        fitness_sum = 0, eval_steps = 0;
        if (isnan(kept_fitness) || frand() < expf(fminf(0, (fitness - kept_fitness) / params.nca_temperature))) {
            memcpy(kept, w, sizeof(w));
            kept_fitness = fitness;
        } else {
            memcpy(w, kept, sizeof(w));
        }
        propose();
    }

    // Moves each weight by nca_step of their spread; the pull towards zero keeps that spread at w_std().
    void propose() {
        const float step = params.nca_step, k = 0.5f * step * step;
        for (auto &row : w)
            for (auto &x : row) x += -k * x + step * w_std() * gauss();
    }

    // Leaves weights that keep dying after one small step, which is kept whatever it scores.
    void nudge() {
        propose();
        kept_fitness = NAN, fitness_sum = 0, eval_steps = 0;
    }

    void scramble() {
        int c = frand() * GRAPH_NODES;
        for (auto &x : state[c]) x = frand(-1, 1);
        for (int e = GRAPH_OFFSETS[c]; e < GRAPH_OFFSETS[c + 1]; e++)
            for (auto &x : state[GRAPH_NEIGHBOURS[e]]) x = frand(-1, 1);
    }

    // Replaces each LED's values with the weighted mean of itself and its neighbours.
    static void blur(const float in[][3], float out[][3]) {
        for (int i = 0; i < GRAPH_NODES; i++) {
            float total = 1;
            for (int k = 0; k < 3; k++) out[i][k] = in[i][k];
            for (int e = GRAPH_OFFSETS[i]; e < GRAPH_OFFSETS[i + 1]; e++) {
                total += GRAPH_WEIGHTS[e];
                for (int k = 0; k < 3; k++) out[i][k] += GRAPH_WEIGHTS[e] * in[GRAPH_NEIGHBOURS[e]][k];
            }
            for (int k = 0; k < 3; k++) out[i][k] /= total;
        }
    }

    // Variance of the shown channels in a middle band of spatial frequencies, as a difference of
    // blurs: one blur removes speckle, and subtracting three removes large blobs.
    float space_band() {
        static float s[GRAPH_NODES][3], once[GRAPH_NODES][3], a[GRAPH_NODES][3], b[GRAPH_NODES][3];
        for (int i = 0; i < GRAPH_NODES; i++)
            for (int k = 0; k < 3; k++) s[i][k] = state[i][k];
        blur(s, once);
        blur(once, a);
        blur(a, b);
        float energy = 0;
        const int channels = params.nca_rgb ? 3 : 1;
        for (int k = 0; k < channels; k++) {
            float sum = 0, sq = 0;
            for (int i = 0; i < GRAPH_NODES; i++) {
                float d = once[i][k] - b[i][k];
                sum += d, sq += d * d;
            }
            float m = sum / GRAPH_NODES;
            energy += fmaxf(sq / GRAPH_NODES - m * m, 0) / channels;
        }
        return energy;
    }

    // Mean square of the shown channels in a middle band of temporal frequencies, as a difference
    // of moving averages: the fast one removes step-to-step flicker, the slow one stillness.
    float time_band() {
        float energy = 0;
        const int channels = params.nca_rgb ? 3 : 1;
        for (int i = 0; i < GRAPH_NODES; i++)
            for (int k = 0; k < channels; k++) {
                fast[i][k] += (state[i][k] - fast[i][k]) * params.nca_fast;
                slow[i][k] += (state[i][k] - slow[i][k]) * params.nca_slow;
                float d = fast[i][k] - slow[i][k];
                energy += d * d;
            }
        return energy / (GRAPH_NODES * channels);
    }

    void step() {
        seed_steps++;
        const float leak = params.nca_leak, odds = params.nca_update_odds;
        float next[GRAPH_NODES][C];
        for (int i = 0; i < GRAPH_NODES; i++) {
            const float *s = state[i];
            float mean[C] = {}, grad[C][3] = {}, total = 0;
            for (int e = GRAPH_OFFSETS[i]; e < GRAPH_OFFSETS[i + 1]; e++) {
                const float *n = state[GRAPH_NEIGHBOURS[e]], we = GRAPH_WEIGHTS[e];
                total += we;
                for (int c = 0; c < C; c++) {
                    mean[c] += we * n[c];
                    for (int k = 0; k < 3; k++) grad[c][k] += we * (n[c] - s[c]) * edge_dir[e][k];
                }
            }
            memcpy(next[i], s, sizeof(next[i]));
            // Cells update asynchronously, each with these odds per step.
            if (frand() >= odds) continue;
            float f[F];
            for (int c = 0; c < C; c++) {
                float lap = mean[c] / total - s[c];
                float g = sqrtf(grad[c][0] * grad[c][0] + grad[c][1] * grad[c][1] + grad[c][2] * grad[c][2]) / total;
                f[c] = s[c], f[C + c] = lap, f[2 * C + c] = g, f[3 * C + c] = fabsf(s[c]), f[4 * C + c] = fabsf(lap);
            }
            for (int c = 0; c < C; c++) {
                float d = b[c] - leak * s[c];
                for (int k = 0; k < F; k++) d += w[c][k] * f[k];
                next[i][c] = tanhf(s[c] + d);
            }
        }
        memcpy(state, next, sizeof(state));

        // Mean over channels of the spread across LEDs, for spotting a flat pattern.
        float spread = 0;
        for (int c = 0; c < C; c++) {
            float sum = 0, sq = 0;
            for (auto &cell : state) sum += cell[c], sq += cell[c] * cell[c];
            float m = sum / GRAPH_NODES;
            spread += sqrtf(fmaxf(sq / GRAPH_NODES - m * m, 0)) / C;
        }
        fitness_sum += params.nca_space_weight * space_band() + params.nca_time_weight * time_band();
        if (++eval_steps >= (uint32_t)params.nca_eval_steps) evaluate();
        flat_steps = spread < params.nca_flat ? flat_steps + 1 : 0;
        if (flat_steps >= FLAT_STEPS && millis() - last_seed > RESEED_MS) {
            if (seed_steps < TOOK_STEPS) nudge();
            scramble();
            last_seed = millis(), seed_steps = 0;
        }
    }

    void update(Display &display) {
        const uint32_t step_ms = params.nca_step_ms;
        if (millis() - last_step < step_ms) return;
        last_step = millis();
        step();
        if (params.nca_rgb) {
            float values[GRAPH_NODES][3];
            for (int i = 0; i < GRAPH_NODES; i++)
                for (int k = 0; k < 3; k++) values[i][k] = 0.5f * (state[i][k] + 1);
            display.push(values, step_ms, look());
        } else {
            float values[GRAPH_NODES];
            for (int i = 0; i < GRAPH_NODES; i++) values[i] = 0.5f * (state[i][0] + 1);
            display.push(values, step_ms, look());
        }
    }
};

// Every program and the shared display. Each pass of the main loop runs step() then render(),
// which writes the frame before brightness and the power limit.
struct Lantern {
    Strip strips[OUTPUTS];
    Excitable excitable;
    GrayScott gray_scott;
    Cyclic cyclic;
    Nca nca;
    Display display;

    void step() {
        switch (params.program) {
            case 1: excitable.update(display); break;
            case 2: gray_scott.update(display); break;
            case 3: cyclic.update(display); break;
            case 4: nca.update(display); break;
            default:
                for (auto &s : strips) s.step();
        }
    }

    void render(CRGB out[OUTPUTS][LEDS]) {
        if (params.program >= 1 && params.program <= 4) display.render(out);
        else
            for (int o = 0; o < OUTPUTS; o++) strips[o].render(out[o]);
    }
};
