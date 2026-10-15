// Drives the LED strips on a Wroomy board with an animation program, and accepts serial commands
// so a host can light individual LEDs for mapping.
//
// Serial commands, one per line:
//   c                clear the mapping framebuffer
//   s OUT IDX R G B  set one pixel
//   w                show the framebuffer, reply "ok N" (N = lit pixels)
//   a                go back to the animation
//   p N              switch animation program (0 sparkles, 1 excitable medium,
//                    2 reaction-diffusion, 3 cyclic automaton, 4 neural automaton)
//   m N              colour map for programs 1-4 (0 twilight, 1 magma, 2 inferno, 3 viridis,
//                    4 ember; -1 each program's own)
//   r N              neural automaton shows channels 0-2 as RGB (1, the default) or channel 0
//                    on a colour map (0)
// c, s and w switch from the animation into mapping mode.

#include <Arduino.h>
#include <FastLED.h>

#include "colormaps.h"
#include "lantern_graph.h"

constexpr int OUTPUTS = 2;
constexpr int LEDS = 45;  // per output
constexpr uint8_t BRIGHTNESS = 255;
constexpr int MAX_COMETS = 4;
// Show only the strongest colour channel of each pixel, for strips without their 12V supply.
constexpr bool SINGLE_CHANNEL = false;
// FastLED's per-channel current model is close enough for these 12V strips; stays under a 1A supply.
constexpr uint32_t MAX_MILLIAMPS = 900;
static_assert(GRAPH_NODES == OUTPUTS * LEDS, "re-run tools/export_graph.py for this LED count");

CRGB leds[OUTPUTS][LEDS];

float frand() { return (float)esp_random() / 4294967296.0f; }
float frand(float lo, float hi) { return lo + (hi - lo) * frand(); }

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
        return {frand(0.80, 0.98), frand(0.0, 0.6),   frand(0.0, 0.04), frand(0.02, 0.5),
                frand(0.0005, 0.006), frand(0.6, 1.0), (uint32_t)frand(600, 2000)};
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
                c = {forward ? 0.0f : LEDS - 1.0f, frand(0.1, 0.8) * (forward ? 1 : -1), color(), true};
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
    int map_override = -1;  // set over serial to try a map on any program that uses one

    void push(const float *values, uint32_t duration_ms, Look l) {
        float rows[GRAPH_NODES][3] = {};
        for (int i = 0; i < GRAPH_NODES; i++) rows[i][0] = values[i];
        push(rows, duration_ms, l);
    }

    void push(const float values[][3], uint32_t duration_ms, Look l) {
        memcpy(from, shown, sizeof(from));
        memcpy(to, values, sizeof(to));
        start = millis(), duration = max(duration_ms, (uint32_t)1), look = l;
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
        const Colormap &map = COLORMAP[map_override >= 0 ? map_override : look.map];
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
            if (SINGLE_CHANNEL) c = c.b > c.r ? CRGB(0, 0, c.b) : CRGB(c.r, 0, 0);
            out[i / LEDS][i % LEDS] = c;
        }
    }

    static CRGB lookup(const Colormap &map, float t) {
        float f = map.cyclic ? (t - floorf(t)) * COLORMAP_STOPS : fminf(fmaxf(t, 0), 1) * (COLORMAP_STOPS - 1);
        int k = min((int)f, map.cyclic ? COLORMAP_STOPS - 1 : COLORMAP_STOPS - 2);
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
    static constexpr int RECOVERY = 10;
    static constexpr Look LOOK = {INFERNO, false, 0.3f};
    uint8_t state[GRAPH_NODES] = {};  // 0 resting, 1 firing, 2..RECOVERY recovering
    float threshold, spontaneous;
    uint32_t step_ms, steps_left = 0, last_step = 0;

    void reroll() {
        threshold = frand(0.1, 0.3);
        spontaneous = frand(0.0005, 0.006);
        step_ms = frand(60, 160);
        steps_left = frand(300, 900);
    }

    void step() {
        if (steps_left == 0) reroll();
        steps_left--;
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
                next[i] = state[i] >= RECOVERY ? 0 : state[i] + 1;
            }
        }
        memcpy(state, next, sizeof(state));
    }

    void update(Display &display) {
        if (millis() - last_step < step_ms) return;
        last_step = millis();
        step();
        float values[GRAPH_NODES];
        for (int i = 0; i < GRAPH_NODES; i++)
            values[i] = state[i] == 1 ? 1.0f : state[i] >= 2 ? 0.6f * (RECOVERY - state[i]) / (RECOVERY - 2) : 0.0f;
        display.push(values, step_ms, LOOK);
    }
};

// Gray-Scott reaction-diffusion on a sphere mesh fitted to the lantern, read out at each LED as a
// smooth blend of nearby vertices. Drifts between a chaotic regime and one of wandering spots,
// and reseeds gradually if the pattern dies out.
struct GrayScott {
    static constexpr float DU = 0.2f, DV = 0.1f;
    static constexpr int STEPS_PER_FRAME = 1;
    static constexpr uint32_t RESEED_MS = 3000;
    static constexpr Look LOOK = {EMBER, false, 0};
    float buf[4][SPHERE_VERTS];
    float *u = buf[0], *v = buf[1], *next_u = buf[2], *next_v = buf[3];
    float regime = 0, target_regime = 0;  // 0 chaotic, 1 spots
    float sample[GRAPH_NODES];
    float range_lo = 0, range_hi = 0.35f;
    uint32_t last_reseed = 0;
    // RAM copies of the mesh tables, which are slow to read from flash.
    uint16_t offsets[SPHERE_VERTS + 1], neighbours[SPHERE_EDGES];
    float inv_degree[SPHERE_VERTS];

    GrayScott() {
        memcpy(offsets, SPHERE_OFFSETS, sizeof(offsets));
        memcpy(neighbours, SPHERE_NEIGHBOURS, sizeof(neighbours));
        for (int i = 0; i < SPHERE_VERTS; i++) {
            u[i] = 1, v[i] = 0;
            inv_degree[i] = 1.0f / (offsets[i + 1] - offsets[i]);
        }
    }

    void drop() {
        const float *c = SPHERE_POS[(int)(frand() * SPHERE_VERTS)];
        for (int i = 0; i < SPHERE_VERTS; i++) {
            const float *p = SPHERE_POS[i];
            if (p[0] * c[0] + p[1] * c[1] + p[2] * c[2] > 0.9f) u[i] = 0.5f, v[i] = 0.25f;
        }
    }

    void step() {
        float feed = 0.018f + 0.007f * regime, kill = 0.051f + 0.004f * regime;
        for (int i = 0; i < SPHERE_VERTS; i++) {
            float su = 0, sv = 0;
            for (int e = offsets[i]; e < offsets[i + 1]; e++) {
                su += u[neighbours[e]];
                sv += v[neighbours[e]];
            }
            float uvv = u[i] * v[i] * v[i];
            next_u[i] = u[i] + DU * (su * inv_degree[i] - u[i]) - uvv + feed * (1 - u[i]);
            next_v[i] = v[i] + DV * (sv * inv_degree[i] - v[i]) + uvv - (feed + kill) * v[i];
        }
        std::swap(u, next_u);
        std::swap(v, next_v);
    }

    void update(Display &display) {
        if (frand() < 0.0003f) target_regime = frand();
        regime += (target_regime - regime) * 0.0005f;
        float peak = 0;
        for (int i = 0; i < SPHERE_VERTS; i++) peak = fmaxf(peak, v[i]);
        // Once the pattern has died, reseed one spot at a time so it regrows from a point.
        if (peak < 0.02f && millis() - last_reseed > RESEED_MS) {
            drop();
            last_reseed = millis();
        } else if (frand() < 0.0005f) {
            drop();
        }
        for (int s = 0; s < STEPS_PER_FRAME; s++) step();

        float lo = 1e9f, hi = -1e9f;
        for (int i = 0; i < GRAPH_NODES; i++) {
            float x = 0;
            for (int e = SAMPLE_OFFSETS[i]; e < SAMPLE_OFFSETS[i + 1]; e++) x += SAMPLE_WEIGHTS[e] * v[SAMPLE_VERTS[e]];
            sample[i] = x;
            lo = fminf(lo, x), hi = fmaxf(hi, x);
        }
        // Stretch to the current range of the pattern, tracked slowly so the overall level doesn't flicker.
        range_lo += (lo - range_lo) * 0.01f;
        range_hi += (fmaxf(hi, range_lo + 0.15f) - range_hi) * 0.01f;
        for (int i = 0; i < GRAPH_NODES; i++) {
            float t = fminf(fmaxf((sample[i] - range_lo) / (range_hi - range_lo), 0), 1);
            sample[i] = t * t * (3 - 2 * t);
        }
        display.push(sample, 50, LOOK);
    }
};

// Cyclic cellular automaton on the mapped neighbour graph: a cell advances to the next of a few
// cyclic states when enough of its neighbours (by distance weight) are already in that state,
// which settles into waves and spirals, shown as each LED's place in the cycle. Small random
// patches are rescrambled now and then, and whenever the pattern freezes.
struct Cyclic {
    static constexpr int FROZEN_STEPS = 20;
    static constexpr Look LOOK = {TWILIGHT, true, 0.5f};
    uint8_t state[GRAPH_NODES], states = 5;
    float threshold;
    uint32_t step_ms, steps_left = 0, last_step = 0, frozen = 0;

    Cyclic() {
        for (auto &s : state) s = frand() * states;
    }

    void reroll() {
        states = 4 + frand() * 3;
        for (auto &s : state) s %= states;
        threshold = frand(0.0, 0.08);
        step_ms = frand(80, 200);
        steps_left = frand(400, 1200);
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
        if (frozen % FROZEN_STEPS == 1 || frand() < 0.003f) scramble();
    }

    void update(Display &display) {
        if (millis() - last_step < step_ms) return;
        last_step = millis();
        step();
        float values[GRAPH_NODES];
        for (int i = 0; i < GRAPH_NODES; i++) values[i] = (float)state[i] / states;
        display.push(values, step_ms, LOOK);
    }
};

// Neural cellular automaton in the style of muNCA: each LED holds a few channels and updates them
// with one linear layer over isotropic features of its neighbourhood (its own state, the graph
// Laplacian and the size of the local gradient, plus absolute values), so no direction on the
// lantern is favoured. Updates pass through tanh with a slight pull towards zero, which keeps
// the channels off their limits. The weights take a Metropolis walk: every few seconds a small
// Ornstein-Uhlenbeck step is proposed, run, and kept or undone by how its fitness (spread across
// the LEDs, movement up to a cap, and structure at a middle scale in the shown channels) compares
// with the last kept weights. When the pattern goes
// flat, a small patch is scrambled; if it goes flat again soon after, the weights are redrawn.
// Shown either as channel 0 on a colour map, or as channels 0-2 directly as red, green and blue,
// so the colours evolve with the weights.
struct Nca {
    static constexpr int C = 8, F = 5 * C;
    // Weights have a gain of 1.6 over sqrt(F).
    static constexpr float W_STD = 1.6f / 6.3246f, LEAK = 0.4f, FLAT = 0.05f, DRIFT_STEPS = 3000;
    static constexpr float MOVE_CAP = 0.03f, MOVE_WEIGHT = 5, BAND_WEIGHT = 1, TEMPERATURE = 0.03f;
    static constexpr uint32_t STEP_MS = 40, FLAT_STEPS = 25, RESEED_MS = 2000, TOOK_STEPS = 150;
    static constexpr uint32_t EVAL_STEPS = 50;
    static constexpr Look LOOK = {VIRIDIS, false, 0.3f}, RGB_LOOK = {VIRIDIS, false, 0.3f, true};
    bool rgb = true;
    float state[GRAPH_NODES][C];
    float w[C][F], b[C], kept[C][F];
    float fitness_sum = 0, kept_fitness = -1;  // -1 until the current weights have been scored
    float edge_dir[GRAPH_EDGES][3];  // unit direction of each edge in the tangent plane at its source
    uint32_t last_step = 0, last_seed = 0, flat_steps = 0, seed_steps = 0, eval_steps = 0;

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
            for (auto &x : row) x = W_STD * gauss();
        for (auto &x : b) x = 0.02f * gauss();
        kept_fitness = -1, fitness_sum = 0, eval_steps = 0;
    }

    // Keeps or undoes the weights just scored, then proposes the next step from the kept ones.
    void evaluate() {
        float fitness = fitness_sum / EVAL_STEPS;
        fitness_sum = 0, eval_steps = 0;
        if (kept_fitness < 0 || frand() < expf(fminf(0, (fitness - kept_fitness) / TEMPERATURE))) {
            memcpy(kept, w, sizeof(w));
            kept_fitness = fitness;
        } else {
            memcpy(w, kept, sizeof(w));
        }
        const float k = (float)EVAL_STEPS / DRIFT_STEPS, noise = sqrtf(2 * k) * W_STD;
        for (auto &row : w)
            for (auto &x : row) x += -k * x + noise * gauss();
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

    // Share of the shown channels' variance in a middle band of spatial frequencies, as a
    // difference of blurs: one blur removes speckle, and subtracting three removes large blobs.
    float band() {
        static float s[GRAPH_NODES][3], once[GRAPH_NODES][3], a[GRAPH_NODES][3], b[GRAPH_NODES][3];
        for (int i = 0; i < GRAPH_NODES; i++)
            for (int k = 0; k < 3; k++) s[i][k] = state[i][k];
        blur(s, once);
        blur(once, a);
        blur(a, b);
        float fraction = 0;
        const int channels = rgb ? 3 : 1;
        for (int k = 0; k < channels; k++) {
            float sum = 0, sq = 0, band_sum = 0, band_sq = 0;
            for (int i = 0; i < GRAPH_NODES; i++) {
                float d = once[i][k] - b[i][k];
                sum += s[i][k], sq += s[i][k] * s[i][k], band_sum += d, band_sq += d * d;
            }
            float m = sum / GRAPH_NODES, band_m = band_sum / GRAPH_NODES;
            float var = sq / GRAPH_NODES - m * m, band_var = band_sq / GRAPH_NODES - band_m * band_m;
            fraction += fmaxf(band_var, 0) / (fmaxf(var, 0) + 1e-4f) / channels;
        }
        return fraction;
    }

    void step() {
        seed_steps++;
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
            // Cells update asynchronously, each with even odds per step.
            if (frand() < 0.5f) continue;
            float f[F];
            for (int c = 0; c < C; c++) {
                float lap = mean[c] / total - s[c];
                float g = sqrtf(grad[c][0] * grad[c][0] + grad[c][1] * grad[c][1] + grad[c][2] * grad[c][2]) / total;
                f[c] = s[c], f[C + c] = lap, f[2 * C + c] = g, f[3 * C + c] = fabsf(s[c]), f[4 * C + c] = fabsf(lap);
            }
            for (int c = 0; c < C; c++) {
                float d = b[c] - LEAK * s[c];
                for (int k = 0; k < F; k++) d += w[c][k] * f[k];
                next[i][c] = tanhf(s[c] + d);
            }
        }
        float move = 0;
        for (int i = 0; i < GRAPH_NODES; i++)
            for (int c = 0; c < C; c++) move += fabsf(next[i][c] - state[i][c]);
        move /= GRAPH_NODES * C;
        memcpy(state, next, sizeof(state));

        // Mean over channels of the spread across LEDs.
        float spread = 0;
        for (int c = 0; c < C; c++) {
            float sum = 0, sq = 0;
            for (auto &cell : state) sum += cell[c], sq += cell[c] * cell[c];
            float m = sum / GRAPH_NODES;
            spread += sqrtf(fmaxf(sq / GRAPH_NODES - m * m, 0)) / C;
        }
        fitness_sum += spread + MOVE_WEIGHT * fminf(move, MOVE_CAP) + BAND_WEIGHT * band();
        if (++eval_steps == EVAL_STEPS) evaluate();
        flat_steps = spread < FLAT ? flat_steps + 1 : 0;
        if (flat_steps >= FLAT_STEPS && millis() - last_seed > RESEED_MS) {
            if (seed_steps < TOOK_STEPS) redraw();
            scramble();
            last_seed = millis(), seed_steps = 0;
        }
    }

    void update(Display &display) {
        if (millis() - last_step < STEP_MS) return;
        last_step = millis();
        step();
        if (rgb) {
            float values[GRAPH_NODES][3];
            for (int i = 0; i < GRAPH_NODES; i++)
                for (int k = 0; k < 3; k++) values[i][k] = 0.5f * (state[i][k] + 1);
            display.push(values, STEP_MS, RGB_LOOK);
        } else {
            float values[GRAPH_NODES];
            for (int i = 0; i < GRAPH_NODES; i++) values[i] = 0.5f * (state[i][0] + 1);
            display.push(values, STEP_MS, LOOK);
        }
    }
};

Strip strips[OUTPUTS];
Excitable excitable;
GrayScott gray_scott;
Cyclic cyclic;
Nca nca;
Display display;
int program = 2;
bool mapping = false;
char line[64];
int line_len = 0;

void handle(char *cmd) {
    int out, idx, r, g, b;
    switch (cmd[0]) {
        case 'c':
            FastLED.clear();
            break;
        case 's':
            if (sscanf(cmd + 1, "%d %d %d %d %d", &out, &idx, &r, &g, &b) != 5) return;
            if (out < 0 || out >= OUTPUTS || idx < 0 || idx >= LEDS) return;
            leds[out][idx] = CRGB(r, g, b);
            break;
        case 'w': {
            FastLED.setBrightness(255);
            FastLED.show();
            int lit = 0;
            for (auto &strip : leds)
                for (auto &p : strip) lit += (bool)p;
            Serial.printf("ok %d\n", lit);
            break;
        }
        case 'a':
            mapping = false;
            return;
        case 'p':
            program = atoi(cmd + 1);
            mapping = false;
            return;
        case 'm':
            display.map_override = constrain(atoi(cmd + 1), -1, COLORMAPS - 1);
            return;
        case 'r':
            nca.rgb = atoi(cmd + 1);
            return;
        default:
            return;
    }
    mapping = true;
}

void setup() {
    Serial.setRxBufferSize(4096);
    Serial.begin(115200);
    FastLED.addLeds<WS2813, 23, RGB>(leds[0], LEDS);
    FastLED.addLeds<WS2813, 19, RGB>(leds[1], LEDS);
    FastLED.setMaxPowerInVoltsAndMilliamps(5, MAX_MILLIAMPS);
    FastLED.clear(true);
    uint32_t t = micros();
    for (int i = 0; i < 100; i++) gray_scott.step();
    Serial.printf("lantern: %d x %d leds, reaction-diffusion %lu us/step\n", OUTPUTS, LEDS, (micros() - t) / 100);
}

void loop() {
    while (Serial.available()) {
        char ch = Serial.read();
        if (ch != '\n') {
            if (line_len < (int)sizeof(line) - 1) line[line_len++] = ch;
            continue;
        }
        line[line_len] = 0;
        line_len = 0;
        handle(line);
    }
    if (mapping) return;

    if (program >= 1 && program <= 4) {
        if (program == 1) excitable.update(display);
        if (program == 2) gray_scott.update(display);
        if (program == 3) cyclic.update(display);
        if (program == 4) nca.update(display);
        display.render(leds);
    } else {
        for (int o = 0; o < OUTPUTS; o++) {
            strips[o].step();
            strips[o].render(leds[o]);
        }
    }
    FastLED.setBrightness(BRIGHTNESS);
    FastLED.show();
    delay(10);
}
