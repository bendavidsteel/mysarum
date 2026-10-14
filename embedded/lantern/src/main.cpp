// Drives the LED strips on a Wroomy board with an animation program, and accepts serial commands
// so a host can light individual LEDs for mapping.
//
// Serial commands, one per line:
//   c                clear the mapping framebuffer
//   s OUT IDX R G B  set one pixel
//   w                show the framebuffer, reply "ok N" (N = lit pixels)
//   a                go back to the animation
//   p N              switch animation program (0 sparkles, 1 excitable medium,
//                    2 reaction-diffusion)
// c, s and w switch from the animation into mapping mode.

#include <Arduino.h>
#include <FastLED.h>

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

// Greenberg-Hastings excitable medium on the mapped neighbour graph: a resting LED fires when the
// distance-weighted share of its firing neighbours passes a threshold, then recovers for a few
// steps. Rare spontaneous firings seed new waves; the rule parameters drift randomly over time.
struct Excitable {
    static constexpr int RECOVERY = 10;
    uint8_t state[GRAPH_NODES] = {};  // 0 resting, 1 firing, 2..RECOVERY recovering
    float fire_glow[GRAPH_NODES] = {}, rest_glow[GRAPH_NODES] = {};
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

    void render(CRGB out[OUTPUTS][LEDS]) {
        if (millis() - last_step >= step_ms) {
            last_step = millis();
            step();
        }
        for (int i = 0; i < GRAPH_NODES; i++) {
            float fire = state[i] == 1 ? 1.0f : 0.0f;
            float rest = state[i] >= 2 ? 0.5f * (RECOVERY - state[i]) / (RECOVERY - 2) : 0.0f;
            // Light up quickly, fade out slowly.
            fire_glow[i] += (fire - fire_glow[i]) * (fire > fire_glow[i] ? 0.5f : 0.08f);
            rest_glow[i] += (rest - rest_glow[i]) * 0.1f;
            CRGB c(dim8_video(rest_glow[i] * 255), 0, dim8_video(fire_glow[i] * 255));
            if (SINGLE_CHANNEL) c = fire_glow[i] >= rest_glow[i] ? CRGB(0, 0, c.b) : CRGB(c.r, 0, 0);
            out[i / LEDS][i % LEDS] = c;
        }
    }
};

// Gray-Scott reaction-diffusion on a sphere mesh fitted to the lantern, read out at each LED as a
// smooth blend of nearby vertices. Drifts between a chaotic regime and one of wandering spots,
// and reseeds if the pattern dies out.
struct GrayScott {
    static constexpr float DU = 0.2f, DV = 0.1f;
    static constexpr int STEPS_PER_FRAME = 8;
    float buf[4][SPHERE_VERTS];
    float *u = buf[0], *v = buf[1], *next_u = buf[2], *next_v = buf[3];
    float regime = 0, target_regime = 0;  // 0 chaotic, 1 spots
    float glow[GRAPH_NODES] = {};
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
            if (p[0] * c[0] + p[1] * c[1] + p[2] * c[2] > 0.97f) u[i] = 0.5f, v[i] = 0.25f;
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

    void render(CRGB out[OUTPUTS][LEDS]) {
        if (frand() < 0.0003f) target_regime = frand();
        regime += (target_regime - regime) * 0.0005f;
        float peak = 0;
        for (int i = 0; i < SPHERE_VERTS; i++) peak = fmaxf(peak, v[i]);
        if (peak < 0.02f)
            for (int k = 0; k < 6; k++) drop();
        else if (frand() < 0.0005f)
            drop();
        for (int s = 0; s < STEPS_PER_FRAME; s++) step();

        for (int i = 0; i < GRAPH_NODES; i++) {
            float x = 0;
            for (int e = SAMPLE_OFFSETS[i]; e < SAMPLE_OFFSETS[i + 1]; e++) x += SAMPLE_WEIGHTS[e] * v[SAMPLE_VERTS[e]];
            glow[i] += (fminf(x / 0.35f, 1.0f) - glow[i]) * 0.3f;
            // Blue cores with red edges; one channel per pixel in single-channel mode.
            float blue = fmaxf(glow[i] - 0.4f, 0) / 0.6f, red = fminf(glow[i], 0.5f) * (1 - blue);
            CRGB c(dim8_video(red * 255), 0, dim8_video(blue * 255));
            if (SINGLE_CHANNEL) c = blue > red ? CRGB(0, 0, c.b) : CRGB(c.r, 0, 0);
            out[i / LEDS][i % LEDS] = c;
        }
    }
};

Strip strips[OUTPUTS];
Excitable excitable;
GrayScott gray_scott;
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

    if (program == 2) {
        gray_scott.render(leds);
    } else if (program == 1) {
        excitable.render(leds);
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
