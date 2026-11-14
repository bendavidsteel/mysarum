// The firmware's programs compiled for the browser: index.html calls these exports to run the
// main loop on a simulated clock and read back each frame as the strips would show it.
#include <stdio.h>

#include "host.h"
#include "../src/lantern.h"

#define EXPORT extern "C" __attribute__((visibility("default")))

static double clock_ms = 0, pending_ms = 0;
static uint64_t rng = 1;
static Lantern *lantern = nullptr;
static CRGB leds[OUTPUTS][LEDS];
static uint8_t frame[GRAPH_NODES * 3];
static float milliamps = 0;
static int shown_brightness = 255;
static char status[256];

uint32_t millis() { return (uint32_t)clock_ms; }

uint32_t esp_random() {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return rng >> 32;
}

// FastLED's power model for these strips: per-channel mW at full brightness, plus the MCU.
static void show() {
    uint32_t r = 0, g = 0, b = 0;
    for (auto &strip : leds)
        for (auto &p : strip) r += p.r, g += p.g, b += p.b;
    uint32_t total_mw = ((r * 80) >> 8) + ((g * 55) >> 8) + ((b * 75) >> 8) + 5 * GRAPH_NODES + 125;
    uint32_t target = params.brightness, max_mw = 5 * params.max_milliamps;
    uint32_t requested = total_mw * target / 256;
    shown_brightness = requested > max_mw ? target * max_mw / requested : target;
    milliamps = (float)total_mw * shown_brightness / 256 / 5;
    for (int i = 0; i < GRAPH_NODES; i++)
        for (int k = 0; k < 3; k++) frame[3 * i + k] = scale8(leds[i / LEDS][i % LEDS][k], shown_brightness);
}

EXPORT void sim_reset(uint32_t seed) {
    rng = 0x9E3779B97F4A7C15ull ^ ((uint64_t)seed << 1 | 1);
    clock_ms = pending_ms = 0;
    delete lantern;
    lantern = new Lantern();
}

// Runs as many passes of the main loop as fit in elapsed_ms. On the ESP a pass takes loop_ms, of
// which render_ms passes between the program's step and the render; the display's fades depend on it.
EXPORT int sim_advance(float elapsed_ms, float loop_ms, float render_ms) {
    pending_ms += elapsed_ms;
    render_ms = fminf(render_ms, loop_ms);
    int passes = 0;
    while (pending_ms >= loop_ms && passes < 5000) {
        lantern->step();
        clock_ms += render_ms;
        lantern->render(leds);
        clock_ms += loop_ms - render_ms, pending_ms -= loop_ms, passes++;
    }
    if (passes == 5000) pending_ms = 0;
    show();
    return passes;
}

EXPORT const uint8_t *sim_frame() { return frame; }
EXPORT float sim_milliamps() { return milliamps; }
EXPORT int sim_brightness() { return shown_brightness; }
EXPORT double sim_clock() { return clock_ms; }
EXPORT const float *sim_positions() { return &GRAPH_POS[0][0]; }
EXPORT int sim_nodes() { return GRAPH_NODES; }

EXPORT int sim_param_count() { return PARAM_COUNT; }
EXPORT const char *sim_param_name(int i) { return PARAM_INFO[i].name; }
EXPORT int sim_param_is_int(int i) { return PARAM_INFO[i].is_int; }
EXPORT float sim_param_lo(int i) { return PARAM_INFO[i].lo; }
EXPORT float sim_param_hi(int i) { return PARAM_INFO[i].hi; }
EXPORT float sim_param_get(int i) { return get_param(i); }
EXPORT void sim_param_set(int i, float v) { set_param(i, v); }

// What the current program has rolled for itself, as name=value pairs separated by spaces.
EXPORT const char *sim_status() {
    const Lantern &l = *lantern;
    switch (params.program) {
        case 1:
            snprintf(status, sizeof(status), "threshold=%.3f spontaneous=%.4f step_ms=%u steps_left=%u",
                     l.excitable.threshold, l.excitable.spontaneous, l.excitable.step_ms, l.excitable.steps_left);
            break;
        case 2:
            snprintf(status, sizeof(status), "regime=%.2f target=%.2f range=%.3f..%.3f", l.gray_scott.regime,
                     l.gray_scott.target_regime, l.gray_scott.range_lo, l.gray_scott.range_hi);
            break;
        case 3:
            snprintf(status, sizeof(status), "states=%d threshold=%.3f step_ms=%u steps_left=%u frozen=%u",
                     l.cyclic.states, l.cyclic.threshold, l.cyclic.step_ms, l.cyclic.steps_left, l.cyclic.frozen);
            break;
        case 4:
            snprintf(status, sizeof(status), "kept_fitness=%.2f eval_step=%u flat_steps=%u", l.nca.kept_fitness,
                     l.nca.eval_steps, l.nca.flat_steps);
            break;
        default: {
            const Mood &a = l.strips[0].mood, &b = l.strips[1].mood;
            snprintf(status, sizeof(status), "decay=%.2f/%.2f sparkle=%.2f/%.2f comet=%.3f/%.3f frames_left=%u/%u",
                     a.decay, b.decay, a.sparkle_rate, b.sparkle_rate, a.comet_rate, b.comet_rate, a.frames_left,
                     b.frames_left);
        }
    }
    return status;
}
