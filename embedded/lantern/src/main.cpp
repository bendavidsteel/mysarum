// Drives the LED strips on a Wroomy board with the programs in lantern.h, and accepts serial
// commands so a host can light individual LEDs for mapping, or tune the programs live.
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
//   k NAME VALUE     set a setting from params.h until the next reboot, reply "ok" or "unknown"
//   i                reply with the mean time of one pass of the main loop
// c, s and w switch from the animation into mapping mode.

#include <Arduino.h>
#include <FastLED.h>

#include "lantern.h"

CRGB leds[OUTPUTS][LEDS];
Lantern lantern;
bool mapping = false;
char line[64];
int line_len = 0;
float loop_ms = 0;  // moving average

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
            set_param("program", atoi(cmd + 1));
            mapping = false;
            return;
        case 'm':
            set_param("map_override", atoi(cmd + 1));
            return;
        case 'r':
            set_param("nca_rgb", atoi(cmd + 1));
            return;
        case 'k': {
            char name[32];
            float value;
            if (sscanf(cmd + 1, "%31s %f", name, &value) != 2) return;
            Serial.println(set_param(name, value) ? "ok" : "unknown");
            return;
        }
        case 'i':
            Serial.printf("loop %.2f ms\n", loop_ms);
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
    FastLED.setMaxPowerInVoltsAndMilliamps(5, params.max_milliamps);
    FastLED.clear(true);
    uint32_t t = micros();
    for (int i = 0; i < 100; i++) lantern.gray_scott.step();
    Serial.printf("lantern: %d x %d leds, reaction-diffusion %lu us/step\n", OUTPUTS, LEDS, (micros() - t) / 100);
}

void loop() {
    static uint32_t last = micros();
    uint32_t now = micros();
    loop_ms += ((now - last) / 1000.0f - loop_ms) * 0.01f;
    last = now;

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

    lantern.step();
    lantern.render(leds);
    // FastLED's per-channel current model is close enough for these 12V strips.
    FastLED.setMaxPowerInVoltsAndMilliamps(5, params.max_milliamps);
    FastLED.setBrightness(params.brightness);
    FastLED.show();
    delay(10);
}
