// Stand-ins for the parts of Arduino and FastLED that lantern.h uses, so it compiles for the
// browser. The colour maths is FastLED 3.10's (MIT licence), with FASTLED_SCALE8_FIXED.
#pragma once

#include <math.h>
#include <stdint.h>

constexpr float TWO_PI = 6.283185307179586f;

uint32_t millis();
uint32_t esp_random();

inline uint8_t scale8(uint8_t i, uint8_t scale) { return ((uint16_t)i * (1 + (uint16_t)scale)) >> 8; }
inline uint8_t scale8_video(uint8_t i, uint8_t scale) { return (((int)i * (int)scale) >> 8) + ((i && scale) ? 1 : 0); }
inline uint8_t dim8_video(uint8_t x) { return scale8_video(x, x); }

struct CHSV {
    uint8_t hue, sat, val;
    CHSV(uint8_t h, uint8_t s, uint8_t v) : hue(h), sat(s), val(v) {}
};

struct CRGB {
    uint8_t r, g, b;

    CRGB() = default;
    CRGB(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}
    CRGB(const CHSV &hsv);

    uint8_t &operator[](int i) { return (&r)[i]; }
    uint8_t operator[](int i) const { return (&r)[i]; }
    explicit operator bool() const { return r || g || b; }

    CRGB &nscale8_video(uint8_t scale) {
        uint8_t nonzero = scale != 0;
        for (int k = 0; k < 3; k++) (*this)[k] = (*this)[k] ? (((int)(*this)[k] * scale) >> 8) + nonzero : 0;
        return *this;
    }
};

// FastLED's hsv2rgb_rainbow, with its default moderate yellow boost.
inline CRGB::CRGB(const CHSV &hsv) {
    uint8_t hue = hsv.hue, sat = hsv.sat, val = hsv.val;
    uint8_t offset8 = (hue & 0x1F) << 3;
    uint8_t third = scale8(offset8, 256 / 3);
    uint8_t twothirds = scale8(offset8, (256 * 2) / 3);
    uint8_t r, g, b;
    switch (hue >> 5) {
        case 0: r = 255 - third, g = third, b = 0; break;
        case 1: r = 171, g = 85 + third, b = 0; break;
        case 2: r = 171 - twothirds, g = 170 + third, b = 0; break;
        case 3: r = 0, g = 255 - third, b = third; break;
        case 4: r = 0, g = 171 - twothirds, b = 85 + twothirds; break;
        case 5: r = third, g = 0, b = 255 - third; break;
        case 6: r = 85 + third, g = 0, b = 171 - third; break;
        default: r = 170 + third, g = 0, b = 85 - third; break;
    }
    if (sat != 255) {
        if (sat == 0) {
            r = g = b = 255;
        } else {
            uint8_t desat = 255 - sat;
            desat = scale8_video(desat, desat);
            uint8_t satscale = 255 - desat;
            r = scale8(r, satscale) + desat;
            g = scale8(g, satscale) + desat;
            b = scale8(b, satscale) + desat;
        }
    }
    if (val != 255) {
        val = scale8_video(val, val);
        if (val == 0) {
            r = g = b = 0;
        } else {
            r = scale8(r, val), g = scale8(g, val), b = scale8(b, val);
        }
    }
    this->r = r, this->g = g, this->b = b;
}
