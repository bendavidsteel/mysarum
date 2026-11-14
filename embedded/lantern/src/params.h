// Tunable settings: type, name, value at boot, lowest, highest. Ranges named _lo/_hi are
// re-rolled within at random as the programs run; set both ends equal to pin a value.
// The simulator in sim/ edits the values in place, so keep one X(...) per line.
// Colour maps: 0 twilight, 1 magma, 2 inferno, 3 viridis, 4 ember.
#pragma once

#define LANTERN_PARAMS(X)                        \
    X(int, program, 4, 0, 4)                     \
    X(int, brightness, 255, 0, 255)              \
    X(int, max_milliamps, 900, 100, 1000)        \
    X(int, single_channel, 0, 0, 1)              \
    X(int, map_override, -1, -1, 4)              \
    X(float, mood_decay_lo, 0.8, 0.5, 1)         \
    X(float, mood_decay_hi, 0.98, 0.5, 1)        \
    X(float, mood_sparkle_lo, 0, 0, 1)           \
    X(float, mood_sparkle_hi, 0.6, 0, 1)         \
    X(float, mood_comet_lo, 0, 0, 0.2)           \
    X(float, mood_comet_hi, 0.04, 0, 0.2)        \
    X(float, mood_spread_lo, 0.02, 0, 1)         \
    X(float, mood_spread_hi, 0.5, 0, 1)          \
    X(float, mood_wander_lo, 0.0005, 0, 0.02)    \
    X(float, mood_wander_hi, 0.006, 0, 0.02)     \
    X(float, mood_saturation_lo, 0.6, 0, 1)      \
    X(float, mood_saturation_hi, 1, 0, 1)        \
    X(int, mood_frames_lo, 600, 50, 5000)        \
    X(int, mood_frames_hi, 2000, 50, 5000)       \
    X(float, comet_speed_lo, 0.1, 0.01, 2)       \
    X(float, comet_speed_hi, 0.8, 0.01, 2)       \
    X(int, ex_map, 2, 0, 4)                      \
    X(float, ex_blend, 0.3, 0, 2)                \
    X(int, ex_recovery, 10, 3, 30)               \
    X(float, ex_threshold_lo, 0.1, 0, 1)         \
    X(float, ex_threshold_hi, 0.3, 0, 1)         \
    X(float, ex_spontaneous_lo, 0.0005, 0, 0.05) \
    X(float, ex_spontaneous_hi, 0.006, 0, 0.05)  \
    X(int, ex_step_ms_lo, 60, 10, 500)           \
    X(int, ex_step_ms_hi, 160, 10, 500)          \
    X(int, ex_steps_lo, 300, 10, 3000)           \
    X(int, ex_steps_hi, 900, 10, 3000)           \
    X(int, gs_map, 4, 0, 4)                      \
    X(float, gs_blend, 0, 0, 2)                  \
    X(float, gs_du, 0.2, 0, 0.5)                 \
    X(float, gs_dv, 0.1, 0, 0.5)                 \
    X(float, gs_feed_chaos, 0.018, 0, 0.1)       \
    X(float, gs_feed_spots, 0.025, 0, 0.1)       \
    X(float, gs_kill_chaos, 0.051, 0, 0.1)       \
    X(float, gs_kill_spots, 0.055, 0, 0.1)       \
    X(int, gs_steps_per_frame, 1, 1, 8)          \
    X(float, gs_regime_change, 0.0003, 0, 0.01)  \
    X(float, gs_regime_rate, 0.0005, 0, 0.01)    \
    X(float, gs_drop_chance, 0.0005, 0, 0.01)    \
    X(float, gs_drop_size, 0.9, 0.5, 0.99)       \
    X(int, gs_reseed_ms, 3000, 100, 20000)       \
    X(float, gs_contrast, 0.15, 0.01, 1)         \
    X(int, cy_map, 0, 0, 4)                      \
    X(float, cy_blend, 0.5, 0, 2)                \
    X(int, cy_states_lo, 4, 2, 12)               \
    X(int, cy_states_hi, 6, 2, 12)               \
    X(float, cy_threshold_lo, 0, 0, 0.5)         \
    X(float, cy_threshold_hi, 0.08, 0, 0.5)      \
    X(int, cy_step_ms_lo, 80, 10, 500)           \
    X(int, cy_step_ms_hi, 200, 10, 500)          \
    X(int, cy_steps_lo, 400, 10, 5000)           \
    X(int, cy_steps_hi, 1200, 10, 5000)          \
    X(float, cy_scramble, 0.003, 0, 0.05)        \
    X(int, nca_rgb, 1, 0, 1)                     \
    X(int, nca_map, 3, 0, 4)                     \
    X(float, nca_blend, 0.3, 0, 2)               \
    X(float, nca_gain, 1.6, 0, 4)                \
    X(float, nca_leak, 0.4, 0, 2)                \
    X(float, nca_update_odds, 0.5, 0.05, 1)      \
    X(int, nca_step_ms, 40, 10, 500)             \
    X(int, nca_eval_steps, 125, 10, 1000)        \
    X(float, nca_step, 0.12, 0.005, 0.5)         \
    X(float, nca_temperature, 0.1, 0.001, 2)     \
    X(float, nca_space_weight, 1, 0, 5)          \
    X(float, nca_time_weight, 1, 0, 5)           \
    X(float, nca_fast, 0.3333, 0.01, 1)          \
    X(float, nca_slow, 0.04, 0.001, 1)           \
    X(float, nca_flat, 0.05, 0, 0.5)
