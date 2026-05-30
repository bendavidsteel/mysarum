//! Birdsong production model after G.B. Mindlin, "The physics of birdsong
//! production", Contemporary Physics 54:2 (2013), 91-96.
//!
//! The avian vocal organ (syrinx) is modelled as a low-dimensional nonlinear
//! dynamical system. Three coupled differential mechanisms are chained:
//!
//!   1. SOUND SOURCE — the oscillating labia, written in the paper's "normal
//!      form" reduction (Sitt/Arneodo/Mindlin). Two first-order ODEs:
//!
//!          dx/dt = y
//!          dy/dt = -alpha*g^2 - beta*g^2*x - g^2*x^3 - g*x^2*y + g^2*x^2 - g*x*y
//!
//!      where  alpha <-> air-sac pressure,  beta <-> syringeal tension,
//!      and    g (gamma) <-> a time constant fixing the spectral range.
//!      x(t) is the midpoint labial displacement; it modulates the airflow.
//!
//!   2. TRACHEA — a tube carrying the wave, modelled as a delay line with a
//!      reflection at the (open) beak end, per the schematic in Fig. 1:
//!
//!          Pi(t) = x(t) - r * Pi(t - T)
//!
//!   3. OEC — the oro-esophageal cavity, a Helmholtz resonator [4] that shapes
//!      the harmonic content. A driven, damped second-order oscillator:
//!
//!          Pout'' + (1/tau)*Pout' + w_oec^2 * Pout = G * dPi/dt
//!
//! The window shows three views of the synthesised voice: a vectorscope
//! (delay embedding of the labial limit cycle, top-left), an oscilloscope of
//! the radiated pressure wave (top), and a scrolling spectrogram waterfall
//! (short-time DFT, 0–8 kHz, ~6 s of history) filling the lower region.
//!
//! The bird "sings" by moving along a path (alpha(t), beta(t)) in parameter
//! space. Here YOU are the bird: the mouse position is that path.
//!     mouse X  ->  alpha (air-sac pressure)   left = quiet, right = strong
//!     mouse Y  ->  beta  (syringeal tension)   down = low pitch, up = high
//!     scroll   ->  gamma (overall time scale / frequency range)
//!     space    ->  pause / resume the airflow
//!     s        ->  save a screenshot

use std::sync::{Arc, Mutex};

use nannou::prelude::bevy_asset::RenderAssetUsages;
use nannou::prelude::*;
use nannou_audio as audio;
use nannou_audio::Buffer;
use ringbuf::{
    traits::{Consumer, Producer, Split},
    HeapRb,
};

// ── Window / scope geometry ────────────────────────────────────────────────

const WIN_W: u32 = 1200;
const WIN_H: u32 = 720;
const SCOPE_LEN: usize = 2048; // samples held for the oscilloscope

// ── Parameter ranges (the phonating region of parameter space) ──────────────

const ALPHA_MIN: f32 = -0.05; // below ~0 the labia do not oscillate (silence)
const ALPHA_MAX: f32 = 0.55;
const BETA_MIN: f32 = -0.20;
const BETA_MAX: f32 = 0.40;
const GAMMA_MIN: f32 = 8_000.0;
const GAMMA_MAX: f32 = 60_000.0;

const OVERSAMPLE: usize = 24; // ODE substeps per audio frame (Euler stability)

// ── Spectrogram (short-time DFT waterfall) ──────────────────────────────────

const FFT_SIZE: usize = 1024; // samples per analysis window
const N_FREQ: usize = 192; // frequency bins / texture height
const TIME_COLS: usize = 360; // history columns / texture width (~6 s at 60 fps)
const FREQ_MAX: f32 = 8_000.0; // top of the displayed frequency axis (Hz)
const DB_FLOOR: f32 = -55.0; // magnitudes at/below this map to black

// ── Shared, mouse-driven parameters ─────────────────────────────────────────

#[derive(Clone, Copy)]
struct Params {
    alpha: f32,
    beta: f32,
    gamma: f32,
    volume: f32,
    paused: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            alpha: 0.20, // comfortably inside the phonating region
            beta: 0.05,
            gamma: 24_000.0,
            volume: 0.6,
            paused: false,
        }
    }
}

// ── Audio-thread state ──────────────────────────────────────────────────────

struct Syrinx {
    params: Arc<Mutex<Params>>,
    cached: Params,

    // 1. labial oscillator
    x: f32,
    y: f32,

    // 2. trachea delay line (reflection comb)
    trachea: Vec<f32>,
    trachea_pos: usize,
    r: f32, // reflection coefficient at the beak

    // 3. oro-esophageal cavity — Helmholtz resonator, realised as a normalised
    //    2-pole state-variable filter (its bandpass response IS a driven,
    //    damped second-order oscillator; the bandpass removes the DC offset so
    //    the non-phonating region is genuinely silent).
    oec_low: f32,
    oec_band: f32,
    oec_f: f32, // tuning coefficient (set from the formant frequency)
    oec_q: f32, // damping (1/Q)

    scope: ringbuf::HeapProd<f32>,
}

fn render_audio(s: &mut Syrinx, buffer: &mut Buffer) {
    if let Ok(p) = s.params.try_lock() {
        s.cached = *p;
    }
    let p = s.cached;
    let sr = buffer.sample_rate() as f32;
    let dt = 1.0 / (sr * OVERSAMPLE as f32);
    let g = p.gamma;
    let g2 = g * g;
    let inv_os = 1.0 / OVERSAMPLE as f32;

    for frame in buffer.frames_mut() {
        if p.paused {
            for ch in frame.iter_mut() {
                *ch = 0.0;
            }
            s.scope.try_push(0.0).ok();
            continue;
        }

        let mut sample_acc = 0.0f32;

        for _ in 0..OVERSAMPLE {
            // 1. Labial normal form (forward Euler on the oversampled grid).
            let x = s.x;
            let y = s.y;
            let dx = y;
            let dy = -p.alpha * g2 - p.beta * g2 * x - g2 * x * x * x - g * x * x * y
                + g2 * x * x
                - g * x * y;
            s.x = x + dx * dt;
            s.y = y + dy * dt;
            // Keep the integrator from blowing up at extreme parameters.
            s.x = s.x.clamp(-3.0, 3.0);
            s.y = s.y.clamp(-3.0 * g, 3.0 * g);

            // 2. Trachea: Pi(t) = source - r * Pi(t - T).
            let delayed = s.trachea[s.trachea_pos];
            let pi = s.x - s.r * delayed;
            s.trachea[s.trachea_pos] = pi;
            s.trachea_pos = (s.trachea_pos + 1) % s.trachea.len();

            // 3. OEC: Chamberlin state-variable filter; take the bandpass tap.
            s.oec_low += s.oec_f * s.oec_band;
            let high = pi - s.oec_low - s.oec_q * s.oec_band;
            s.oec_band += s.oec_f * high;

            sample_acc += s.oec_band;
        }

        // Average the oversampled output, scale, and soft-clip.
        let raw = sample_acc * inv_os;
        let out = (raw * 2.0 * p.volume).tanh();

        for ch in frame.iter_mut() {
            *ch = out;
        }
        s.scope.try_push(out).ok();
    }
}

// ── Model ───────────────────────────────────────────────────────────────────

struct Model {
    window: Entity,
    params: Arc<Mutex<Params>>,
    // Wrapped in a Mutex so `Model` is `Sync` (required by the app runtime);
    // the consumer itself holds a non-Sync Cell.
    scope_rx: Mutex<ringbuf::HeapCons<f32>>,
    scope: Vec<f32>, // rolling oscilloscope buffer

    // Spectrogram: a GPU texture scrolled one column per frame. The short-time
    // DFT is evaluated directly at N_FREQ linearly-spaced bins via precomputed
    // twiddle tables (cheap at this size, and avoids an FFT dependency).
    spectro: Handle<Image>,
    hann: Vec<f32>,         // analysis window, len FFT_SIZE
    dft_cos: Vec<f32>,      // N_FREQ * FFT_SIZE
    dft_sin: Vec<f32>,      // N_FREQ * FFT_SIZE
    dft_norm: f32,          // amplitude normalisation (2 / sum(hann))
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let window = app
        .new_window()
        .size(WIN_W, WIN_H)
        .title("Birdsong — Mindlin syrinx model")
        .mouse_moved(mouse_moved)
        .mouse_wheel(mouse_wheel)
        .key_pressed(key_pressed)
        .view(view)
        .build();

    let params = Arc::new(Mutex::new(Params::default()));

    // Ring buffer carrying output samples from the audio thread to the scope.
    let (scope_tx, scope_rx) = HeapRb::<f32>::new(SCOPE_LEN * 8).split();

    // ~1.5 ms round-trip trachea -> a short reflection comb.
    let trachea_len = 64usize;

    // OEC formant near ~2.2 kHz. The SVF runs on the oversampled grid.
    let formant_hz = 2200.0_f32;
    let sr_os = 44_100.0_f32 * OVERSAMPLE as f32;
    let oec_f = 2.0 * (PI * formant_hz / sr_os).sin();

    let syrinx = Syrinx {
        params: Arc::clone(&params),
        cached: Params::default(),
        x: 0.01,
        y: 0.0,
        trachea: vec![0.0; trachea_len],
        trachea_pos: 0,
        r: 0.75,
        oec_low: 0.0,
        oec_band: 0.0,
        oec_f,
        oec_q: 0.5, // mild formant emphasis (Q = 2)
        scope: scope_tx,
    };

    // The audio stream is not Send/Sync, so it lives on its own thread that
    // owns it for the lifetime of the program.
    std::thread::spawn(move || {
        let audio_host = audio::Host::new();
        let stream = audio_host
            .new_output_stream(syrinx)
            .render(render_audio)
            .build()
            .unwrap();
        stream.play().unwrap();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });

    // Spectrogram texture: width = time columns, height = frequency bins.
    let spectro = app.assets_mut::<Image>().add(Image::new(
        Extent3d {
            width: TIME_COLS as u32,
            height: N_FREQ as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0u8; TIME_COLS * N_FREQ * 4],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(), // MAIN_WORLD | RENDER_WORLD: editable + uploaded
    ));

    // Hann window and per-bin DFT twiddle tables (freq linear in [0, FREQ_MAX]).
    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|n| {
            let w = 2.0 * PI * n as f32 / (FFT_SIZE as f32 - 1.0);
            0.5 - 0.5 * w.cos()
        })
        .collect();
    let hann_sum: f32 = hann.iter().sum();
    let analysis_sr = 44_100.0_f32; // the scope buffer runs at the device rate
    let mut dft_cos = vec![0.0f32; N_FREQ * FFT_SIZE];
    let mut dft_sin = vec![0.0f32; N_FREQ * FFT_SIZE];
    for k in 0..N_FREQ {
        let freq = FREQ_MAX * k as f32 / (N_FREQ as f32 - 1.0);
        let omega = 2.0 * PI * freq / analysis_sr;
        for n in 0..FFT_SIZE {
            dft_cos[k * FFT_SIZE + n] = (omega * n as f32).cos();
            dft_sin[k * FFT_SIZE + n] = (omega * n as f32).sin();
        }
    }

    Model {
        window,
        params,
        scope_rx: Mutex::new(scope_rx),
        scope: vec![0.0; SCOPE_LEN],
        spectro,
        hann,
        dft_cos,
        dft_sin,
        dft_norm: 2.0 / hann_sum,
    }
}

fn update(app: &App, model: &mut Model) {
    // Drain new samples and keep the most recent SCOPE_LEN in a rolling window.
    let mut incoming = Vec::new();
    if let Ok(mut rx) = model.scope_rx.lock() {
        while let Some(s) = rx.try_pop() {
            incoming.push(s);
        }
    }
    if !incoming.is_empty() {
        let n = incoming.len();
        if n >= SCOPE_LEN {
            model.scope.copy_from_slice(&incoming[n - SCOPE_LEN..]);
        } else {
            model.scope.drain(0..n);
            model.scope.extend_from_slice(&incoming);
        }
    }

    // ── Spectrogram: STDFT of the latest window → scroll texture one column ──
    update_spectrogram(app, model);

    // ── egui readout (this fork has no draw.text(), so all text lives here) ──
    let freq = estimate_freq(&model.scope, 44_100.0);
    let mut p = model.params.lock().map(|g| *g).unwrap_or_default();

    let mut egui_ctx = app.egui_for_window(model.window);
    let ctx = egui_ctx.get_mut();
    egui::Window::new("Syrinx").show(&ctx, |ui| {
        ui.label("Mindlin labial normal form");
        ui.separator();
        ui.label(format!("α  pressure  {:+.3}   [mouse X]", p.alpha));
        ui.label(format!("β  tension   {:+.3}   [mouse Y]", p.beta));
        ui.label(format!("γ  time const {:.0}   [scroll]", p.gamma));
        ui.label(format!("fundamental ~ {:.0} Hz", freq));
        ui.separator();
        ui.add(egui::Slider::new(&mut p.volume, 0.0..=1.0).text("volume"));
        ui.checkbox(&mut p.paused, "pause airflow (space)");
        ui.separator();
        ui.label(format!("spectrogram: 0 – {:.0} kHz, ~6 s", FREQ_MAX / 1000.0));
        ui.label("Move the mouse to trace a path through");
        ui.label("parameter space — that is the bird's gesture.");
    });
    drop(egui_ctx);

    if let Ok(mut g) = model.params.lock() {
        // Preserve α/β/γ (mouse/scroll own them); take UI-owned fields.
        g.volume = p.volume;
        g.paused = p.paused;
    }
}

/// Evaluate the short-time DFT of the most recent FFT_SIZE scope samples and
/// push the resulting magnitude column onto the right edge of the scrolling
/// spectrogram texture (oldest time at the left, high frequency at the top).
fn update_spectrogram(app: &App, model: &mut Model) {
    let n = model.scope.len();
    if n < FFT_SIZE {
        return;
    }
    let frame = &model.scope[n - FFT_SIZE..];

    // Magnitude per bin → an RGBA column (one texel per frequency bin).
    let mut column = [[0u8; 4]; N_FREQ];
    for k in 0..N_FREQ {
        let (cos, sin) = (
            &model.dft_cos[k * FFT_SIZE..(k + 1) * FFT_SIZE],
            &model.dft_sin[k * FFT_SIZE..(k + 1) * FFT_SIZE],
        );
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for i in 0..FFT_SIZE {
            let s = frame[i] * model.hann[i];
            re += s * cos[i];
            im -= s * sin[i];
        }
        let mag = (re * re + im * im).sqrt() * model.dft_norm;
        let db = 20.0 * (mag + 1e-9).log10();
        let t = ((db - DB_FLOOR) / -DB_FLOOR).clamp(0.0, 1.0);
        let (r, g, b) = inferno(t);
        // Row 0 is the top of the texture → highest frequency.
        column[N_FREQ - 1 - k] = [r, g, b, 255];
    }

    // Scroll every row one texel to the left, then write the new column.
    let mut img = app.assets_mut::<Image>();
    let Some(img) = img.get_mut(&model.spectro) else {
        return;
    };
    let Some(buf) = img.data.as_mut() else {
        return;
    };
    let w = TIME_COLS;
    for row in 0..N_FREQ {
        let base = row * w * 4;
        buf.copy_within(base + 4..base + w * 4, base);
        let last = base + (w - 1) * 4;
        buf[last..last + 4].copy_from_slice(&column[row]);
    }
}

/// Compact "inferno"-style colormap (black → purple → red → orange → pale
/// yellow) via piecewise-linear interpolation over a handful of control colors.
fn inferno(t: f32) -> (u8, u8, u8) {
    const STOPS: [(f32, f32, f32, f32); 6] = [
        (0.00, 0.00, 0.00, 0.02),
        (0.20, 0.18, 0.03, 0.30),
        (0.40, 0.53, 0.06, 0.42),
        (0.60, 0.84, 0.22, 0.26),
        (0.80, 0.98, 0.55, 0.04),
        (1.00, 0.99, 0.96, 0.66),
    ];
    let t = t.clamp(0.0, 1.0);
    let mut i = 0;
    while i + 1 < STOPS.len() && t > STOPS[i + 1].0 {
        i += 1;
    }
    let (t0, r0, g0, b0) = STOPS[i];
    let (t1, r1, g1, b1) = STOPS[(i + 1).min(STOPS.len() - 1)];
    let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    let lerp = |a: f32, b: f32| ((a + (b - a) * f) * 255.0).round().clamp(0.0, 255.0) as u8;
    (lerp(r0, r1), lerp(g0, g1), lerp(b0, b1))
}

// ── Input: the mouse IS the bird's motor gesture ────────────────────────────

fn mouse_moved(_app: &App, model: &mut Model, pos: Vec2) {
    // pos is centred at (0,0); map across the known window extents.
    let fx = (pos.x / WIN_W as f32 + 0.5).clamp(0.0, 1.0);
    let fy = (pos.y / WIN_H as f32 + 0.5).clamp(0.0, 1.0);
    if let Ok(mut p) = model.params.lock() {
        p.alpha = ALPHA_MIN + fx * (ALPHA_MAX - ALPHA_MIN);
        p.beta = BETA_MIN + fy * (BETA_MAX - BETA_MIN);
    }
}

fn mouse_wheel(_app: &App, model: &mut Model, wheel: MouseWheel) {
    if let Ok(mut p) = model.params.lock() {
        let factor = 1.0 + wheel.y * 0.05;
        p.gamma = (p.gamma * factor).clamp(GAMMA_MIN, GAMMA_MAX);
    }
}

fn key_pressed(app: &App, model: &mut Model, key: KeyCode) {
    match key {
        KeyCode::Space => {
            if let Ok(mut p) = model.params.lock() {
                p.paused = !p.paused;
            }
        }
        KeyCode::KeyS => {
            app.window(model.window)
                .save_screenshot(app.exe_name().unwrap() + ".png");
        }
        _ => {}
    }
}

// ── View: oscilloscope + phase portrait + parameter readout ─────────────────

fn view(app: &App, model: &Model) {
    let draw = app.draw();
    draw.background().color(Color::srgb(0.04, 0.04, 0.06));

    // Use the known window entity (window_rect() needs a *focused* window,
    // which isn't registered yet on the first frame and would panic).
    let win = app.window(model.window).rect();

    let n = model.scope.len();
    let margin = 20.0;

    // The window splits into a top strip (vectorscope + oscilloscope) and a
    // larger spectrogram waterfall filling the rest.
    let top_bot = win.top() - win.h() * 0.30;

    // ── Phase portrait of the labial oscillator (delay embedding), top-right;
    //    the top-left is occupied by the egui info panel. ──
    let pp_cx = win.right() - win.w() * 0.07;
    let pp_cy = (win.top() - margin + top_bot) * 0.5;
    let pp_r = (win.top() - margin - top_bot) * 0.5 * 0.92;
    draw.ellipse()
        .x_y(pp_cx, pp_cy)
        .radius(pp_r * 1.05)
        .no_fill()
        .stroke_weight(1.0)
        .stroke(Color::srgb(0.15, 0.15, 0.22));
    // Delay of a few samples ≈ a quarter period, so the limit cycle opens into
    // a loop rather than collapsing onto the diagonal.
    let lag = 6usize;
    let emb = (0..n - lag).map(|i| {
        let xv = model.scope[i].clamp(-1.0, 1.0);
        let yv = model.scope[i + lag].clamp(-1.0, 1.0);
        (pt2(pp_cx + xv * pp_r, pp_cy + yv * pp_r), Color::srgb(0.95, 0.55, 0.30))
    });
    draw.polyline().weight(1.0).points_colored(emb);

    // ── Oscilloscope (top strip, between the info panel and the vectorscope) ──
    let scope_left = win.left() + win.w() * 0.31;
    let scope_right = pp_cx - pp_r * 1.25 - 20.0;
    let scope_mid = pp_cy;
    let scope_amp = pp_r;
    draw.line()
        .start(pt2(scope_left, scope_mid))
        .end(pt2(scope_right, scope_mid))
        .weight(1.0)
        .color(Color::srgb(0.15, 0.15, 0.20));
    let waveform = (0..n).map(|i| {
        let x = map_range(i, 0, n - 1, scope_left, scope_right);
        let y = scope_mid + model.scope[i].clamp(-1.0, 1.0) * scope_amp;
        (pt2(x, y), Color::srgb(0.35, 0.95, 0.65))
    });
    draw.polyline().weight(1.4).points_colored(waveform);

    // ── Spectrogram waterfall (textured rect filling the lower region) ──
    // A plain `draw.rect().texture()` is the path this fork actually supports
    // (see examples/draw/draw_texture.rs): the rect supplies its own UVs and
    // `.texture()` binds the image to the shader model.
    let sg_l = win.left() + margin;
    let sg_r = win.right() - margin;
    let sg_t = top_bot - 10.0;
    let sg_b = win.bottom() + margin;
    draw.rect()
        .x_y((sg_l + sg_r) * 0.5, (sg_t + sg_b) * 0.5)
        .w_h(sg_r - sg_l, sg_t - sg_b)
        .texture(&model.spectro);
    // Frame + frequency ticks (0, 2, 4, 6, 8 kHz) along the left edge.
    draw.rect()
        .x_y((sg_l + sg_r) * 0.5, (sg_t + sg_b) * 0.5)
        .w_h(sg_r - sg_l, sg_t - sg_b)
        .no_fill()
        .stroke_weight(1.0)
        .stroke(Color::srgb(0.2, 0.2, 0.26));
    for k in 0..=4 {
        let frac = k as f32 / 4.0; // 0 kHz at bottom, FREQ_MAX at top
        let y = sg_b + frac * (sg_t - sg_b);
        draw.line()
            .start(pt2(sg_l, y))
            .end(pt2(sg_l + 8.0, y))
            .weight(1.0)
            .color(Color::srgb(0.35, 0.35, 0.42));
    }
    // Text (labels, parameter readout, axis units) is drawn via egui in
    // `update`, since draw.text() is unimplemented in this nannou fork.
}

/// Rough fundamental-frequency estimate from positive-going zero crossings.
fn estimate_freq(buf: &[f32], sample_rate: f32) -> f32 {
    let mut crossings = 0u32;
    let mut first = None;
    let mut last = 0usize;
    for i in 1..buf.len() {
        if buf[i - 1] <= 0.0 && buf[i] > 0.0 {
            if first.is_none() {
                first = Some(i);
            }
            last = i;
            crossings += 1;
        }
    }
    if let (Some(f), true) = (first, crossings > 1) {
        let span = (last - f) as f32;
        if span > 0.0 {
            return (crossings - 1) as f32 * sample_rate / span;
        }
    }
    0.0
}
