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
//! The bird "sings" by moving along a path (alpha(t), beta(t)) in parameter
//! space. Here YOU are the bird: the mouse position is that path.
//!     mouse X  ->  alpha (air-sac pressure)   left = quiet, right = strong
//!     mouse Y  ->  beta  (syringeal tension)   down = low pitch, up = high
//!     scroll   ->  gamma (overall time scale / frequency range)
//!     space    ->  pause / resume the airflow
//!     s        ->  save a screenshot

use std::sync::{Arc, Mutex};

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

    Model {
        window,
        params,
        scope_rx: Mutex::new(scope_rx),
        scope: vec![0.0; SCOPE_LEN],
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

    // ── Oscilloscope (top portion of the window) ──
    let scope_top = win.top() - 30.0;
    let scope_bot = win.bottom() + win.h() * 0.40;
    let scope_mid = (scope_top + scope_bot) * 0.5;
    let scope_amp = (scope_top - scope_bot) * 0.5 * 0.92;

    // zero line
    draw.line()
        .start(pt2(win.left() + 20.0, scope_mid))
        .end(pt2(win.right() - 20.0, scope_mid))
        .weight(1.0)
        .color(Color::srgb(0.15, 0.15, 0.20));

    let n = model.scope.len();
    let waveform = (0..n).map(|i| {
        let x = map_range(i, 0, n - 1, win.left() + 20.0, win.right() - 20.0);
        let y = scope_mid + model.scope[i].clamp(-1.0, 1.0) * scope_amp;
        (pt2(x, y), Color::srgb(0.35, 0.95, 0.65))
    });
    draw.polyline().weight(1.6).points_colored(waveform);

    // ── Phase portrait of the labial oscillator (delay embedding) ──
    let pp_cx = win.left() + win.w() * 0.18;
    let pp_cy = win.bottom() + win.h() * 0.20;
    let pp_r = win.h() * 0.16;
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
    // Text (labels, parameter readout) is drawn via egui in `update`, since
    // draw.text() is unimplemented in this nannou fork.
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
