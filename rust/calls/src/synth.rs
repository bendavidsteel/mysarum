//! Syrinx synthesiser, a sample-by-sample port of `callsong/synth.py`:
//! generalised labial oscillator (RK4) → trachea reflection comb → two
//! state-variable formants → tanh. Streaming, so the audio thread can run it
//! live; `render` wraps it for whole calls with the Python warmup + peak norm.

use crate::params::{N_INSTRUMENT, Phys};

pub const OVERSAMPLE: usize = 6;
const NOISE_FLOW_REF: f64 = 0.05;
const WARMUP_S: f64 = 0.1;
const TRACHEA_MS_MAX: f64 = 4.0;
const X_CLAMP: f64 = 3.0;
const Y_CLAMP_FACTOR: f64 = 3.0;

/// Instrument parameters in physical units, plus derived filter coefficients.
#[derive(Clone, Copy)]
pub struct Instrument {
    gamma: f64,
    k_cub: f64,
    k_sq: f64,
    k_sqy: f64,
    k_xy: f64,
    k_vdp: f64,
    delay: usize,
    r: f64,
    f1: f64,
    d1: f64,
    g1: f64,
    f2: f64,
    d2: f64,
    g2: f64,
    drive: f64,
    noise_gain: f64,
}

impl Instrument {
    pub fn new(phys: &Phys, sr: f64) -> Self {
        let p = &phys[..N_INSTRUMENT];
        let sr_os = sr * OVERSAMPLE as f64;
        let len = trachea_len(sr);
        let svf = |hz: f64| 2.0 * (std::f64::consts::PI * hz / sr_os).sin();
        Instrument {
            gamma: p[0],
            k_cub: p[1],
            k_sq: p[2],
            k_sqy: p[3],
            k_xy: p[4],
            k_vdp: p[5],
            delay: ((p[6] * 1e-3 * sr_os).round() as usize).clamp(1, len - 1),
            r: p[7],
            f1: svf(p[8]),
            d1: 1.0 / p[9],
            g1: p[10],
            f2: svf(p[11]),
            d2: 1.0 / p[12],
            g2: p[13],
            drive: p[14],
            noise_gain: p[15],
        }
    }
}

fn trachea_len(sr: f64) -> usize {
    (TRACHEA_MS_MAX * 1e-3 * sr * OVERSAMPLE as f64).ceil() as usize + 1
}

pub struct Syrinx {
    dt: f64,
    x: f64,
    y: f64,
    buf: Vec<f64>,
    pos: usize,
    lo1: f64,
    ba1: f64,
    lo2: f64,
    ba2: f64,
}

impl Syrinx {
    pub fn new(sr: f64) -> Self {
        Syrinx {
            dt: 1.0 / (sr * OVERSAMPLE as f64),
            x: 0.01,
            y: 0.0,
            buf: vec![0.0; trachea_len(sr)],
            pos: 0,
            lo1: 0.0,
            ba1: 0.0,
            lo2: 0.0,
            ba2: 0.0,
        }
    }

    /// Back to the initial state, then hold the first gesture value for the
    /// warmup so the call starts at rest (as `synth.render_one`).
    pub fn start_call(&mut self, ins: &Instrument, a0: f64, b0: f64, sr: f64) {
        (self.x, self.y, self.pos) = (0.01, 0.0, 0);
        (self.lo1, self.ba1, self.lo2, self.ba2) = (0.0, 0.0, 0.0, 0.0);
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        for _ in 0..(WARMUP_S * sr).round() as usize {
            self.step(ins, a0, b0, 0.0);
        }
    }

    /// One audio frame: `OVERSAMPLE` substeps at fixed (alpha, beta, noise).
    #[inline]
    pub fn step(&mut self, ins: &Instrument, a: f64, b: f64, nz: f64) -> f64 {
        let dt = self.dt;
        let g = ins.gamma;
        let g2 = g * g;
        let y_clamp = Y_CLAMP_FACTOR * g;
        let deriv = |x: f64, y: f64| {
            let dy = g2 * (-a - b * x - ins.k_cub * x * x * x + ins.k_sq * x * x)
                + g * (-ins.k_sqy * x * x * y - ins.k_xy * x * y + ins.k_vdp * (1.0 - x * x) * y);
            (y, dy)
        };
        let flow = (a / NOISE_FLOW_REF).clamp(0.0, 1.0);
        let len = self.buf.len();
        let mut acc = 0.0;
        for _ in 0..OVERSAMPLE {
            let (x, y) = (self.x, self.y);
            let (k1x, k1y) = deriv(x, y);
            let (k2x, k2y) = deriv(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
            let (k3x, k3y) = deriv(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
            let (k4x, k4y) = deriv(x + dt * k3x, y + dt * k3y);
            self.x = (x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)).clamp(-X_CLAMP, X_CLAMP);
            self.y = (y + dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)).clamp(-y_clamp, y_clamp);
            // A blown-up state (NaN) would otherwise stay NaN forever.
            if !self.x.is_finite() || !self.y.is_finite() {
                self.x = 0.01;
                self.y = 0.0;
            }

            let src = self.x + ins.noise_gain * flow * nz;
            let read = if self.pos >= ins.delay { self.pos - ins.delay } else { self.pos + len - ins.delay };
            let pi = src - ins.r * self.buf[read];
            self.buf[self.pos] = pi;
            self.pos = if self.pos + 1 == len { 0 } else { self.pos + 1 };

            self.lo1 += ins.f1 * self.ba1;
            let hi1 = pi - self.lo1 - ins.d1 * self.ba1;
            self.ba1 += ins.f1 * hi1;
            self.lo2 += ins.f2 * self.ba2;
            let hi2 = pi - self.lo2 - ins.d2 * self.ba2;
            self.ba2 += ins.f2 * hi2;
            if !self.ba1.is_finite() || !self.ba2.is_finite() {
                (self.lo1, self.ba1, self.lo2, self.ba2) = (0.0, 0.0, 0.0, 0.0);
            }

            acc += ins.g1 * self.ba1 + ins.g2 * self.ba2;
        }
        (ins.drive * acc / OVERSAMPLE as f64).tanh()
    }
}

/// xorshift64* white noise in [-1, 1].
pub struct Noise(u64);

impl Noise {
    pub fn new(seed: u64) -> Self {
        Noise(seed.max(1))
    }
    #[inline]
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Whole call, as `synth.render_one`: warmup held at the first gesture value,
/// discarded, then peak-normalised. Returns (wave, pre-normalisation peak).
pub fn render(phys: &Phys, alpha: &[f32], beta: &[f32], sr: f64, seed: u64) -> (Vec<f32>, f32) {
    let ins = Instrument::new(phys, sr);
    let mut syr = Syrinx::new(sr);
    let mut noise = Noise::new(seed);
    syr.start_call(&ins, *alpha.first().unwrap_or(&0.0) as f64, *beta.first().unwrap_or(&0.0) as f64, sr);
    let mut wave: Vec<f32> = alpha
        .iter()
        .zip(beta)
        .map(|(&a, &b)| syr.step(&ins, a as f64, b as f64, noise.next()) as f32)
        .collect();
    let peak = wave.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak > 1e-4 {
        wave.iter_mut().for_each(|v| *v /= peak);
    }
    (wave, peak)
}
