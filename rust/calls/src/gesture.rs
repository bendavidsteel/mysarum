//! Motor score from the forced Wilson–Cowan CPG — a port of
//! `callsong/gestures.py` (`integrate`, `readout`, `_to_audio`).

use crate::params::{N_INSTRUMENT, Phys};

pub const ALPHA_MIN: f64 = -0.05;
pub const ALPHA_MAX: f64 = 0.30;
pub const ALPHA_ONSET: f64 = 0.02;
pub const BETA_MIN: f64 = 0.0;
pub const BETA_MAX: f64 = 2.6;

const A_EE: f64 = 10.0;
const A_EI: f64 = -10.0;
const A_IE: f64 = 10.0;
const A_II: f64 = 2.0;
const CTRL_SR: f64 = 4_000.0;
const PHRASE_LO: f64 = 1.0 / 3.0;
const PHRASE_HI: f64 = 2.0 / 3.0;
const PHRASE_ATTACK_S: f64 = 0.03;
const PHRASE_DECAY_S: f64 = 0.15;

pub struct Gesture {
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
}

struct Cpg {
    rho_e: f64,
    rho_i: f64,
    rate_e: f64,
    rate_i: f64,
    f_clock: f64,
    amp: f64,
    duty: f64,
    lag_ms: f64,
    w_e: f64,
    w_i: f64,
    b_bias: f64,
    b_ramp: f64,
    a_gain: f64,
    a_bias: f64,
}

impl Cpg {
    fn new(phys: &Phys) -> Self {
        let q = &phys[N_INSTRUMENT..];
        Cpg {
            rho_e: q[0],
            rho_i: q[1],
            rate_e: q[2],
            rate_i: q[3],
            f_clock: q[4],
            amp: q[5],
            duty: q[6],
            lag_ms: q[7],
            w_e: q[8],
            w_i: q[9],
            b_bias: q[10],
            b_ramp: q[11],
            a_gain: q[12],
            a_bias: q[13],
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-60.0, 60.0)).exp())
}

/// The expiratory pulse: one phrase in the middle of the window.
fn phrase_gate(t: f64, duration: f64) -> f64 {
    let (t0, t1) = (PHRASE_LO * duration, PHRASE_HI * duration);
    let atk = PHRASE_ATTACK_S.min(0.4 * (t1 - t0));
    let dec = PHRASE_DECAY_S.min(0.4 * (t1 - t0));
    let pi = std::f64::consts::PI;
    if t <= t0 || t >= t1 {
        0.0
    } else if t < t0 + atk {
        0.5 * (1.0 - (pi * (t - t0) / atk).cos())
    } else if t > t1 - dec {
        0.5 * (1.0 - (pi * (t1 - t) / dec).cos())
    } else {
        1.0
    }
}

/// HVC burst clock, gated to the phrase.
fn forcing(t: f64, c: &Cpg, duration: f64) -> f64 {
    let gate = phrase_gate(t, duration);
    if gate == 0.0 {
        return 0.0;
    }
    let phase = (t * c.f_clock).rem_euclid(1.0);
    if phase < c.duty {
        let u = phase / c.duty;
        gate * c.amp * 0.5 * (1.0 - (2.0 * std::f64::consts::PI * u).cos())
    } else {
        0.0
    }
}

fn integrate(c: &Cpg, n_steps: usize) -> (Vec<f64>, Vec<f64>) {
    let dt = 1.0 / CTRL_SR;
    let duration = n_steps as f64 / CTRL_SR;
    let rhs = |e: f64, i: f64, f: f64| {
        (
            c.rate_e * (-e + sigmoid(c.rho_e + A_EE * e + A_EI * i + f)),
            c.rate_i * (-i + sigmoid(c.rho_i + A_IE * e + A_II * i)),
        )
    };
    let (mut e, mut i) = (0.0, 0.0);
    let mut out_e = Vec::with_capacity(n_steps);
    let mut out_i = Vec::with_capacity(n_steps);
    let mut f0 = forcing(0.0, c, duration);
    for k in 0..n_steps {
        out_e.push(e);
        out_i.push(i);
        let t = k as f64 * dt;
        let f1 = forcing(t + 0.5 * dt, c, duration);
        let f2 = forcing(t + dt, c, duration);
        let (k1e, k1i) = rhs(e, i, f0);
        let (k2e, k2i) = rhs(e + 0.5 * dt * k1e, i + 0.5 * dt * k1i, f1);
        let (k3e, k3i) = rhs(e + 0.5 * dt * k2e, i + 0.5 * dt * k2i, f1);
        let (k4e, k4i) = rhs(e + dt * k3e, i + dt * k3i, f2);
        e += dt / 6.0 * (k1e + 2.0 * k2e + 2.0 * k3e + k4e);
        i += dt / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i);
        f0 = f2;
    }
    (out_e, out_i)
}

/// CPG activity → (alpha, beta) at the control rate.
fn readout(e: &[f64], i: &[f64], c: &Cpg) -> (Vec<f64>, Vec<f64>) {
    let k = e.len();
    let shift = (c.lag_ms * 1e-3 * CTRL_SR).round() as isize;
    let e_lag: Vec<f64> = (0..k as isize)
        .map(|n| e[(n - shift).clamp(0, k as isize - 1) as usize])
        .collect();
    let duration = k as f64 / CTRL_SR;
    let mut alpha = Vec::with_capacity(k);
    let mut beta = Vec::with_capacity(k);
    for n in 0..k {
        let env = phrase_gate(n as f64 / CTRL_SR, duration);
        let ramp = if k > 1 { n as f64 / (k - 1) as f64 } else { 0.0 };
        let a_unit = env * (c.a_bias + c.a_gain * e[n]).clamp(0.0, 1.0);
        let b_unit = env * (c.b_bias + c.w_e * e_lag[n] + c.w_i * i[n] + c.b_ramp * ramp).clamp(0.0, 1.0);
        alpha.push(ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * a_unit);
        beta.push(BETA_MIN + (BETA_MAX - BETA_MIN) * b_unit);
    }
    (alpha, beta)
}

/// Control rate → audio rate: linear interpolation, then a short Hann smoother.
fn to_audio(x: &[f64], n_samples: usize, sr: f64) -> Vec<f32> {
    let k = x.len();
    let interp: Vec<f64> = (0..n_samples)
        .map(|n| {
            let pos = n as f64 / sr * CTRL_SR;
            let j = pos.floor() as usize;
            if j + 1 >= k {
                x[k - 1]
            } else {
                let f = pos - j as f64;
                x[j] * (1.0 - f) + x[j + 1] * f
            }
        })
        .collect();

    let win = ((2.0 * sr / CTRL_SR) as usize | 1).max(3);
    let m = win + 2;
    let mut kern: Vec<f64> = (1..=win)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (m - 1) as f64).cos())
        .collect();
    let s: f64 = kern.iter().sum();
    kern.iter_mut().for_each(|v| *v /= s);
    let pad = win / 2;
    (0..n_samples)
        .map(|n| {
            let mut acc = 0.0;
            for (j, w) in kern.iter().enumerate() {
                let idx = (n + j).saturating_sub(pad).min(n_samples - 1);
                acc += w * interp[idx];
            }
            acc as f32
        })
        .collect()
}

pub fn from_phys(phys: &Phys, duration: f64, sr: f64) -> Gesture {
    let n_samples = (duration * sr).round().max(1.0) as usize;
    let n_steps = ((n_samples as f64 / sr * CTRL_SR).round() as usize).max(2);
    let c = Cpg::new(phys);
    let (e, i) = integrate(&c, n_steps);
    let (a, b) = readout(&e, &i, &c);
    Gesture { alpha: to_audio(&a, n_samples, sr), beta: to_audio(&b, n_samples, sr) }
}
