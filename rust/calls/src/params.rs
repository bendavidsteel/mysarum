//! The 30-parameter genome: 16 instrument + 14 CPG gesture, mirroring
//! `python/calls/callsong/genome.py` and `gestures.py` (same order, bounds and
//! defaults, so genomes round-trip with the Python archives).

use serde::{Deserialize, Serialize};

pub struct Spec {
    pub name: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub default: f64,
    pub help: &'static str,
    /// Slider on a log scale (frequencies, rates, time constants).
    pub log: bool,
}

const fn p(name: &'static str, lo: f64, hi: f64, default: f64, log: bool, help: &'static str) -> Spec {
    Spec { name, lo, hi, default, help, log }
}

const fn g(name: &'static str, lo: f64, hi: f64, log: bool, help: &'static str) -> Spec {
    Spec { name, lo, hi, default: 0.5 * (lo + hi), help, log }
}

pub const N_INSTRUMENT: usize = 16;
pub const N_GESTURE: usize = 14;
pub const N_PARAMS: usize = N_INSTRUMENT + N_GESTURE;

pub static SPEC: [Spec; N_PARAMS] = [
    // ── source (labial oscillator) ──
    p("gamma", 8_000.0, 60_000.0, 24_000.0, true, "time constant / spectral range"),
    p("k_cub", 0.2, 3.0, 1.0, false, "cubic restoring (-g² x³)"),
    p("k_sq", -1.0, 2.0, 1.0, false, "quadratic asymmetry (+g² x²)"),
    p("k_sqy", 0.0, 3.0, 1.0, false, "x² y nonlinear damping"),
    p("k_xy", 0.0, 3.0, 1.0, false, "x y nonlinear damping"),
    p("k_vdp", 0.0, 1.5, 0.0, false, "van der Pol negative damping"),
    // ── trachea reflection comb ──
    p("trachea_ms", 0.05, 4.0, 0.25, true, "round-trip delay T (ms)"),
    p("r", 0.0, 0.95, 0.75, false, "beak reflection (comb depth)"),
    // ── formant bank ──
    p("f1_hz", 300.0, 6_000.0, 2_200.0, true, "formant 1 centre"),
    p("q1", 1.0, 20.0, 2.0, true, "formant 1 Q"),
    p("g1", 0.0, 1.0, 1.0, false, "formant 1 gain"),
    p("f2_hz", 300.0, 8_000.0, 4_000.0, true, "formant 2 centre"),
    p("q2", 1.0, 20.0, 3.0, true, "formant 2 Q"),
    p("g2", 0.0, 1.0, 0.3, false, "formant 2 gain"),
    // ── output shaping ──
    p("drive", 0.5, 6.0, 2.0, false, "tanh saturation drive"),
    p("noise_gain", 0.0, 0.3, 0.02, false, "aspiration noise into source"),
    // ── CPG gesture (Wilson–Cowan pair + HVC clock + readout) ──
    g("rho_e", -4.0, 4.0, false, "standing drive to the excitatory population"),
    g("rho_i", -11.0, -1.0, false, "standing drive to the inhibitory population"),
    g("rate_e", 40.0, 600.0, true, "1/tau_e (s⁻¹)"),
    g("rate_i", 10.0, 300.0, true, "1/tau_i (s⁻¹)"),
    g("f_clock", 1.0, 60.0, true, "forcing rate (Hz) — syllable clock"),
    g("amp", 0.0, 8.0, false, "forcing amplitude (0 = autonomous CPG)"),
    g("duty", 0.05, 0.9, false, "forcing burst width, fraction of period"),
    g("lag_ms", -20.0, 20.0, false, "beta lag behind alpha; sign = sweep direction"),
    g("w_e", -1.2, 1.2, false, "beta ← lagged excitatory activity"),
    g("w_i", -1.2, 1.2, false, "beta ← inhibitory activity"),
    g("b_bias", -0.3, 1.0, false, "beta offset (mean tension)"),
    g("b_ramp", -0.8, 0.8, false, "slow tension drift across the phrase"),
    g("a_gain", 0.6, 1.8, false, "alpha ← excitatory activity"),
    g("a_bias", -0.7, 0.15, false, "alpha offset (how far above/below onset)"),
];

/// Section headers in the UI: (title, first index, end index).
pub const GROUPS: [(&str, usize, usize); 5] = [
    ("Source (labial oscillator)", 0, 6),
    ("Trachea", 6, 8),
    ("Formants", 8, 14),
    ("Output", 14, 16),
    ("Gesture (vocal CPG)", 16, 30),
];

/// Physical-unit parameter vector.
pub type Phys = [f64; N_PARAMS];

pub fn default_phys() -> Phys {
    std::array::from_fn(|i| SPEC[i].default)
}

pub fn decode(genome: &[f32]) -> Phys {
    std::array::from_fn(|i| {
        let u = genome.get(i).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f64;
        SPEC[i].lo + u * (SPEC[i].hi - SPEC[i].lo)
    })
}

pub fn encode(phys: &Phys) -> [f64; N_PARAMS] {
    std::array::from_fn(|i| ((phys[i] - SPEC[i].lo) / (SPEC[i].hi - SPEC[i].lo)).clamp(0.0, 1.0))
}

pub fn from_unit(i: usize, u: f64) -> f64 {
    SPEC[i].lo + u.clamp(0.0, 1.0) * (SPEC[i].hi - SPEC[i].lo)
}

/// Saved sound: physical parameters keyed by name so presets survive reordering.
#[derive(Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    #[serde(default)]
    pub notes: String,
    pub params: std::collections::BTreeMap<String, f64>,
}

impl Preset {
    pub fn from_phys(name: &str, notes: &str, phys: &Phys) -> Self {
        let params = SPEC.iter().zip(phys).map(|(s, &v)| (s.name.to_string(), v)).collect();
        Preset { name: name.to_string(), notes: notes.to_string(), params }
    }

    /// Missing names fall back to defaults; values are clamped into bounds.
    pub fn to_phys(&self) -> Phys {
        std::array::from_fn(|i| {
            let s = &SPEC[i];
            self.params.get(s.name).copied().unwrap_or(s.default).clamp(s.lo, s.hi)
        })
    }
}
