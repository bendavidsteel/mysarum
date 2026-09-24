//! Log-frequency magnitude spectrograms, rendered to egui images.

use std::sync::Arc;

use bevy_egui::egui::{Color32, ColorImage};
use rustfft::{Fft, FftPlanner, num_complex::Complex};

pub const N_FFT: usize = 1024;
pub const HOP: usize = 256;
pub const N_BINS: usize = 192;
pub const F_LO: f64 = 100.0;
pub const F_HI: f64 = 10_000.0;
const DB_RANGE: f32 = 70.0;

pub struct Analyser {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// Fractional FFT bin for each output row, bottom row first.
    rows: Vec<f64>,
    buf: Vec<Complex<f32>>,
}

impl Analyser {
    pub fn new(sr: f64) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(N_FFT);
        let window = (0..N_FFT)
            .map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        let rows = (0..N_BINS)
            .map(|r| {
                let f = F_LO * (F_HI / F_LO).powf(r as f64 / (N_BINS - 1) as f64);
                f / sr * N_FFT as f64
            })
            .collect();
        Analyser { fft, window, rows, buf: vec![Complex::default(); N_FFT] }
    }

    /// dB magnitude per row for one frame (`frame.len() == N_FFT`).
    pub fn column(&mut self, frame: &[f32]) -> [f32; N_BINS] {
        for (b, (&x, &w)) in self.buf.iter_mut().zip(frame.iter().zip(&self.window)) {
            *b = Complex::new(x * w, 0.0);
        }
        self.fft.process(&mut self.buf);
        let mag = |k: usize| self.buf[k.min(N_FFT / 2)].norm();
        std::array::from_fn(|r| {
            let pos = self.rows[r];
            let k = pos.floor() as usize;
            let f = (pos - k as f64) as f32;
            let m = mag(k) * (1.0 - f) + mag(k + 1) * f;
            20.0 * (m + 1e-9).log10()
        })
    }

    /// Whole-call spectrogram, normalised to its own peak.
    pub fn image(&mut self, wave: &[f32]) -> ColorImage {
        let n_cols = (wave.len().saturating_sub(N_FFT) / HOP + 1).max(1);
        let mut frame = vec![0.0; N_FFT];
        let cols: Vec<[f32; N_BINS]> = (0..n_cols)
            .map(|c| {
                let s = c * HOP;
                frame.iter_mut().enumerate().for_each(|(i, v)| *v = wave.get(s + i).copied().unwrap_or(0.0));
                self.column(&frame)
            })
            .collect();
        let top = cols.iter().flatten().fold(f32::MIN, |m, &v| m.max(v));
        let mut img = ColorImage::new([n_cols, N_BINS], vec![Color32::BLACK; n_cols * N_BINS]);
        for (c, col) in cols.iter().enumerate() {
            for (r, &db) in col.iter().enumerate() {
                img.pixels[(N_BINS - 1 - r) * n_cols + c] = magma((db - top + DB_RANGE) / DB_RANGE);
            }
        }
        img
    }
}

/// Scrolling live spectrogram fed from the audio scope.
pub struct Live {
    analyser: Analyser,
    pending: Vec<f32>,
    pub cols: usize,
    pub image: ColorImage,
}

impl Live {
    pub fn new(sr: f64, cols: usize) -> Self {
        Live {
            analyser: Analyser::new(sr),
            pending: Vec::new(),
            cols,
            image: ColorImage::new([cols, N_BINS], vec![Color32::BLACK; cols * N_BINS]),
        }
    }

    pub fn push(&mut self, samples: impl Iterator<Item = f32>) {
        self.pending.extend(samples);
        while self.pending.len() >= N_FFT {
            let col = self.analyser.column(&self.pending[..N_FFT]);
            self.pending.drain(..HOP * 2);
            let w = self.cols;
            for r in 0..N_BINS {
                let row = &mut self.image.pixels[(N_BINS - 1 - r) * w..(N_BINS - r) * w];
                row.copy_within(1.., 0);
                // Fixed reference: the live output is already gain-normalised.
                row[w - 1] = magma((col[r] + DB_RANGE - 45.0) / DB_RANGE);
            }
        }
    }
}

pub fn magma(t: f32) -> Color32 {
    const STOPS: [(f32, [f32; 3]); 6] = [
        (0.0, [0.0, 0.0, 0.02]),
        (0.2, [0.16, 0.07, 0.38]),
        (0.4, [0.45, 0.12, 0.51]),
        (0.6, [0.74, 0.22, 0.45]),
        (0.8, [0.98, 0.50, 0.37]),
        (1.0, [0.99, 0.99, 0.75]),
    ];
    let t = t.clamp(0.0, 1.0);
    let i = STOPS.iter().position(|s| s.0 >= t).unwrap_or(5).max(1);
    let (t0, c0) = STOPS[i - 1];
    let (t1, c1) = STOPS[i];
    let f = (t - t0) / (t1 - t0);
    let ch = |k: usize| ((c0[k] + (c1[k] - c0[k]) * f) * 255.0) as u8;
    Color32::from_rgb(ch(0), ch(1), ch(2))
}

/// 16-bit mono PCM WAV, peak-normalised to 0.95 like `audio_io.save_wav`.
pub fn write_wav(path: &std::path::Path, wave: &[f32], sr: u32) -> std::io::Result<()> {
    let peak = wave.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if peak > 1e-6 { 0.95 / peak } else { 1.0 };
    let data_len = (wave.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&(sr * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &v in wave {
        out.extend_from_slice(&(((v * scale).clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, out)
}

