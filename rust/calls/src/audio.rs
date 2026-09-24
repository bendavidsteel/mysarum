//! Live output: the syrinx runs in the cpal callback (at `RENDER_SR`, resampled), driven either by the
//! current CPG gesture (looped or one-shot) or by the XY pad.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use ringbuf::traits::{Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};

use crate::RENDER_SR;
use crate::gesture::{ALPHA_MIN, Gesture};
use crate::params::{Phys, default_phys};
use crate::synth::{Instrument, Noise, Syrinx};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Call,
    Pad,
}

/// Written by the UI, read (via `try_lock`) by the audio thread.
pub struct Control {
    pub phys: Phys,
    /// Gesture at `RENDER_SR`.
    pub gesture: Arc<Gesture>,
    /// 1 / peak of the offline render, so the live call matches the saved wav.
    pub call_gain: f32,
    pub source: Source,
    pub pad: (f64, f64),
    pub pad_on: bool,
    pub playing: bool,
    pub looping: bool,
    pub volume: f32,
    /// Noise seed, shared with the offline render so the two match.
    pub seed: u64,
    /// Bumped to restart the call from the top.
    pub restart: u64,
    /// Bumped whenever `phys` or `gesture` change.
    pub version: u64,
}

/// Published by the audio thread.
#[derive(Default)]
pub struct Status {
    pub position: AtomicUsize,
    pub active: AtomicBool,
}

pub struct Audio {
    pub control: Arc<Mutex<Control>>,
    pub status: Arc<Status>,
    pub sample_rate: f64,
    pub scope: HeapCons<f32>,
    _stream: cpal::Stream,
}

#[derive(Clone, Copy)]
struct Snap {
    source: Source,
    pad: (f64, f64),
    pad_on: bool,
    looping: bool,
    call_gain: f32,
    volume: f32,
}

struct Engine {
    snap: Snap,
    control: Arc<Mutex<Control>>,
    status: Arc<Status>,
    sr: f64,
    ratio: f64,
    frac: f64,
    prev: f64,
    cur: f64,
    version: u64,
    restart: u64,
    ins: Instrument,
    gesture: Arc<Gesture>,
    syr: Syrinx,
    noise: Noise,
    seed: u64,
    pos: usize,
    active: bool,
    alpha: f64,
    beta: f64,
    smooth: f64,
    peak: f64,
    peak_decay: f64,
    scope: HeapProd<f32>,
}

impl Engine {
    fn start_call(&mut self) {
        let g = &self.gesture;
        let (a0, b0) = (*g.alpha.first().unwrap_or(&0.0) as f64, *g.beta.first().unwrap_or(&0.0) as f64);
        self.syr.start_call(&self.ins, a0, b0, self.sr);
        self.noise = Noise::new(self.seed);
    }

    /// Pull UI changes; on contention keep last buffer's settings.
    fn sync(&mut self) -> Snap {
        let control = self.control.clone();
        let Ok(c) = control.try_lock() else {
            return self.snap;
        };
        if c.version != self.version {
            self.version = c.version;
            self.ins = Instrument::new(&c.phys, self.sr);
            self.gesture = c.gesture.clone();
        }
        self.seed = c.seed;
        if c.restart != self.restart {
            self.restart = c.restart;
            self.pos = 0;
            self.active = true;
            self.start_call();
        }
        if !c.playing {
            self.active = false;
        }
        self.snap = Snap {
            source: c.source,
            pad: c.pad,
            pad_on: c.pad_on,
            looping: c.looping,
            call_gain: c.call_gain,
            volume: c.volume,
        };
        self.snap
    }

    /// One sample at the synthesis rate.
    fn next(&mut self, snap: &Snap) -> f64 {
        let n = self.gesture.alpha.len();
        // Also catches a shorter gesture arriving mid-call.
        if n > 0 && self.pos >= n && self.active {
            if snap.looping {
                self.pos = 0;
                if snap.source == Source::Call {
                    self.start_call()
                }
            } else {
                self.active = false
            }
        }
        let (ta, tb) = match snap.source {
            Source::Pad if snap.pad_on => snap.pad,
            Source::Call if self.active && self.pos < n => {
                let p = self.pos;
                self.pos += 1;
                (self.gesture.alpha[p] as f64, self.gesture.beta[p] as f64)
            }
            _ => (ALPHA_MIN, self.beta),
        };
        // The call is already smooth; only the pad needs de-zippering.
        if snap.source == Source::Pad {
            self.alpha += self.smooth * (ta - self.alpha);
            self.beta += self.smooth * (tb - self.beta);
        } else {
            (self.alpha, self.beta) = (ta, tb);
        }

        let raw = self.syr.step(&self.ins, self.alpha, self.beta, self.noise.next());
        self.peak = (self.peak * self.peak_decay).max(raw.abs());
        let gain = match snap.source {
            Source::Call => snap.call_gain as f64,
            Source::Pad => 1.0 / self.peak.max(1e-3),
        };
        (raw * gain).clamp(-1.0, 1.0) * snap.volume as f64
    }

    /// Synthesise at `RENDER_SR` (the model's dt depends on the rate) and
    /// linearly resample to the device.
    fn fill<T: SizedSample + FromSample<f32>>(&mut self, out: &mut [T], channels: usize) {
        let snap = self.sync();
        for frame in out.chunks_mut(channels) {
            self.frac += self.ratio;
            while self.frac >= 1.0 {
                self.frac -= 1.0;
                self.prev = self.cur;
                self.cur = self.next(&snap);
            }
            let v = (self.prev + (self.cur - self.prev) * self.frac) as f32;
            let _ = self.scope.try_push(v);
            let s = T::from_sample(v);
            frame.iter_mut().for_each(|c| *c = s);
        }
        self.status.position.store(self.pos, Ordering::Relaxed);
        self.status.active.store(self.active, Ordering::Relaxed);
    }
}

impl Audio {
    pub fn start() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or("no audio output device")?;
        // Prefer running at the synthesis rate; linear resampling aliases.
        let want = cpal::SampleRate(RENDER_SR as u32);
        let config = device
            .supported_output_configs()
            .ok()
            .and_then(|mut cs| cs.find(|c| c.min_sample_rate() <= want && want <= c.max_sample_rate() && c.sample_format() == cpal::SampleFormat::F32))
            .map(|c| c.with_sample_rate(want))
            .map_or_else(|| device.default_output_config().map_err(|e| e.to_string()), Ok)?;
        let sr = config.sample_rate().0 as f64;
        let channels = config.channels() as usize;

        let control = Arc::new(Mutex::new(Control {
            phys: default_phys(),
            gesture: Arc::new(Gesture { alpha: vec![ALPHA_MIN as f32], beta: vec![0.0] }),
            call_gain: 1.0,
            source: Source::Call,
            pad: (ALPHA_MIN, 0.0),
            pad_on: false,
            playing: false,
            looping: true,
            volume: 0.5,
            seed: 1,
            restart: 0,
            version: 0,
        }));
        let status = Arc::new(Status::default());
        let (prod, cons) = HeapRb::<f32>::new(1 << 16).split();
        let mut engine = Engine {
            snap: Snap { source: Source::Call, pad: (ALPHA_MIN, 0.0), pad_on: false, looping: true, call_gain: 1.0, volume: 0.0 },
            control: control.clone(),
            status: status.clone(),
            sr: RENDER_SR,
            ratio: RENDER_SR / sr,
            frac: 0.0,
            prev: 0.0,
            cur: 0.0,
            version: u64::MAX,
            restart: 0,
            ins: Instrument::new(&default_phys(), RENDER_SR),
            gesture: Arc::new(Gesture { alpha: vec![], beta: vec![] }),
            syr: Syrinx::new(RENDER_SR),
            noise: Noise::new(1),
            seed: 1,
            pos: 0,
            active: false,
            alpha: ALPHA_MIN,
            beta: 0.0,
            smooth: 1.0 - (-1.0 / (0.01 * RENDER_SR)).exp(),
            peak: 0.0,
            peak_decay: (-1.0 / (1.5 * RENDER_SR)).exp(),
            scope: prod,
        };

        engine.start_call();
        let err = |e| eprintln!("audio stream error: {e}");
        let cfg: cpal::StreamConfig = config.clone().into();
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream(&cfg, move |d: &mut [f32], _| engine.fill(d, channels), err, None),
            cpal::SampleFormat::I16 => device.build_output_stream(&cfg, move |d: &mut [i16], _| engine.fill(d, channels), err, None),
            cpal::SampleFormat::U16 => device.build_output_stream(&cfg, move |d: &mut [u16], _| engine.fill(d, channels), err, None),
            f => return Err(format!("unsupported sample format {f:?}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;

        Ok(Audio { control, status, sample_rate: sr, scope: cons, _stream: stream })
    }
}
