//! Interactive explorer for the `python/calls` syrinx + CPG call model.
//!
//! Every slider edits the 30-parameter genome. The instrument half applies to
//! the live audio stream immediately; the whole call (gesture + offline render
//! for the spectrogram, gain and wav export) is recomputed on a worker thread.

mod archive;
mod audio;
mod gesture;
mod params;
mod spectro;
mod synth;
mod ui;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use bevy::prelude::*;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass, egui};

use crate::audio::Audio;
use crate::gesture::Gesture;
use crate::params::{N_PARAMS, Phys, Preset, default_phys};

/// Offline renders use the Python pipeline's rate, so exported wavs match.
pub const RENDER_SR: f64 = 48_000.0;

struct Job {
    id: u64,
    phys: Phys,
    duration: f64,
    seed: u64,
}

pub struct Rendered {
    pub id: u64,
    pub phys: Phys,
    pub wave: Vec<f32>,
    pub peak: f32,
    pub gesture: Arc<Gesture>,
    pub image: egui::ColorImage,
    pub ms: f64,
}

fn worker(jobs: Receiver<Job>, done: Sender<Rendered>) {
    let mut analyser = spectro::Analyser::new(RENDER_SR);
    while let Ok(mut job) = jobs.recv() {
        // Only the newest request matters while a slider is being dragged.
        while let Ok(newer) = jobs.try_recv() {
            job = newer;
        }
        let t0 = std::time::Instant::now();
        let g = gesture::from_phys(&job.phys, job.duration, RENDER_SR);
        let (wave, peak) = synth::render(&job.phys, &g.alpha, &g.beta, RENDER_SR, job.seed);
        let image = analyser.image(&wave);
        let msg = Rendered {
            id: job.id,
            phys: job.phys,
            wave,
            peak,
            gesture: Arc::new(g),
            image,
            ms: t0.elapsed().as_secs_f64() * 1e3,
        };
        if done.send(msg).is_err() {
            break;
        }
    }
}

pub struct PresetEntry {
    pub path: PathBuf,
    pub preset: Preset,
}

pub struct State {
    pub audio: Audio,
    pub phys: Phys,
    pub locked: [bool; N_PARAMS],
    pub duration: f64,
    pub seed: u64,
    pub sigma: f64,
    pub rng: synth::Noise,

    pub history: Vec<Phys>,
    pub history_pos: usize,

    jobs: Sender<Job>,
    results: Receiver<Rendered>,
    next_job: u64,
    pub dirty: bool,
    pub rendered: Option<Rendered>,
    pub call_tex: Option<egui::TextureHandle>,
    pub live: spectro::Live,
    pub live_tex: Option<egui::TextureHandle>,

    pub preset_dir: PathBuf,
    pub export_dir: PathBuf,
    pub presets: Vec<PresetEntry>,
    pub preset_name: String,
    pub preset_notes: String,
    pub preset_filter: String,
    pub confirm_delete: Option<PathBuf>,

    pub archive_paths: Vec<PathBuf>,
    pub archive: Option<archive::Archive>,
    pub archive_cell: Option<(usize, usize)>,

    pub pad_hold: bool,
    pub status: String,
}

impl State {
    fn new(audio: Audio) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let (jobs, job_rx) = channel();
        let (done_tx, results) = channel();
        std::thread::spawn(move || worker(job_rx, done_tx));
        let sr = audio.sample_rate;
        let mut s = State {
            audio,
            phys: default_phys(),
            locked: [false; N_PARAMS],
            duration: 3.0,
            seed: 1,
            sigma: 0.12,
            rng: synth::Noise::new(
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(7),
            ),
            history: vec![default_phys()],
            history_pos: 0,
            jobs,
            results,
            next_job: 0,
            dirty: true,
            rendered: None,
            call_tex: None,
            live: spectro::Live::new(sr, 640),
            live_tex: None,
            preset_dir: root.join("presets"),
            export_dir: root.join("exports"),
            presets: Vec::new(),
            preset_name: "untitled".into(),
            preset_notes: String::new(),
            preset_filter: String::new(),
            confirm_delete: None,
            archive_paths: archive::discover(&root.join("../../python/calls/outputs")),
            archive: None,
            archive_cell: None,
            pad_hold: false,
            status: String::new(),
        };
        s.refresh_presets();
        s.push_control();
        s
    }

    /// Mark the genome changed: live instrument now, full call on the worker.
    pub fn changed(&mut self) {
        self.dirty = true;
        self.push_control();
    }

    fn push_control(&mut self) {
        let mut c = self.audio.control.lock().unwrap();
        c.phys = self.phys;
        c.version += 1;
    }

    /// Record the current genome as an undo step.
    pub fn commit(&mut self) {
        if self.history.get(self.history_pos) == Some(&self.phys) {
            return;
        }
        self.history.truncate(self.history_pos + 1);
        self.history.push(self.phys);
        self.history_pos = self.history.len() - 1;
    }

    pub fn set_phys(&mut self, phys: Phys) {
        self.phys = phys;
        self.changed();
        self.commit();
    }

    pub fn undo(&mut self) {
        if self.history_pos > 0 {
            self.history_pos -= 1;
            self.phys = self.history[self.history_pos];
            self.changed();
        }
    }

    pub fn redo(&mut self) {
        if self.history_pos + 1 < self.history.len() {
            self.history_pos += 1;
            self.phys = self.history[self.history_pos];
            self.changed();
        }
    }

    fn uniform(&mut self) -> f64 {
        0.5 * (self.rng.next() + 1.0)
    }

    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Uniform resample of the unlocked parameters in `range`.
    pub fn randomize(&mut self, range: std::ops::Range<usize>) {
        let mut p = self.phys;
        for i in range {
            if !self.locked[i] {
                p[i] = params::from_unit(i, self.uniform());
            }
        }
        self.set_phys(p);
    }

    /// Gaussian step in normalised space with reflecting bounds (as `genome.mutate`).
    pub fn mutate(&mut self) {
        let unit = params::encode(&self.phys);
        let mut p = self.phys;
        for i in 0..N_PARAMS {
            if !self.locked[i] {
                let g = (unit[i] + self.sigma * self.gauss()).abs();
                p[i] = params::from_unit(i, 1.0 - (1.0 - g).abs());
            }
        }
        self.set_phys(p);
    }

    pub fn play(&mut self) {
        let mut c = self.audio.control.lock().unwrap();
        c.source = audio::Source::Call;
        c.playing = true;
        c.restart += 1;
    }

    pub fn stop(&mut self) {
        self.audio.control.lock().unwrap().playing = false;
    }

    pub fn refresh_presets(&mut self) {
        let mut v: Vec<PresetEntry> = std::fs::read_dir(&self.preset_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .filter_map(|path| {
                let text = std::fs::read_to_string(&path).ok()?;
                let preset = serde_json::from_str(&text).ok()?;
                Some(PresetEntry { path, preset })
            })
            .collect();
        v.sort_by_key(|e| e.preset.name.to_lowercase());
        self.presets = v;
    }

    pub fn save_preset(&mut self) {
        let name = self.preset_name.trim();
        let slug: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c.to_ascii_lowercase() } else { '_' })
            .collect();
        if slug.is_empty() {
            self.status = "give the sound a name first".into();
            return;
        }
        let preset = Preset::from_phys(name, &self.preset_notes, &self.phys);
        let path = self.preset_dir.join(format!("{slug}.json"));
        let res = std::fs::create_dir_all(&self.preset_dir)
            .and_then(|_| std::fs::write(&path, serde_json::to_string_pretty(&preset).unwrap()));
        self.status = match res {
            Ok(()) => format!("saved {}", path.display()),
            Err(e) => format!("save failed: {e}"),
        };
        self.refresh_presets();
    }

    pub fn load_preset(&mut self, idx: usize) {
        let e = &self.presets[idx];
        let phys = e.preset.to_phys();
        self.preset_name = e.preset.name.clone();
        self.preset_notes = e.preset.notes.clone();
        self.status = format!("loaded {}", e.path.display());
        self.set_phys(phys);
        self.play();
    }

    pub fn export_wav(&mut self) {
        let Some(r) = &self.rendered else { return };
        let slug: String = self.preset_name.trim().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        let path = self.export_dir.join(format!("{}.wav", if slug.is_empty() { "call" } else { &slug }));
        let res = std::fs::create_dir_all(&self.export_dir)
            .and_then(|_| spectro::write_wav(&path, &r.wave, RENDER_SR as u32));
        self.status = match res {
            Ok(()) => format!("wrote {}", path.display()),
            Err(e) => format!("export failed: {e}"),
        };
    }

    pub fn load_archive(&mut self, path: PathBuf) {
        match archive::Archive::load(&path) {
            Ok(a) => {
                self.status = format!("archive {}×{}, {} params", a.res, a.res, a.n_params);
                self.archive = Some(a);
                self.archive_cell = None;
            }
            Err(e) => self.status = format!("archive load failed: {e}"),
        }
    }

    /// Load an elite. Older archives without the gesture half keep the current one.
    pub fn load_cell(&mut self, i: usize, j: usize) {
        let Some(a) = &self.archive else { return };
        let g = a.genome(i, j);
        let decoded = params::decode(g);
        let mut p = self.phys;
        p[..g.len().min(N_PARAMS)].copy_from_slice(&decoded[..g.len().min(N_PARAMS)]);
        self.archive_cell = Some((i, j));
        self.preset_name = format!("elite_{i}_{j}");
        self.status = format!("elite ({i}, {j}) fitness {:.3}", a.fitness(i, j));
        self.set_phys(p);
        self.play();
    }

    /// Submit dirty renders, apply finished ones, drain the live scope.
    fn tick(&mut self, ctx: &egui::Context) {
        if self.dirty {
            self.dirty = false;
            self.next_job += 1;
            let _ = self.jobs.send(Job {
                id: self.next_job,
                phys: self.phys,
                duration: self.duration,
                seed: self.seed,
            });
        }
        while let Ok(r) = self.results.try_recv() {
            {
                let mut c = self.audio.control.lock().unwrap();
                c.gesture = r.gesture.clone();
                c.call_gain = if r.peak > 1e-4 { 1.0 / r.peak } else { 1.0 };
                c.seed = self.seed;
                c.phys = self.phys;
                c.version += 1;
            }
            upload(ctx, &mut self.call_tex, "call", &r.image);
            self.rendered = Some(r);
        }

        use ringbuf::traits::Consumer;
        self.live.push(self.audio.scope.pop_iter());
        upload(ctx, &mut self.live_tex, "live", &self.live.image);
    }
}

/// Same-size updates go through `set_partial`: bevy_egui turns a full `set`
/// into a fresh image asset, which isn't on the GPU yet when the frame draws.
fn upload(ctx: &egui::Context, tex: &mut Option<egui::TextureHandle>, name: &str, image: &egui::ColorImage) {
    let opts = egui::TextureOptions::LINEAR;
    match tex {
        Some(t) if t.size() == image.size => t.set_partial([0, 0], image.clone(), opts),
        _ => *tex = Some(ctx.load_texture(name, image.clone(), opts)),
    }
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn ui_system(mut contexts: bevy_egui::EguiContexts, mut state: NonSendMut<State>) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    state.tick(&ctx);
    ui::draw(&ctx, &mut state);
    Ok(())
}

fn main() {
    let audio = Audio::start().unwrap_or_else(|e| panic!("could not open audio output: {e}"));
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "calls".into(),
                resolution: (1600, 950).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.06)))
        .insert_non_send_resource(State::new(audio))
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui_system)
        .run();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dumps a noise-free default call for comparison against the Python model.
    #[test]
    fn dump_reference() {
        let Ok(out) = std::env::var("CALLS_DUMP") else { return };
        let mut phys = default_phys();
        phys[15] = 0.0;
        let g = gesture::from_phys(&phys, 3.0, RENDER_SR);
        let (wave, _) = synth::render(&phys, &g.alpha, &g.beta, RENDER_SR, 1);
        let bytes: Vec<u8> = [g.alpha, g.beta, wave].concat().iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out, bytes).unwrap();
    }

    #[test]
    fn preset_roundtrip() {
        let mut phys = default_phys();
        phys[0] = 31_000.0;
        phys[20] = 12.5;
        let json = serde_json::to_string(&Preset::from_phys("x", "", &phys)).unwrap();
        let back: Preset = serde_json::from_str(&json).unwrap();
        assert_eq!(back.to_phys(), phys);
    }

    #[test]
    fn loads_python_archive() {
        let outputs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/calls/outputs");
        let Some(path) = archive::discover(&outputs).into_iter().next() else { return };
        let a = archive::Archive::load(&path).unwrap();
        assert_eq!(a.fitness.len(), a.res * a.res);
        assert!(a.fitness.iter().any(|f| f.is_finite()));
    }
}
