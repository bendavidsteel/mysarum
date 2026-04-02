#![allow(dead_code)]

mod barnes_hut;
mod gpu;
mod render;
mod reverb;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicI8, Ordering};

use nannou::prelude::*;
use serde::Deserialize;
use ringbuf::{traits::{Consumer, Producer, Split, Observer}, HeapRb};
use nannou_audio as audio;
use nannou_audio::Buffer;
use crossbeam_channel::{self, Receiver};

use barnes_hut::BarnesHut;
use gpu::{GpuCompute, SimParams, AudioParams, ModalFreqs, RenderUniforms, Particle,
          MAX_GRA_NODES, SAMPLE_RATE, CHUNK_SIZE, NUM_CHANNELS, CHUNK_FLOATS,
          NUM_AUDIO_STAGING_BUFS, NUM_MODAL_BANDS, MODAL_CHEB_ORDER, compute_bin_counts,
          create_gpu_compute, upload_particles, upload_gra_topology,
          upload_gra_forces, upload_gra_discrete_states, NodeTrialInfo,
          update_sim_params, update_render_uniforms,
          update_audio_params, upload_modal_coefficients, update_modal_freqs,
          dispatch_particle_bin_sort, dispatch_gra_bin_sort,
          dispatch_particle_sim, dispatch_gra_physics,
          encode_modal_chebyshev, encode_audio_pass,
          readback_gra_positions};
use render::RenderState;

// ── Constants ────────────────────────────────────────────────────────────────

const WORLD_HALF: f32 = 8.0;
const NUM_PARTICLES: u32 = 8192;

// Audio buffer management
const INITIAL_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 2;
const MIN_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS;
const MAX_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 4;
const BUFFER_HISTORY_SIZE: usize = 64;

// Modal synthesis
const MODAL_BASE_FREQ: f32 = 120.0;

// GRA trial management
const NUM_INITIAL_TRIALS: usize = 16;
const GRID_SPACING: f32 = 3.2;
const MIN_VIABLE_NODES: usize = 4;
const GRACE_FRAMES: u64 = 200;
const SPAWN_INTERVAL: u64 = 120;
const MAX_TRIALS: usize = 30;
const NUM_DISCRETE_STATES: u8 = 3;

// Particle spatial hash
const P_BIN_SIZE: f32 = 0.5;

// GRA spatial hash (for particle→GRA lookups)
const G_BIN_SIZE: f32 = 0.5;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn wrap_pos(x: f32, y: f32) -> (f32, f32) {
    let size = WORLD_HALF * 2.0;
    let wx = ((x + WORLD_HALF) % size + size) % size - WORLD_HALF;
    let wy = ((y + WORLD_HALF) % size + size) % size - WORLD_HALF;
    (wx, wy)
}

fn wrap_delta(a: f32, b: f32) -> f32 {
    let d = b - a;
    let size = WORLD_HALF * 2.0;
    if d > WORLD_HALF { d - size }
    else if d < -WORLD_HALF { d + size }
    else { d }
}

fn toroidal_avg(positions: &[(f32, f32)]) -> (f32, f32) {
    if positions.is_empty() { return (0.0, 0.0); }
    let (rx, ry) = positions[0];
    let n = positions.len() as f32;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for &(px, py) in positions {
        sx += rx + wrap_delta(rx, px);
        sy += ry + wrap_delta(ry, py);
    }
    wrap_pos(sx / n, sy / n)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// ── Discrete CA helpers ──────────────────────────────────────────────────────

fn num_discrete_configs(num_states: u8) -> usize {
    (num_states as usize) * 7
}

fn discrete_config(own_state: u8, neighbor_sum: u32) -> usize {
    let capped = neighbor_sum.min(6) as usize;
    (own_state as usize) * 7 + capped
}

fn random_rule_table(num_states: u8) -> Vec<u8> {
    let n = num_discrete_configs(num_states);
    (0..n).map(|_| random_range(0u8, num_states)).collect()
}

fn random_topo_table(num_states: u8) -> Vec<u8> {
    let n = num_discrete_configs(num_states);
    let num_actions = if num_states <= 2 { 2 } else { 3 };
    (0..n).map(|_| random_range(0u8, num_actions)).collect()
}

fn canonical_topo_action(raw: u8, num_states: u8) -> u8 {
    if num_states <= 2 {
        if raw == 0 { 1 } else { 2 }
    } else {
        raw
    }
}

// ── Modal synthesis coefficients ─────────────────────────────────────────

fn compute_modal_band_coefficients(base_freq: f32) -> (Vec<[f32; 4]>, Vec<[f32; 4]>, [f32; 8]) {
    let mut coeffs_lo = vec![[0.0f32; 4]; MODAL_CHEB_ORDER];
    let mut coeffs_hi = vec![[0.0f32; 4]; MODAL_CHEB_ORDER];
    let mut freqs = [0.0f32; 8];

    let pi = std::f32::consts::PI;
    let order = MODAL_CHEB_ORDER as f32;

    for m in 0..NUM_MODAL_BANDS {
        // Laplacian eigenvalue evenly spaced in (0, 2)
        let lambda_l = (2.0 * m as f32 + 1.0) / NUM_MODAL_BANDS as f32;
        // Corresponding adjacency eigenvalue in (-1, 1)
        let lambda_a = (1.0 - lambda_l).clamp(-1.0, 1.0);
        let theta = lambda_a.acos();

        // Natural frequency: sqrt of Laplacian eigenvalue scaled by base
        freqs[m] = base_freq * lambda_l.sqrt();

        for k in 0..MODAL_CHEB_ORDER {
            let kf = k as f32;

            // Jackson kernel damping (suppresses Gibbs oscillations)
            let jackson = if k == 0 {
                1.0
            } else {
                let n1 = order + 1.0;
                ((order - kf + 1.0) * (pi * kf / n1).cos()
                    + (pi * kf / n1).sin() / (pi / n1).tan())
                    / n1
            };

            let scale = if k == 0 { 1.0 } else { 2.0 };
            let c = scale * (kf * theta).cos() * jackson;

            if m < 4 {
                coeffs_lo[k][m] = c;
            } else {
                coeffs_hi[k][m - 4] = c;
            }
        }
    }

    (coeffs_lo, coeffs_hi, freqs)
}

// ── Per-trial parameters ─────────────────────────────────────────────────────

#[derive(Clone)]
struct TrialParams {
    rule_state: Vec<u8>,
    rule_topo: Vec<u8>,
    discrete_step_interval: u32,
    max_divisions_per_step: u32,
    hue_base: f32,
    base_freq: f32,
}

const MUTATION_INTERVAL: u64 = 300;
const MUTATION_PROB: f32 = 0.3;

fn mutate_trial_params(params: &mut TrialParams) {
    let n = params.rule_state.len();
    let idx = random_range(0, n);
    if random_f32() < 0.5 {
        params.rule_state[idx] = random_range(0u8, NUM_DISCRETE_STATES);
    } else {
        let num_actions = if NUM_DISCRETE_STATES <= 2 { 2 } else { 3 };
        params.rule_topo[idx] = random_range(0u8, num_actions);
    }
}

fn random_trial_params() -> TrialParams {
    TrialParams {
        rule_state: random_rule_table(NUM_DISCRETE_STATES),
        rule_topo: random_topo_table(NUM_DISCRETE_STATES),
        discrete_step_interval: random_range(10u32, 40),
        max_divisions_per_step: random_range(1u32, 5),
        hue_base: random_range(0.0f32, 360.0),
        base_freq: random_range(120.0f32, 400.0),
    }
}

struct Trial {
    id: u32,
    spawn_frame: u64,
    initial_count: usize,
    params: TrialParams,
}

struct TrialState {
    trials: Vec<Trial>,
    next_trial_id: u32,
    graphs: Vec<GraphEntry>,
}

static TRIAL_STATE: OnceLock<Mutex<TrialState>> = OnceLock::new();

fn with_trials<R>(f: impl FnOnce(&mut TrialState) -> R) -> R {
    let mut ts = TRIAL_STATE.get().unwrap().lock().unwrap();
    f(&mut ts)
}

fn build_node_trial_info(trial_ids: &[u32]) -> Vec<NodeTrialInfo> {
    with_trials(|ts| {
        let trial_map: HashMap<u32, &Trial> = ts.trials.iter().map(|t| (t.id, t)).collect();
        trial_ids.iter().map(|&tid| {
            if let Some(trial) = trial_map.get(&tid) {
                NodeTrialInfo {
                    hue_base: trial.params.hue_base,
                    freq_scale: trial.params.base_freq / MODAL_BASE_FREQ,
                }
            } else {
                NodeTrialInfo { hue_base: 280.0, freq_scale: 1.0 }
            }
        }).collect()
    })
}

// ── Audio ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct AudioFeedback {
    min_buffer_level: u32,
    current_buffer_level: u32,
}

#[derive(Clone, Copy)]
struct ReverbParams {
    room_size: f32,
    damp: f32,
    wet: f32,
}

struct AudioModel {
    consumer: ringbuf::HeapCons<f32>,
    feedback_tx: crossbeam_channel::Sender<AudioFeedback>,
    buffer_history: VecDeque<u32>,
    lp_prev: [f32; 2],
    lp_alpha: f32,
    reverb: reverb::Freeverb,
    reverb_params: Arc<Mutex<ReverbParams>>,
}

fn audio_fn(audio_model: &mut AudioModel, buffer: &mut Buffer) {
    if let Ok(p) = audio_model.reverb_params.try_lock() {
        audio_model.reverb.set_room_size(p.room_size);
        audio_model.reverb.set_damp(p.damp);
        audio_model.reverb.set_wet(p.wet);
    }
    for frame in buffer.frames_mut() {
        let raw_l = audio_model.consumer.try_pop().unwrap_or(0.0);
        let raw_r = audio_model.consumer.try_pop().unwrap_or(0.0);
        let a = audio_model.lp_alpha;
        audio_model.lp_prev[0] += a * (raw_l - audio_model.lp_prev[0]);
        audio_model.lp_prev[1] += a * (raw_r - audio_model.lp_prev[1]);
        let (l, r) = audio_model.reverb.process(audio_model.lp_prev[0], audio_model.lp_prev[1]);
        frame[0] = l;
        frame[1] = r;
    }
    let buffer_len = audio_model.consumer.occupied_len() as u32;
    audio_model.buffer_history.push_back(buffer_len);
    if audio_model.buffer_history.len() > BUFFER_HISTORY_SIZE {
        audio_model.buffer_history.pop_front();
    }
    let min_buffer = audio_model.buffer_history.iter().copied().min().unwrap_or(0);
    audio_model.feedback_tx.send(AudioFeedback {
        min_buffer_level: min_buffer,
        current_buffer_level: buffer_len,
    }).ok();
}

// ── Model ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct Model {
    window: Entity,
    pub(crate) gpu: Option<GpuCompute>,
    pub(crate) render_state: Arc<Mutex<RenderState>>,

    // GRA node data (all trials flattened)
    gra_positions: Vec<(f32, f32)>,
    gra_states: Vec<[f32; 3]>,
    gra_connections: Vec<(usize, usize)>,
    gra_trial_ids: Vec<u32>,
    gra_discrete_states: Vec<u8>,

    // Barnes-Hut repulsion
    bh: BarnesHut,
    gra_charge: f32,
    gra_charge_max_dist: f32,
    gra_charge_epsilon: f32,
    gra_theta: f32,

    // GRA physics
    gra_spring_length: f32,
    gra_spring_stiffness: f32,
    gra_damping: f32,
    gra_max_velocity: f32,
    gra_node_radius: f32,

    // Particle physics
    dt: f32,
    particle_radius: f32,
    particle_collision_radius: f32,
    particle_collision_strength: f32,
    particle_max_force: f32,
    particle_friction: f32,
    particle_mass: f32,
    particle_copy_radius: f32,
    particle_copy_cos_sim: f32,
    particle_copy_prob: f32,
    particle_size: f32,

    // Particle↔GRA repulsion
    gra_repulsion_radius: f32,
    gra_repulsion_strength: f32,

    frame: u64,
    hit_max_nodes: bool,
    time: f32,
    gra_readback_pending: bool,
    gra_readback_count: usize, // node count when readback was submitted

    // Camera
    zoom: f32,
    pan: Vec2,
    dragging_pan: bool,
    last_mouse: Vec2,

    // Audio
    audio_producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
    audio_active: bool,
    audio_volume: f32,
    modal_base_freq: f32,
    modal_coeffs_uploaded: bool,
    request_threshold: u32,
    last_buffer_level: u32,
    last_audio_staging_idx: Arc<AtomicI8>,
    audio_feedback_rx: Arc<Receiver<AudioFeedback>>,
    _audio_thread: Arc<std::thread::JoinHandle<()>>,
    integral_error: f32,
    latest_feedback: Option<AudioFeedback>,
    reverb_params: Arc<Mutex<ReverbParams>>,
}

// ── App ──────────────────────────────────────────────────────────────────────

fn main() {
    nannou::app(model)
        .update(update)
        .render(render::render)
        .run();
}

#[derive(Deserialize, Clone)]
struct GraphEntry {
    state: Vec<u8>,
    edgelist: Vec<[usize; 2]>,
}

fn load_graphs() -> Vec<GraphEntry> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("graphs.json");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", path.display(), e))
}

fn spawn_trial(model: &mut Model, center: (f32, f32)) {
    let params = random_trial_params();
    with_trials(|ts| {
        let graph = ts.graphs[random_range(0, ts.graphs.len())].clone();
        let count = graph.state.len();
        let base_idx = model.gra_positions.len();
        let tid = ts.next_trial_id;
        ts.next_trial_id += 1;

        for _ in 0..count {
            model.gra_positions.push(wrap_pos(
                center.0 + random_range(-model.gra_spring_length, model.gra_spring_length),
                center.1 + random_range(-model.gra_spring_length, model.gra_spring_length),
            ));
            model.gra_states.push([0.0; 3]);
            model.gra_trial_ids.push(tid);
            model.gra_discrete_states.push(random_range(0u8, NUM_DISCRETE_STATES));
        }

        for &[a, b] in &graph.edgelist {
            model.gra_connections.push((base_idx + a, base_idx + b));
        }

        ts.trials.push(Trial {
            id: tid,
            spawn_frame: model.frame,
            initial_count: count,
            params,
        });
    });
}

fn cull_inviable_trials(model: &mut Model) -> bool {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for &tid in &model.gra_trial_ids {
        *counts.entry(tid).or_default() += 1;
    }

    let to_remove: HashSet<u32> = with_trials(|ts| {
        let to_remove: HashSet<u32> = ts.trials.iter()
            .filter(|t| {
                let elapsed = model.frame.saturating_sub(t.spawn_frame);
                let count = counts.get(&t.id).copied().unwrap_or(0);
                elapsed > GRACE_FRAMES && count < MIN_VIABLE_NODES
            })
            .map(|t| t.id)
            .collect();

        if !to_remove.is_empty() {
            ts.trials.retain(|t| !to_remove.contains(&t.id));
        }
        to_remove
    });

    if to_remove.is_empty() { return false; }

    let n = model.gra_positions.len();
    let keep: Vec<bool> = model.gra_trial_ids.iter()
        .map(|tid| !to_remove.contains(tid))
        .collect();

    let mut new_idx = vec![0usize; n];
    let mut cursor = 0usize;
    for i in 0..n {
        new_idx[i] = cursor;
        if keep[i] { cursor += 1; }
    }

    let mut new_positions = Vec::with_capacity(cursor);
    let mut new_states = Vec::with_capacity(cursor);
    let mut new_trial_ids = Vec::with_capacity(cursor);
    let mut new_discrete = Vec::with_capacity(cursor);
    for i in 0..n {
        if keep[i] {
            new_positions.push(model.gra_positions[i]);
            new_states.push(model.gra_states[i]);
            new_trial_ids.push(model.gra_trial_ids[i]);
            new_discrete.push(model.gra_discrete_states[i]);
        }
    }

    let new_connections: Vec<(usize, usize)> = model.gra_connections.iter()
        .filter(|&&(a, b)| a < n && b < n && keep[a] && keep[b])
        .map(|&(a, b)| (new_idx[a], new_idx[b]))
        .collect();

    model.gra_positions = new_positions;
    model.gra_states = new_states;
    model.gra_trial_ids = new_trial_ids;
    model.gra_discrete_states = new_discrete;
    model.gra_connections = new_connections;

    if let Some(ref mut gpu) = model.gpu {
        gpu.topology_dirty = true;
    }
    model.gra_readback_pending = false; // topology changed, stale readback
    true
}

fn init_particles(world_half: f32) -> Vec<Particle> {
    let n = NUM_PARTICLES as usize;
    let mut particles = Vec::with_capacity(n);

    let map_x0 = -world_half;
    let map_x1 = world_half;
    let map_y0 = -world_half;
    let map_y1 = world_half;

    let min_bin_species = 1usize;
    let max_bin_species = 3usize;
    let initial_velocity = 0.1f32;

    // Tile the world into a grid; each tile gets 1-3 random species
    let bin_size = 2.0f32;
    let grid_size_x = ((map_x1 - map_x0) / bin_size).ceil() as u32;
    let grid_size_y = ((map_y1 - map_y0) / bin_size).ceil() as u32;
    let bin_count = grid_size_x * grid_size_y;

    for j in 0..grid_size_y {
        for i in 0..grid_size_x {
            let bin_index = j * grid_size_x + i;
            let bin_start = (NUM_PARTICLES * bin_index / bin_count) as usize;
            let bin_end = if i == grid_size_x - 1 && j == grid_size_y - 1 {
                n
            } else {
                (NUM_PARTICLES * (bin_index + 1) / bin_count) as usize
            };
            let bin_particle_count = bin_end - bin_start;

            // Random species set for this tile
            let num_bin_species = random_range(min_bin_species, max_bin_species + 1)
                .min(max_bin_species);
            let species_in_bin: Vec<([f32; 2], [f32; 2])> = (0..num_bin_species)
                .map(|_| (
                    [random_range(-1.0f32, 1.0), random_range(-1.0f32, 1.0)],
                    [random_range(-1.0f32, 1.0), random_range(-1.0f32, 1.0)],
                ))
                .collect();

            for k in 0..bin_particle_count {
                let (species, alpha) = &species_in_bin[k % num_bin_species];

                // Position within this tile
                let x = map_x0 + (i as f32 + random_f32()) * (map_x1 - map_x0) / grid_size_x as f32;
                let y = map_y0 + (j as f32 + random_f32()) * (map_y1 - map_y0) / grid_size_y as f32;

                particles.push(Particle {
                    pos: [x, y],
                    vel: [
                        initial_velocity * random_range(-1.0f32, 1.0),
                        initial_velocity * random_range(-1.0f32, 1.0),
                    ],
                    phase: random_f32(),
                    energy: 0.0,
                    species: *species,
                    alpha: *alpha,
                    interaction: [0.0, 0.0],
                    amp_phase: random_f32(),
                    _pad: 0.0,
                });
            }
        }
    }
    particles
}

fn model(app: &App) -> Model {
    let w_id = app.new_window()
        .size(1200, 900)
        .key_pressed(key_pressed)
        .mouse_pressed(mouse_pressed)
        .mouse_released(mouse_released)
        .mouse_moved(mouse_moved)
        .mouse_wheel(mouse_wheel)
        .build();

    let graphs = load_graphs();

    // Audio setup
    let ring_buf_size = CHUNK_SIZE as usize * NUM_CHANNELS as usize * 32;
    let ring_buf = HeapRb::<f32>::new(ring_buf_size);
    let (audio_producer, audio_consumer) = ring_buf.split();

    let (audio_feedback_tx, audio_feedback_rx) = crossbeam_channel::unbounded();
    let reverb_params = Arc::new(Mutex::new(ReverbParams {
        room_size: 0.85,
        damp: 0.5,
        wet: 0.9,
    }));
    // One-pole lowpass at ~12 kHz to knock out digital clicks before reverb
    let lp_cutoff = 12000.0_f32;
    let lp_alpha = 1.0 - (-std::f32::consts::TAU * lp_cutoff / SAMPLE_RATE as f32).exp();
    let audio_model = AudioModel {
        consumer: audio_consumer,
        feedback_tx: audio_feedback_tx,
        buffer_history: VecDeque::with_capacity(BUFFER_HISTORY_SIZE),
        lp_prev: [0.0; 2],
        lp_alpha,
        reverb: reverb::Freeverb::new(SAMPLE_RATE),
        reverb_params: Arc::clone(&reverb_params),
    };

    let audio_host = audio::Host::new();
    let audio_thread = std::thread::spawn(move || {
        let stream = audio_host
            .new_output_stream(audio_model)
            .render(audio_fn)
            .channels(NUM_CHANNELS as usize)
            .sample_rate(SAMPLE_RATE)
            .build()
            .unwrap();
        stream.play().unwrap();
        loop { std::thread::sleep(std::time::Duration::from_secs(1)); }
    });

    TRIAL_STATE.set(Mutex::new(TrialState {
        trials: Vec::new(),
        next_trial_id: 0,
        graphs,
    })).ok();

    let mut m = Model {
        window: w_id,
        gpu: None,
        render_state: Arc::new(Mutex::new(RenderState::default())),

        gra_positions: Vec::new(),
        gra_states: Vec::new(),
        gra_connections: Vec::new(),
        gra_trial_ids: Vec::new(),
        gra_discrete_states: Vec::new(),

        bh: BarnesHut::new(),
        gra_charge: 0.0001,
        gra_charge_max_dist: 0.5,
        gra_charge_epsilon: 0.01,
        gra_theta: 0.9,

        gra_spring_length: 0.01,
        gra_spring_stiffness: 0.05,
        gra_damping: 0.3,
        gra_max_velocity: 0.005,
        gra_node_radius: 0.05,

        dt: 0.035,
        particle_radius: 0.3,
        particle_collision_radius: 0.1,
        particle_collision_strength: 15.0,
        particle_max_force: 1.0,
        particle_friction: 0.1,
        particle_mass: 1.0,
        particle_copy_radius: 0.2,
        particle_copy_cos_sim: 0.5,
        particle_copy_prob: 0.001,
        particle_size: 0.05,

        gra_repulsion_radius: 0.5,
        gra_repulsion_strength: 4.0,

        frame: 0,
        hit_max_nodes: false,
        time: 0.0,
        gra_readback_pending: false,
        gra_readback_count: 0,

        zoom: 1.0,
        pan: Vec2::ZERO,
        dragging_pan: false,
        last_mouse: Vec2::ZERO,

        audio_producer: Arc::new(Mutex::new(audio_producer)),
        audio_active: true,
        audio_volume: 0.5,
        modal_base_freq: MODAL_BASE_FREQ,
        modal_coeffs_uploaded: false,
        request_threshold: INITIAL_REQUEST_THRESHOLD,
        last_buffer_level: 0,
        last_audio_staging_idx: Arc::new(AtomicI8::new(-1)),
        audio_feedback_rx: Arc::new(audio_feedback_rx),
        _audio_thread: Arc::new(audio_thread),
        integral_error: 0.0,
        latest_feedback: None,
        reverb_params,
    };

    // Spawn initial GRA trials in a grid
    let grid_side = (NUM_INITIAL_TRIALS as f32).sqrt().ceil() as i32;
    let offset = (grid_side as f32 - 1.0) * GRID_SPACING * 0.5;
    let mut count = 0;
    for row in 0..grid_side {
        for col in 0..grid_side {
            if count >= NUM_INITIAL_TRIALS { break; }
            let cx = col as f32 * GRID_SPACING - offset + random_range(-0.4, 0.4);
            let cy = row as f32 * GRID_SPACING - offset + random_range(-0.4, 0.4);
            spawn_trial(&mut m, (cx, cy));
            count += 1;
        }
    }

    m
}

fn update(app: &App, model: &mut Model) {
    // ── egui UI ──────────────────────────────────────────────────────────
    let mut egui_ctx = app.egui_for_window(model.window);
    let ctx = egui_ctx.get_mut();

    egui::Window::new("Pond").show(&ctx, |ui| {
        let n_gra = model.gra_positions.len();
        let n_trials = with_trials(|ts| ts.trials.len());
        ui.label(format!("GRA: {} trials, {} nodes  |  Particles: {}", n_trials, n_gra, NUM_PARTICLES));
        if model.hit_max_nodes {
            ui.colored_label(egui::Color32::YELLOW, "Max GRA nodes reached");
        }
        ui.separator();

        ui.collapsing("GRA Physics", |ui| {
            ui.add(egui::Slider::new(&mut model.gra_charge, 0.0..=0.01).text("charge"));
            ui.add(egui::Slider::new(&mut model.gra_charge_max_dist, 0.01..=2.0).text("charge dist"));
            ui.add(egui::Slider::new(&mut model.gra_charge_epsilon, 0.001..=0.1).text("charge eps"));
            ui.add(egui::Slider::new(&mut model.gra_spring_stiffness, 0.001..=0.5).text("spring k"));
            ui.add(egui::Slider::new(&mut model.gra_spring_length, 0.001..=0.1).text("spring len"));
            ui.add(egui::Slider::new(&mut model.gra_damping, 0.01..=1.0).text("damping"));
            ui.add(egui::Slider::new(&mut model.gra_max_velocity, 0.0001..=0.05).text("max vel"));
        });

        ui.collapsing("Particle Physics", |ui| {
            ui.add(egui::Slider::new(&mut model.dt, 0.001..=0.1).text("dt"));
            ui.add(egui::Slider::new(&mut model.particle_radius, 0.01..=2.0).text("radius"));
            ui.add(egui::Slider::new(&mut model.particle_collision_radius, 0.01..=0.5).text("collision r"));
            ui.add(egui::Slider::new(&mut model.particle_collision_strength, 0.0..=30.0).text("collision str"));
            ui.add(egui::Slider::new(&mut model.particle_max_force, 0.0..=10.0).text("max force"));
            ui.add(egui::Slider::new(&mut model.particle_friction, 0.01..=1.0).text("friction"));
            ui.add(egui::Slider::new(&mut model.particle_mass, 0.1..=10.0).text("mass"));
        });

        ui.collapsing("GRA↔Particle", |ui| {
            ui.add(egui::Slider::new(&mut model.gra_repulsion_radius, 0.1..=2.0).text("repulsion r"));
            ui.add(egui::Slider::new(&mut model.gra_repulsion_strength, 0.0..=20.0).text("repulsion str"));
        });

        ui.separator();
        ui.label("Audio (Modal Synthesis)");
        ui.checkbox(&mut model.audio_active, "Enable audio");
        ui.add(egui::Slider::new(&mut model.audio_volume, 0.0..=1.0).text("volume"));
        if let Ok(mut rp) = model.reverb_params.try_lock() {
            ui.add(egui::Slider::new(&mut rp.wet, 0.0..=1.0).text("reverb"));
            ui.add(egui::Slider::new(&mut rp.room_size, 0.0..=1.0).text("room size"));
            ui.add(egui::Slider::new(&mut rp.damp, 0.0..=1.0).text("damping"));
        }
        let old_base = model.modal_base_freq;
        ui.add(egui::Slider::new(&mut model.modal_base_freq, 50.0..=500.0).text("base freq"));
        if (model.modal_base_freq - old_base).abs() > 0.5 {
            model.modal_coeffs_uploaded = false;
        }
        if let Some(ref fb) = model.latest_feedback {
            ui.label(format!("buf: {} / thresh: {}", fb.current_buffer_level, model.request_threshold));
        }
        ui.separator();
        ui.label("R - Reset  |  Scroll - Zoom  |  Drag - Pan");
    });
    drop(egui_ctx);

    model.frame += 1;
    model.time += 1.0 / 60.0;

    // ── Initialize GPU on second frame ───────────────────────────────────
    if model.gpu.is_none() {
        if model.frame < 2 { return; }
        let window = app.window(model.window);
        let device = window.device();
        model.gpu = Some(create_gpu_compute(&device, WORLD_HALF, P_BIN_SIZE, G_BIN_SIZE));

        // Upload initial particles
        let particles = init_particles(WORLD_HALF);
        let gpu = model.gpu.as_mut().unwrap();
        upload_particles(&window.queue(), gpu, &particles);
        gpu.topology_dirty = true;
    }

    // ── Upload modal coefficients if needed ────────────────────────────────
    if !model.modal_coeffs_uploaded && model.gpu.is_some() {
        let (coeffs_lo, coeffs_hi, freqs) = compute_modal_band_coefficients(model.modal_base_freq);
        let window = app.window(model.window);
        let queue = window.queue();
        let gpu = model.gpu.as_ref().unwrap();
        upload_modal_coefficients(&queue, gpu, &coeffs_lo, &coeffs_hi);
        let modal_freqs = ModalFreqs {
            lo: [freqs[0], freqs[1], freqs[2], freqs[3]],
            hi: [freqs[4], freqs[5], freqs[6], freqs[7]],
        };
        update_modal_freqs(&queue, gpu, &modal_freqs);
        model.modal_coeffs_uploaded = true;
    }

    // ── Upload GRA topology if dirty ─────────────────────────────────────
    if model.gpu.as_ref().unwrap().topology_dirty {
        let window = app.window(model.window);
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        upload_gra_topology(&queue, gpu, &model.gra_positions, &model.gra_states, &model.gra_connections);
        let trial_info = build_node_trial_info(&model.gra_trial_ids);
        upload_gra_discrete_states(&queue, gpu, &model.gra_discrete_states, NUM_DISCRETE_STATES, &trial_info);
        gpu.topology_dirty = false;
    }

    // ── Build SimParams ──────────────────────────────────────────────────
    let (p_bins_x, p_bins_y) = compute_bin_counts(WORLD_HALF, P_BIN_SIZE);
    let (g_bins_x, g_bins_y) = compute_bin_counts(WORLD_HALF, G_BIN_SIZE);

    let sim_params = SimParams {
        world_half: WORLD_HALF,
        dt: model.dt,
        time: model.time,
        num_particles: NUM_PARTICLES,
        particle_friction: model.particle_friction,
        particle_mass: model.particle_mass,
        particle_radius: model.particle_radius,
        particle_collision_radius: model.particle_collision_radius,
        particle_collision_strength: model.particle_collision_strength,
        particle_max_force: model.particle_max_force,
        particle_copy_radius: model.particle_copy_radius,
        particle_copy_cos_sim: model.particle_copy_cos_sim,
        particle_copy_prob: model.particle_copy_prob,
        p_bin_size: P_BIN_SIZE,
        p_num_bins_x: p_bins_x,
        p_num_bins_y: p_bins_y,
        num_gra_nodes: model.gra_positions.len() as u32,
        num_gra_connections: model.gra_connections.len() as u32,
        gra_spring_length: model.gra_spring_length,
        gra_spring_stiffness: model.gra_spring_stiffness,
        gra_damping: model.gra_damping,
        gra_max_velocity: model.gra_max_velocity,
        g_bin_size: G_BIN_SIZE,
        g_num_bins_x: g_bins_x,
        g_num_bins_y: g_bins_y,
        gra_repulsion_radius: model.gra_repulsion_radius,
        gra_repulsion_strength: model.gra_repulsion_strength,
        particle_friction_mu: (0.5f32).powf(model.dt / model.particle_friction),
        current_strength: 0.05,
        _pad0: 0,
    };

    // ── GRA: CPU Barnes-Hut repulsion ────────────────────────────────────
    {
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();

        if !model.gra_positions.is_empty() {
            // Read back PREVIOUS frame's positions (async double-buffered)
            // Only apply if topology hasn't changed (node count must match)
            if model.gra_readback_pending && model.gra_readback_count == model.gra_positions.len() {
                let read_idx = gpu.gra_readback_frame;
                let positions = readback_gra_positions(&device, gpu, model.gra_readback_count, read_idx);
                model.gra_positions = positions.into_iter()
                    .map(|(x, y)| wrap_pos(x, y))
                    .collect();
            }

            let jitter_scale = model.bh.last_leaf_extent() * 0.5;
            let jitter = (
                random_range(-jitter_scale, jitter_scale),
                random_range(-jitter_scale, jitter_scale),
            );
            let forces = model.bh.compute_repulsion(
                &model.gra_positions, model.gra_charge, model.gra_charge_max_dist,
                model.gra_theta, jitter, model.gra_charge_epsilon, WORLD_HALF,
            );

            upload_gra_forces(&queue, gpu, &forces);
        }

        update_sim_params(&queue, gpu, &sim_params);

        // ── Single batched encoder: GRA physics + GRA bin sort + particle bin sort + particle sim
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("physics_encoder"),
        });

        let gra_readback_write_idx = if !model.gra_positions.is_empty() {
            let write_idx = 1 - gpu.gra_readback_frame;
            dispatch_gra_physics(&mut encoder, gpu, write_idx);
            Some(write_idx)
        } else {
            None
        };

        dispatch_gra_bin_sort(&mut encoder, gpu);
        dispatch_particle_bin_sort(&mut encoder, gpu);
        dispatch_particle_sim(&mut encoder, gpu);

        queue.submit(Some(encoder.finish()));

        if let Some(write_idx) = gra_readback_write_idx {
            gpu.gra_readback_frame = write_idx;
            model.gra_readback_pending = true;
            model.gra_readback_count = model.gra_positions.len();
        }
    }

    // ── Audio ────────────────────────────────────────────────────────────
    if model.audio_active && model.gpu.is_some() {
        let mut latest_feedback = None;
        while let Ok(fb) = model.audio_feedback_rx.try_recv() {
            latest_feedback = Some(fb);
        }
        if let Some(feedback) = latest_feedback {
            model.last_buffer_level = feedback.current_buffer_level;
            model.latest_feedback = Some(feedback);

            let target = CHUNK_FLOATS as f32;
            let error = target - feedback.min_buffer_level as f32;
            model.integral_error = (model.integral_error + error * 0.01).clamp(-1000.0, 1000.0);
            let adjustment = error * 0.1 + model.integral_error * 0.05;
            let new_thresh = (model.request_threshold as f32 + adjustment).clamp(
                MIN_REQUEST_THRESHOLD as f32, MAX_REQUEST_THRESHOLD as f32,
            );
            model.request_threshold = new_thresh as u32;
        }

        let need_audio = model.last_buffer_level < model.request_threshold;

        // Read back audio from previous frame
        let prev_staging_idx = model.last_audio_staging_idx.load(Ordering::Relaxed);
        if prev_staging_idx >= 0 {
            let read_idx = prev_staging_idx as usize;
            let gpu = model.gpu.as_ref().unwrap();
            let read_buf = &gpu.audio_staging_bufs[read_idx];
            let audio_buf_size = (CHUNK_SIZE * NUM_CHANNELS * 4) as usize;

            let slice = read_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let window = app.window(model.window);
            let device = window.device();
            device.poll(wgpu::PollType::Wait).unwrap();

            let data = slice.get_mapped_range();
            let floats = bytemuck::cast_slice::<u8, f32>(&data[..audio_buf_size]);
            if let Ok(mut producer) = model.audio_producer.lock() {
                producer.push_slice(floats);
            }
            drop(data);
            read_buf.unmap();
        }

        if need_audio {
            let cx = model.pan.x;
            let cy = model.pan.y;
            let hw = WORLD_HALF / model.zoom;
            let hh = WORLD_HALF / model.zoom;

            let audio_params = AudioParams {
                sample_rate: SAMPLE_RATE as f32,
                num_particles: NUM_PARTICLES,
                num_gra_nodes: model.gra_positions.len() as u32,
                chunk_size: CHUNK_SIZE,
                volume: model.audio_volume,
                _align_pad: 0.0,
                current_x: [cx - hw, cx + hw],
                current_y: [cy - hh, cy + hh],
                max_speed: 5.0,
                energy_scale: 10.0,
                gra_max_speed: model.gra_max_velocity,
                p_map_x0: -WORLD_HALF,
                p_map_y0: -WORLD_HALF,
                p_bin_size: P_BIN_SIZE,
                p_num_bins_x: p_bins_x,
                p_num_bins_y: p_bins_y,
                g_bin_size: G_BIN_SIZE,
                g_num_bins_x: g_bins_x,
                g_num_bins_y: g_bins_y,
                world_half: WORLD_HALF,
                _pad0: 0, _pad1: 0, _pad2: 0, _pad3: 0,
            };

            let window = app.window(model.window);
            let device = window.device();
            let queue = window.queue();
            let gpu = model.gpu.as_ref().unwrap();

            update_audio_params(&queue, gpu, &audio_params);

            let write_idx = if prev_staging_idx < 0 { 0 }
            else { (prev_staging_idx as usize + 1) % NUM_AUDIO_STAGING_BUFS };

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("audio_encoder"),
            });

            // Chebyshev bandpass: decompose GRA state into 8 spectral bands
            encode_modal_chebyshev(&mut encoder, gpu, MODAL_CHEB_ORDER);
            // Combined particle + GRA modal synthesis + phase updates + copy to staging
            encode_audio_pass(&mut encoder, gpu, write_idx);

            queue.submit(Some(encoder.finish()));

            model.last_audio_staging_idx.store(write_idx as i8, Ordering::Relaxed);
        } else {
            model.last_audio_staging_idx.store(-1, Ordering::Relaxed);
        }
    }

    // ── GRA discrete CA step ─────────────────────────────────────────────
    {
        let trial_params_map: HashMap<u32, TrialParams> = with_trials(|ts| {
            ts.trials.iter().map(|t| (t.id, t.params.clone())).collect()
        });

        let stepping_trials: HashSet<u32> = with_trials(|ts| {
            ts.trials.iter()
                .filter(|t| {
                    let elapsed = model.frame.saturating_sub(t.spawn_frame);
                    elapsed > 0 && elapsed % t.params.discrete_step_interval as u64 == 0
                })
                .map(|t| t.id)
                .collect()
        });

        if !stepping_trials.is_empty() {
            // Canonicalize edges: always store (min, max) to prevent duplicates like (a,b)+(b,a)
            for conn in model.gra_connections.iter_mut() {
                if conn.0 > conn.1 { std::mem::swap(&mut conn.0, &mut conn.1); }
            }
            model.gra_connections.sort();
            model.gra_connections.dedup();

            let n = model.gra_positions.len();
            let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
            for &(a, b) in &model.gra_connections {
                if a < n && b < n {
                    adj[a].insert(b);
                    adj[b].insert(a);
                }
            }

            model.gra_discrete_states.resize(n, 1);

            let mut new_states = model.gra_discrete_states.clone();
            let mut topo_actions = vec![1u8; n];

            for i in 0..n {
                let tid = model.gra_trial_ids[i];
                if !stepping_trials.contains(&tid) { continue; }
                let tp = match trial_params_map.get(&tid) {
                    Some(p) => p,
                    None => continue,
                };
                let own = model.gra_discrete_states[i];
                let neighbor_sum: u32 = adj[i].iter()
                    .map(|&nb| model.gra_discrete_states[nb] as u32)
                    .sum();
                let cfg = discrete_config(own, neighbor_sum);

                new_states[i] = if cfg < tp.rule_state.len() { tp.rule_state[cfg] } else { own };
                let raw_topo = if cfg < tp.rule_topo.len() { tp.rule_topo[cfg] } else { 1 };
                topo_actions[i] = canonical_topo_action(raw_topo, NUM_DISCRETE_STATES);
            }

            model.gra_discrete_states = new_states;

            // Merges
            let mut merged: HashSet<usize> = HashSet::new();
            let mut to_remove: Vec<usize> = Vec::new();

            for i in 0..n {
                if topo_actions[i] != 0 || merged.contains(&i) { continue; }
                let neighbours: Vec<usize> = adj[i].iter().copied()
                    .filter(|&nb| !merged.contains(&nb))
                    .collect();
                let mut found = None;
                'tri: for ni in 0..neighbours.len() {
                    let a = neighbours[ni];
                    for nj in (ni + 1)..neighbours.len() {
                        let b = neighbours[nj];
                        if adj[a].contains(&b) {
                            found = Some((a, b));
                            break 'tri;
                        }
                    }
                }
                let (a, b) = match found { Some(pair) => pair, None => continue };

                // The three external neighbours (one per triangle vertex) must be
                // distinct, otherwise dedup after remapping would drop edges and
                // break 3-regularity.
                let ext_i = adj[i].iter().copied().find(|&x| x != a && x != b);
                let ext_a = adj[a].iter().copied().find(|&x| x != i && x != b);
                let ext_b = adj[b].iter().copied().find(|&x| x != i && x != a);
                match (ext_i, ext_a, ext_b) {
                    (Some(ei), Some(ea), Some(eb))
                        if ei != ea && ea != eb && ei != eb => {}
                    _ => continue,
                }

                let avg_pos = toroidal_avg(&[
                    model.gra_positions[i], model.gra_positions[a], model.gra_positions[b],
                ]);
                model.gra_positions[a] = avg_pos;
                for conn in model.gra_connections.iter_mut() {
                    if conn.0 == i || conn.0 == b { conn.0 = a; }
                    if conn.1 == i || conn.1 == b { conn.1 = a; }
                }
                merged.insert(i);
                merged.insert(b);
                to_remove.push(i);
                to_remove.push(b);
            }

            if !to_remove.is_empty() {
                // Canonicalize after merge remap
                for conn in model.gra_connections.iter_mut() {
                    if conn.0 > conn.1 { std::mem::swap(&mut conn.0, &mut conn.1); }
                }
                model.gra_connections.retain(|&(a, b)| a != b);

                to_remove.sort();
                to_remove.dedup();
                let removed_set: HashSet<usize> = to_remove.iter().copied().collect();

                // Remove connections involving removed nodes before remapping indices
                model.gra_connections.retain(|&(a, b)| {
                    !removed_set.contains(&a) && !removed_set.contains(&b)
                });

                // Track where each original index ends up after all swap_removes
                let old_len = model.gra_positions.len();
                let mut pos_of: Vec<usize> = (0..old_len).collect();
                let mut elem_at: Vec<usize> = (0..old_len).collect();

                for &idx in to_remove.iter().rev() {
                    let last = model.gra_positions.len() - 1;
                    model.gra_positions.swap_remove(idx);
                    model.gra_states.swap_remove(idx);
                    model.gra_discrete_states.swap_remove(idx);
                    model.gra_trial_ids.swap_remove(idx);

                    if idx < last {
                        let moved_elem = elem_at[last];
                        pos_of[moved_elem] = idx;
                        elem_at[idx] = moved_elem;
                    }
                }

                // Apply remap to connections in a single pass
                for conn in model.gra_connections.iter_mut() {
                    conn.0 = pos_of[conn.0];
                    conn.1 = pos_of[conn.1];
                }

                // Single final canonicalize
                let new_len = model.gra_positions.len();
                for conn in model.gra_connections.iter_mut() {
                    if conn.0 > conn.1 { std::mem::swap(&mut conn.0, &mut conn.1); }
                }
                model.gra_connections.retain(|&(a, b)| a < new_len && b < new_len && a != b);
                model.gra_connections.sort();
                model.gra_connections.dedup();
            }

            // Splits
            let n = model.gra_positions.len();
            let mut new_nodes: Vec<((f32, f32), u8, u32)> = Vec::new();
            let mut new_connections = Vec::new();
            let mut div_counts: HashMap<u32, usize> = HashMap::new();

            if n < MAX_GRA_NODES {
                for i in 0..n {
                    if i >= topo_actions.len() || topo_actions[i] != 2 || merged.contains(&i) { continue; }
                    if n + new_nodes.len() + 2 > MAX_GRA_NODES {
                        model.hit_max_nodes = true;
                        break;
                    }
                    let tid = model.gra_trial_ids[i];
                    let max_div = trial_params_map.get(&tid)
                        .map(|p| p.max_divisions_per_step as usize)
                        .unwrap_or(3);
                    let count = div_counts.entry(tid).or_default();
                    if *count >= max_div { continue; }

                    let (px, py) = model.gra_positions[i];
                    let state = model.gra_discrete_states[i];
                    let spread = model.gra_spring_length;
                    let na = wrap_pos(px + random_range(-spread, spread), py + random_range(-spread, spread));
                    let nb = wrap_pos(px + random_range(-spread, spread), py + random_range(-spread, spread));

                    let na_idx = n + new_nodes.len();
                    new_nodes.push((na, state, tid));
                    let nb_idx = n + new_nodes.len();
                    new_nodes.push((nb, state, tid));

                    new_connections.push((i, na_idx));
                    new_connections.push((i, nb_idx));
                    new_connections.push((na_idx, nb_idx));

                    let mut counter = 0;
                    for conn in model.gra_connections.iter_mut() {
                        if conn.0 == i {
                            if counter == 1 { conn.0 = na_idx; }
                            else if counter == 2 { conn.0 = nb_idx; }
                            counter += 1;
                        } else if conn.1 == i {
                            if counter == 1 { conn.1 = na_idx; }
                            else if counter == 2 { conn.1 = nb_idx; }
                            counter += 1;
                        }
                        if counter > 2 { break; }
                    }
                    *count += 1;
                }
            }

            if !new_nodes.is_empty() {
                for ((x, y), ds, tid) in &new_nodes {
                    model.gra_positions.push((*x, *y));
                    model.gra_states.push([0.0; 3]);
                    model.gra_discrete_states.push(*ds);
                    model.gra_trial_ids.push(*tid);
                }
                model.gra_connections.extend_from_slice(&new_connections);
            }

            if !to_remove.is_empty() || !new_nodes.is_empty() {
                model.gra_readback_pending = false; // topology changed, stale readback
                let window = app.window(model.window);
                let queue = window.queue();
                let gpu = model.gpu.as_mut().unwrap();
                upload_gra_topology(&queue, gpu, &model.gra_positions, &model.gra_states, &model.gra_connections);
                let trial_info = build_node_trial_info(&model.gra_trial_ids);
                upload_gra_discrete_states(&queue, gpu, &model.gra_discrete_states, NUM_DISCRETE_STATES, &trial_info);
            } else {
                let window = app.window(model.window);
                let queue = window.queue();
                let gpu = model.gpu.as_ref().unwrap();
                let trial_info = build_node_trial_info(&model.gra_trial_ids);
                upload_gra_discrete_states(&queue, gpu, &model.gra_discrete_states, NUM_DISCRETE_STATES, &trial_info);
            }
        }
    }

    // ── Trial lifecycle ──────────────────────────────────────────────────
    if model.frame % 30 == 0 && model.frame > GRACE_FRAMES {
        cull_inviable_trials(model);
    }

    if model.frame % MUTATION_INTERVAL == 0 {
        with_trials(|ts| {
            for trial in ts.trials.iter_mut() {
                if random_f32() < MUTATION_PROB {
                    mutate_trial_params(&mut trial.params);
                }
            }
        });
    }

    if model.frame % SPAWN_INTERVAL == 0 {
        let num_trials = with_trials(|ts| ts.trials.len());
        if num_trials < MAX_TRIALS && model.gra_positions.len() + 20 < MAX_GRA_NODES {
            let cx = random_range(-WORLD_HALF * 0.8, WORLD_HALF * 0.8);
            let cy = random_range(-WORLD_HALF * 0.8, WORLD_HALF * 0.8);
            spawn_trial(model, (cx, cy));
            if let Some(ref mut gpu) = model.gpu {
                gpu.topology_dirty = true;
            }
            model.gra_readback_pending = false; // topology changed, stale readback
        }
    }

    // ── Update render uniforms ───────────────────────────────────────────
    {
        let cx = model.pan.x;
        let cy = model.pan.y;
        let hw = WORLD_HALF / model.zoom;
        let hh = WORLD_HALF / model.zoom;

        let window = app.window(model.window);
        let win_rect = window.rect();
        let window_aspect = win_rect.w() / win_rect.h();

        let uniforms = RenderUniforms {
            min_x: cx - hw,
            min_y: cy - hh,
            max_x: cx + hw,
            max_y: cy + hh,
            particle_size: model.particle_size,
            gra_node_radius: model.gra_node_radius,
            num_particles: NUM_PARTICLES,
            num_gra_nodes: model.gra_positions.len() as u32,
            num_gra_connections: model.gpu.as_ref().map(|g| g.num_gra_connections).unwrap_or(0),
            window_aspect,
            world_half: WORLD_HALF,
            max_speed: 5.0,
            energy_scale: 10.0,
            current_strength: 0.05,
            time: model.time,
            _pad0: 0,
        };
        let queue = window.queue();
        let gpu = model.gpu.as_ref().unwrap();
        update_render_uniforms(&queue, gpu, &uniforms);
    }
}

// ── Input handlers ───────────────────────────────────────────────────────────

fn mouse_pressed(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Left || button == MouseButton::Right || button == MouseButton::Middle {
        model.dragging_pan = true;
    }
}

fn mouse_released(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Left || button == MouseButton::Right || button == MouseButton::Middle {
        model.dragging_pan = false;
    }
}

fn mouse_moved(_app: &App, model: &mut Model, pos: Vec2) {
    if model.dragging_pan {
        let delta = pos - model.last_mouse;
        let view_half = WORLD_HALF / model.zoom;
        model.pan.x -= delta.x * view_half * 2.0 / 1200.0;
        model.pan.y += delta.y * view_half * 2.0 / 900.0;
    }
    model.last_mouse = pos;
}

fn mouse_wheel(_app: &App, model: &mut Model, wheel: MouseWheel) {
    let factor = 1.0 + wheel.y * 0.1;
    model.zoom = (model.zoom * factor).clamp(0.1, 50.0);
}

fn key_pressed(app: &App, model: &mut Model, key: KeyCode) {
    if key == KeyCode::KeyS {
        app.window(model.window)
            .save_screenshot(app.exe_name().unwrap() + ".png");
    }
    if key == KeyCode::KeyR {
        // Reset everything
        model.gra_positions.clear();
        model.gra_states.clear();
        model.gra_connections.clear();
        model.gra_trial_ids.clear();
        model.gra_discrete_states.clear();
        model.frame = 0;
        model.time = 0.0;
        model.hit_max_nodes = false;
        model.zoom = 1.0;
        model.pan = Vec2::ZERO;
        model.modal_coeffs_uploaded = false;

        with_trials(|ts| {
            ts.trials.clear();
            ts.next_trial_id = 0;
        });

        // Respawn GRA trials
        let grid_side = (NUM_INITIAL_TRIALS as f32).sqrt().ceil() as i32;
        let offset = (grid_side as f32 - 1.0) * GRID_SPACING * 0.5;
        let mut count = 0;
        for row in 0..grid_side {
            for col in 0..grid_side {
                if count >= NUM_INITIAL_TRIALS { break; }
                let cx = col as f32 * GRID_SPACING - offset + random_range(-0.4, 0.4);
                let cy = row as f32 * GRID_SPACING - offset + random_range(-0.4, 0.4);
                spawn_trial(model, (cx, cy));
                count += 1;
            }
        }

        // Re-upload particles
        if let Some(ref mut gpu) = model.gpu {
            let particles = init_particles(WORLD_HALF);
            let window = app.window(model.window);
            upload_particles(&window.queue(), gpu, &particles);
            gpu.topology_dirty = true;
        }
    }
}
