#![allow(dead_code)]

mod barnes_hut;
mod gpu;
mod render;

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
use gpu::{GpuCompute, GpuSimParams, RenderUniforms, AudioParams, ModalFreqs, MAX_NODES,
          SAMPLE_RATE, CHUNK_SIZE, NUM_CHANNELS, CHUNK_FLOATS, NUM_AUDIO_STAGING_BUFS,
          NUM_MODAL_BANDS, MODAL_CHEB_ORDER,
          create_gpu_compute, upload_topology, upload_forces,
          readback_bbox_and_positions, update_render_uniforms,
          gpu_dispatch_physics_only, upload_discrete_states,
          update_audio_params,
          upload_modal_coefficients, update_modal_freqs,
          encode_modal_chebyshev, encode_modal_audio_pass};
use render::RenderState;

const INITIAL_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 2;
const MIN_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS;
const MAX_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 4;
const BUFFER_HISTORY_SIZE: usize = 64;

#[derive(Clone, Copy)]
struct AudioFeedback {
    min_buffer_level: u32,
    current_buffer_level: u32,
}

struct AudioModel {
    consumer: ringbuf::HeapCons<f32>,
    feedback_tx: crossbeam_channel::Sender<AudioFeedback>,
    buffer_history: VecDeque<u32>,
}

const NUM_INITIAL_TRIALS: usize = 16;
const GRID_SPACING: f32 = 800.0;
const MIN_VIABLE_NODES: usize = 4;
const GRACE_FRAMES: u64 = 200;
const SPAWN_INTERVAL: u64 = 120;
const MAX_TRIALS: usize = 30;
const WORLD_HALF: f32 = 2000.0;
const NUM_DISCRETE_STATES: u8 = 3;

fn wrap_pos(x: f32, y: f32) -> (f32, f32) {
    let size = WORLD_HALF * 2.0;
    let wx = ((x + WORLD_HALF) % size + size) % size - WORLD_HALF;
    let wy = ((y + WORLD_HALF) % size + size) % size - WORLD_HALF;
    (wx, wy)
}

/// Minimum-image displacement from (ax,ay) to (bx,by) on the torus.
fn wrap_delta(a: f32, b: f32) -> f32 {
    let d = b - a;
    let size = WORLD_HALF * 2.0;
    if d > WORLD_HALF { d - size }
    else if d < -WORLD_HALF { d + size }
    else { d }
}

/// Toroidal-aware average of positions, using `ref_pos` as the anchor.
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

// ── Discrete CA helpers ────────────────────────────────────────────────────────

/// Number of discrete configurations: num_states * 7 (neighbor sums 0..6)
fn num_discrete_configs(num_states: u8) -> usize {
    (num_states as usize) * 7
}

/// config = 7 * own_state + min(neighbor_sum, 6)
fn discrete_config(own_state: u8, neighbor_sum: u32) -> usize {
    let capped = neighbor_sum.min(6) as usize;
    (own_state as usize) * 7 + capped
}

fn random_rule_table(num_states: u8) -> Vec<u8> {
    let n = num_discrete_configs(num_states);
    (0..n).map(|_| random_range(0u8, num_states)).collect()
}

/// Topology actions: for >=3 states: {merge=0, stay=1, split=2}
fn random_topo_table(num_states: u8) -> Vec<u8> {
    let n = num_discrete_configs(num_states);
    let num_actions = if num_states <= 2 { 2 } else { 3 };
    (0..n).map(|_| random_range(0u8, num_actions)).collect()
}

/// Map raw topo table output → canonical action (0=merge, 1=stay, 2=split).
fn canonical_topo_action(raw: u8, num_states: u8) -> u8 {
    if num_states <= 2 {
        if raw == 0 { 1 } else { 2 } // stay / split only
    } else {
        raw
    }
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

// ── Modal synthesis coefficient computation ───────────────────────────────────

const MODAL_BASE_FREQ: f32 = 150.0;

/// Compute Chebyshev bandpass coefficients for 8 spectral bands on the graph.
/// Returns (coeffs_lo, coeffs_hi, frequencies) where:
/// - coeffs_lo[k] = [c_k for bands 0,1,2,3]
/// - coeffs_hi[k] = [c_k for bands 4,5,6,7]
/// - frequencies[m] = natural frequency for band m
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

// ── Per-trial parameters ───────────────────────────────────────────────────────

#[derive(Clone)]
struct TrialParams {
    rule_state: Vec<u8>,
    rule_topo: Vec<u8>,
    discrete_step_interval: u32,
    max_divisions_per_step: u32,
}

fn random_trial_params() -> TrialParams {
    TrialParams {
        rule_state: random_rule_table(NUM_DISCRETE_STATES),
        rule_topo: random_topo_table(NUM_DISCRETE_STATES),
        discrete_step_interval: random_range(10u32, 40),
        max_divisions_per_step: random_range(1u32, 5),
    }
}

// ── Trial ──────────────────────────────────────────────────────────────────────

struct Trial {
    id: u32,
    spawn_frame: u64,
    initial_count: usize,
    params: TrialParams,
}

// ── Model ──────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Model {
    window: Entity,
    gpu: Option<GpuCompute>,
    render_state: Arc<Mutex<RenderState>>,

    // Global node data (all trials flattened)
    positions: Vec<(f32, f32)>,
    states: Vec<[f32; 3]>,     // placeholder for continuous state (used by upload_topology)
    connections: Vec<(usize, usize)>,
    trial_ids: Vec<u32>,
    discrete_states: Vec<u8>,

    // Barnes-Hut repulsion
    bh: BarnesHut,
    charge: f32,
    charge_max_dist: f32,
    theta: f32,

    // Shared physics
    spring_length: f32,
    spring_stiffness: f32,
    damping: f32,
    max_velocity: f32,

    node_radius: f32,

    frame: u64,
    hit_max_nodes: bool,

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

// ── App ────────────────────────────────────────────────────────────────────────

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
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("graphs.json");
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
        let base_idx = model.positions.len();
        let tid = ts.next_trial_id;
        ts.next_trial_id += 1;

        for _ in 0..count {
            model.positions.push(wrap_pos(
                center.0 + random_range(-50.0, 50.0),
                center.1 + random_range(-50.0, 50.0),
            ));
            model.states.push([0.0; 3]);
            model.trial_ids.push(tid);
            model.discrete_states.push(random_range(0u8, NUM_DISCRETE_STATES));
        }

        for &[a, b] in &graph.edgelist {
            model.connections.push((base_idx + a, base_idx + b));
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
    for &tid in &model.trial_ids {
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

    let n = model.positions.len();
    let keep: Vec<bool> = model.trial_ids.iter()
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
            new_positions.push(model.positions[i]);
            new_states.push(model.states[i]);
            new_trial_ids.push(model.trial_ids[i]);
            new_discrete.push(model.discrete_states[i]);
        }
    }

    let new_connections: Vec<(usize, usize)> = model.connections.iter()
        .filter(|&&(a, b)| a < n && b < n && keep[a] && keep[b])
        .map(|&(a, b)| (new_idx[a], new_idx[b]))
        .collect();

    model.positions = new_positions;
    model.states = new_states;
    model.trial_ids = new_trial_ids;
    model.discrete_states = new_discrete;
    model.connections = new_connections;

    if let Some(ref mut gpu) = model.gpu {
        gpu.topology_dirty = true;
    }

    true
}

fn audio_fn(audio_model: &mut AudioModel, buffer: &mut Buffer) {
    for frame in buffer.frames_mut() {
        let left = audio_model.consumer.try_pop().unwrap_or(0.0);
        let right = audio_model.consumer.try_pop().unwrap_or(0.0);
        frame[0] = left;
        frame[1] = right;
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
    let audio_model = AudioModel {
        consumer: audio_consumer,
        feedback_tx: audio_feedback_tx,
        buffer_history: VecDeque::with_capacity(BUFFER_HISTORY_SIZE),
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
        graphs: graphs.clone(),
    })).ok();

    let mut m = Model {
        window: w_id,
        gpu: None,
        render_state: Arc::new(Mutex::new(RenderState::default())),
        positions: Vec::new(),
        states: Vec::new(),
        connections: Vec::new(),
        trial_ids: Vec::new(),
        discrete_states: Vec::new(),
        bh: BarnesHut::new(),
        charge: 4.0,
        charge_max_dist: 3000.0,
        theta: 0.9,
        spring_length: 25.0,
        spring_stiffness: 1.0,
        damping: 0.1,
        max_velocity: 3.0,
        node_radius: 6.0,
        frame: 0,
        hit_max_nodes: false,
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
    };

    // Spawn initial trials in a grid
    let grid_side = (NUM_INITIAL_TRIALS as f32).sqrt().ceil() as i32;
    let offset = (grid_side as f32 - 1.0) * GRID_SPACING * 0.5;
    let mut count = 0;
    for row in 0..grid_side {
        for col in 0..grid_side {
            if count >= NUM_INITIAL_TRIALS { break; }
            let cx = col as f32 * GRID_SPACING - offset + random_range(-100.0, 100.0);
            let cy = row as f32 * GRID_SPACING - offset + random_range(-100.0, 100.0);
            spawn_trial(&mut m, (cx, cy));
            count += 1;
        }
    }

    m
}

fn update(app: &App, model: &mut Model) {
    // egui info panel
    let mut egui_ctx = app.egui_for_window(model.window);
    let ctx = egui_ctx.get_mut();

    egui::Window::new("GRA Sea").show(&ctx, |ui| {
        let n = model.positions.len();
        let num_trials = with_trials(|ts| ts.trials.len());
        ui.label(format!("trials: {}  |  nodes: {} / {}", num_trials, n, MAX_NODES));
        if model.hit_max_nodes {
            ui.colored_label(egui::Color32::YELLOW, "Max nodes reached");
        }
        ui.separator();

        // State distribution
        let mut state_counts = [0usize; NUM_DISCRETE_STATES as usize];
        for &s in &model.discrete_states {
            let idx = (s as usize).min(state_counts.len() - 1);
            state_counts[idx] += 1;
        }
        let state_colors: Vec<egui::Color32> = (0..NUM_DISCRETE_STATES)
            .map(|i| {
                let t = (i as f32) / ((NUM_DISCRETE_STATES.max(2) - 1) as f32);
                let hue = ((280.0 + t * 110.0) % 360.0) / 360.0;
                let (r, g, b) = hsv_to_rgb(hue, 0.7, 0.9);
                egui::Color32::from_rgb(
                    (r * 255.0) as u8,
                    (g * 255.0) as u8,
                    (b * 255.0) as u8,
                )
            })
            .collect();
        ui.horizontal(|ui| {
            for i in 0..NUM_DISCRETE_STATES as usize {
                ui.colored_label(state_colors[i], format!("{}: {}", i, state_counts[i]));
            }
        });
        ui.separator();

        ui.label("Discrete ternary CA (merge / stay / split)");
        ui.label("Each trial has its own random rule tables.");
        ui.separator();

        ui.label("Physics");
        ui.add(egui::Slider::new(&mut model.charge, 0.0..=10.0).text("charge"));
        ui.add(egui::Slider::new(&mut model.charge_max_dist, 100.0..=10000.0).text("charge max dist"));
        ui.add(egui::Slider::new(&mut model.spring_stiffness, 0.001..=1.5).text("spring k"));
        ui.add(egui::Slider::new(&mut model.spring_length, 5.0..=100.0).text("spring len"));
        ui.add(egui::Slider::new(&mut model.damping, 0.01..=1.0).text("damping"));
        ui.add(egui::Slider::new(&mut model.max_velocity, 0.5..=10.0).text("max vel"));
        ui.separator();
        ui.label("Audio (Modal Synthesis)");
        ui.checkbox(&mut model.audio_active, "Enable audio");
        ui.add(egui::Slider::new(&mut model.audio_volume, 0.0..=1.0).text("volume"));
        let old_base = model.modal_base_freq;
        ui.add(egui::Slider::new(&mut model.modal_base_freq, 50.0..=500.0).text("base freq"));
        if (model.modal_base_freq - old_base).abs() > 0.5 {
            model.modal_coeffs_uploaded = false; // retrigger coefficient upload
        }
        if let Some(ref fb) = model.latest_feedback {
            ui.label(format!("buf: {} / thresh: {}", fb.current_buffer_level, model.request_threshold));
        }
        ui.separator();
        ui.label("R - Reset all with new rules");
        ui.label("Scroll - Zoom  |  Right-drag - Pan");
    });
    drop(egui_ctx);

    model.frame += 1;

    // Initialize GPU on second frame
    if model.gpu.is_none() {
        if model.frame < 2 { return; }
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        model.gpu = Some(create_gpu_compute(&device, &queue));
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }

    // Upload modal coefficients once after GPU creation
    if !model.modal_coeffs_uploaded {
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

    // Upload topology if dirty
    if model.gpu.as_ref().unwrap().topology_dirty {
        let window = app.window(model.window);
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        upload_topology(&queue, gpu, &model.positions, &model.states, &model.connections);
        upload_discrete_states(&queue, gpu, &model.discrete_states, NUM_DISCRETE_STATES);
        gpu.topology_dirty = false;
    }

    if model.positions.is_empty() { return; }

    // CPU repulsion (global Barnes-Hut)
    let jitter_scale = model.bh.last_leaf_extent() * 0.5;
    let jitter = (
        random_range(-jitter_scale, jitter_scale),
        random_range(-jitter_scale, jitter_scale),
    );
    let forces = model.bh.compute_repulsion(
        &model.positions, model.charge, model.charge_max_dist, model.theta, jitter, WORLD_HALF,
    );

    // Physics-only GPU dispatch (discrete CA — no Chebyshev/growth)
    let gpu_params = GpuSimParams {
        num_nodes: model.positions.len() as u32,
        num_connections: model.connections.len() as u32,
        spring_length: model.spring_length,
        spring_stiffness: model.spring_stiffness,
        damping: model.damping,
        max_velocity: model.max_velocity,
        state_dt: 0.0,
        cheb_order: 0,
        num_channels: 3,
        world_half: WORLD_HALF,
        _pad1: 0,
        _pad2: 0,
        growth_mu: [0.0; 4],
        growth_sigma: [0.0; 4],
        coupling_row0: [0.0; 4],
        coupling_row1: [0.0; 4],
        coupling_row2: [0.0; 4],
    };

    {
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        upload_forces(&queue, gpu, &forces);
        gpu_dispatch_physics_only(&device, &queue, gpu, &gpu_params);
    }

    // Read positions every frame
    {
        let window = app.window(model.window);
        let device = window.device();
        let gpu = model.gpu.as_ref().unwrap();
        let (_bbox, positions) = readback_bbox_and_positions(&device, gpu, model.positions.len());
        model.positions = positions.into_iter()
            .map(|(x, y)| wrap_pos(x, y))
            .collect();
    }

    // Audio: PID buffer management + synthesis dispatch
    if model.audio_active && model.gpu.is_some() {
        // Process feedback from audio thread
        let mut latest_feedback = None;
        while let Ok(fb) = model.audio_feedback_rx.try_recv() {
            latest_feedback = Some(fb);
        }
        if let Some(feedback) = latest_feedback {
            model.last_buffer_level = feedback.current_buffer_level;
            model.latest_feedback = Some(feedback);

            // PID controller for buffer level
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
            let audio_buf_size = (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as usize;

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

        if need_audio && model.positions.len() > 0 {
            let base_half = WORLD_HALF;
            let cx = model.pan.x;
            let cy = model.pan.y;
            let hw = base_half / model.zoom;
            let hh = base_half / model.zoom;

            let audio_params = AudioParams {
                sample_rate: SAMPLE_RATE as f32,
                num_nodes: model.positions.len() as u32,
                chunk_size: CHUNK_SIZE,
                volume: model.audio_volume,
                current_x: [cx - hw, cx + hw],
                current_y: [cy - hh, cy + hh],
                max_speed: model.max_velocity,
                num_states: NUM_DISCRETE_STATES as u32,
                _pad0: 0,
                _pad1: 0,
            };

            let window = app.window(model.window);
            let device = window.device();
            let queue = window.queue();
            let gpu = model.gpu.as_ref().unwrap();

            update_audio_params(&queue, gpu, &audio_params);

            let write_idx = if prev_staging_idx < 0 {
                0
            } else {
                (prev_staging_idx as usize + 1) % NUM_AUDIO_STAGING_BUFS
            };

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("modal_audio_encoder"),
            });
            // Chebyshev bandpass: decompose current state into 8 spectral bands
            encode_modal_chebyshev(&mut encoder, gpu, MODAL_CHEB_ORDER);
            // Additive modal synthesis + phase update + copy to staging
            encode_modal_audio_pass(&mut encoder, gpu, write_idx);
            queue.submit(Some(encoder.finish()));

            model.last_audio_staging_idx.store(write_idx as i8, Ordering::Relaxed);
        } else {
            model.last_audio_staging_idx.store(-1, Ordering::Relaxed);
        }
    }

    // Discrete CA step — check each trial's interval
    {
        // Build trial_id → params lookup
        let trial_params_map: HashMap<u32, TrialParams> = with_trials(|ts| {
            ts.trials.iter().map(|t| (t.id, t.params.clone())).collect()
        });

        // Find which trials should step this frame
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
            let n = model.positions.len();

            // Build adjacency
            let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
            for &(a, b) in &model.connections {
                if a < n && b < n {
                    adj[a].push(b);
                    adj[b].push(a);
                }
            }

            model.discrete_states.resize(n, 1);

            // Step 1: Compute new states and topo actions for nodes whose trial is stepping
            let mut new_states = model.discrete_states.clone();
            let mut topo_actions = vec![1u8; n]; // default: stay

            for i in 0..n {
                let tid = model.trial_ids[i];
                if !stepping_trials.contains(&tid) { continue; }
                let tp = match trial_params_map.get(&tid) {
                    Some(p) => p,
                    None => continue,
                };

                let own = model.discrete_states[i];
                let neighbor_sum: u32 = adj[i].iter()
                    .map(|&nb| model.discrete_states[nb] as u32)
                    .sum();
                let cfg = discrete_config(own, neighbor_sum);

                new_states[i] = if cfg < tp.rule_state.len() {
                    tp.rule_state[cfg]
                } else {
                    own
                };
                let raw_topo = if cfg < tp.rule_topo.len() {
                    tp.rule_topo[cfg]
                } else {
                    1
                };
                topo_actions[i] = canonical_topo_action(raw_topo, NUM_DISCRETE_STATES);
            }

            model.discrete_states = new_states;

            // Step 2: Merges (topo action == 0)
            let mut merged: HashSet<usize> = HashSet::new();
            let mut to_remove: Vec<usize> = Vec::new();

            for i in 0..n {
                if topo_actions[i] != 0 { continue; }
                if merged.contains(&i) { continue; }

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

                let (a, b) = match found {
                    Some(pair) => pair,
                    None => continue,
                };

                let avg_pos = toroidal_avg(&[
                    model.positions[i], model.positions[a], model.positions[b],
                ]);
                model.positions[a] = avg_pos;

                for conn in model.connections.iter_mut() {
                    if conn.0 == i || conn.0 == b { conn.0 = a; }
                    if conn.1 == i || conn.1 == b { conn.1 = a; }
                }

                merged.insert(i);
                merged.insert(b);
                to_remove.push(i);
                to_remove.push(b);
            }

            if !to_remove.is_empty() {
                model.connections.retain(|&(a, b)| a != b);
                model.connections.sort();
                model.connections.dedup();

                to_remove.sort();
                to_remove.dedup();
                for &idx in to_remove.iter().rev() {
                    let last = model.positions.len() - 1;
                    model.positions.swap_remove(idx);
                    model.states.swap_remove(idx);
                    model.discrete_states.swap_remove(idx);
                    model.trial_ids.swap_remove(idx);

                    if idx < last {
                        for conn in model.connections.iter_mut() {
                            if conn.0 == last { conn.0 = idx; }
                            if conn.1 == last { conn.1 = idx; }
                        }
                    }
                    let new_len = model.positions.len();
                    model.connections.retain(|&(a, b)| a < new_len && b < new_len && a != b);
                }

                model.connections.sort();
                model.connections.dedup();
            }

            // Step 3: Splits (topo action == 2)
            let n = model.positions.len();
            let mut new_nodes: Vec<((f32, f32), u8, u32)> = Vec::new();
            let mut new_connections = Vec::new();

            // Per-trial division counts
            let mut div_counts: HashMap<u32, usize> = HashMap::new();

            if n < MAX_NODES {
                for i in 0..n {
                    if i >= topo_actions.len() || topo_actions[i] != 2 { continue; }
                    if merged.contains(&i) { continue; }
                    if n + new_nodes.len() + 2 > MAX_NODES {
                        model.hit_max_nodes = true;
                        break;
                    }

                    let tid = model.trial_ids[i];
                    let max_div = trial_params_map.get(&tid)
                        .map(|p| p.max_divisions_per_step as usize)
                        .unwrap_or(3);
                    let count = div_counts.entry(tid).or_default();
                    if *count >= max_div { continue; }

                    let (px, py) = model.positions[i];
                    let state = model.discrete_states[i];

                    let na = wrap_pos(
                        px + random_range(-10.0, 10.0),
                        py + random_range(-10.0, 10.0),
                    );
                    let nb = wrap_pos(
                        px + random_range(-10.0, 10.0),
                        py + random_range(-10.0, 10.0),
                    );

                    let na_idx = n + new_nodes.len();
                    new_nodes.push((na, state, tid));
                    let nb_idx = n + new_nodes.len();
                    new_nodes.push((nb, state, tid));

                    new_connections.push((i, na_idx));
                    new_connections.push((i, nb_idx));
                    new_connections.push((na_idx, nb_idx));

                    let mut counter = 0;
                    for conn in model.connections.iter_mut() {
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
                    model.positions.push((*x, *y));
                    model.states.push([0.0; 3]);
                    model.discrete_states.push(*ds);
                    model.trial_ids.push(*tid);
                }
                model.connections.extend_from_slice(&new_connections);
            }

            // Re-upload topology + discrete colors
            if !to_remove.is_empty() || !new_nodes.is_empty() {
                let window = app.window(model.window);
                let queue = window.queue();
                let gpu = model.gpu.as_mut().unwrap();
                upload_topology(&queue, gpu, &model.positions, &model.states, &model.connections);
                upload_discrete_states(&queue, gpu, &model.discrete_states, NUM_DISCRETE_STATES);
            } else {
                // Just update colors for state changes
                let window = app.window(model.window);
                let queue = window.queue();
                let gpu = model.gpu.as_ref().unwrap();
                upload_discrete_states(&queue, gpu, &model.discrete_states, NUM_DISCRETE_STATES);
            }
        }
    }

    // Viability check every 30 frames (after grace period)
    if model.frame % 30 == 0 && model.frame > GRACE_FRAMES {
        cull_inviable_trials(model);
    }

    // Spawn new trials periodically if room
    if model.frame % SPAWN_INTERVAL == 0 {
        let num_trials = with_trials(|ts| ts.trials.len());
        if num_trials < MAX_TRIALS && model.positions.len() + 20 < MAX_NODES {
            let cx = random_range(-WORLD_HALF * 0.8, WORLD_HALF * 0.8);
            let cy = random_range(-WORLD_HALF * 0.8, WORLD_HALF * 0.8);
            spawn_trial(model, (cx, cy));
            if let Some(ref mut gpu) = model.gpu {
                gpu.topology_dirty = true;
            }
        }
    }

    // Update render uniforms — fixed viewbox with zoom/pan
    {
        let base_half = WORLD_HALF;
        let cx = model.pan.x;
        let cy = model.pan.y;
        let hw = base_half / model.zoom;
        let hh = base_half / model.zoom;

        let window = app.window(model.window);
        let win_rect = window.rect();
        let window_aspect = win_rect.w() / win_rect.h();

        let uniforms = RenderUniforms {
            min_x: cx - hw, min_y: cy - hh,
            max_x: cx + hw, max_y: cy + hh,
            node_radius: model.node_radius,
            num_nodes: model.positions.len() as u32,
            num_connections: model.connections.len() as u32,
            window_aspect,
            num_channels: 3,
            world_half: WORLD_HALF,
            _pad1: 0,
            _pad2: 0,
        };
        let queue = window.queue();
        let gpu = model.gpu.as_ref().unwrap();
        update_render_uniforms(&queue, gpu, &uniforms);
    }
}

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
        model.pan.y -= delta.y * view_half * 2.0 / 900.0;
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
        model.positions.clear();
        model.states.clear();
        model.connections.clear();
        model.trial_ids.clear();
        model.discrete_states.clear();
        model.frame = 0;
        model.hit_max_nodes = false;
        model.zoom = 1.0;
        model.pan = Vec2::ZERO;

        with_trials(|ts| {
            ts.trials.clear();
            ts.next_trial_id = 0;
        });

        let grid_side = (NUM_INITIAL_TRIALS as f32).sqrt().ceil() as i32;
        let offset = (grid_side as f32 - 1.0) * GRID_SPACING * 0.5;
        let mut count = 0;
        for row in 0..grid_side {
            for col in 0..grid_side {
                if count >= NUM_INITIAL_TRIALS { break; }
                let cx = col as f32 * GRID_SPACING - offset + random_range(-100.0, 100.0);
                let cy = row as f32 * GRID_SPACING - offset + random_range(-100.0, 100.0);
                spawn_trial(model, (cx, cy));
                count += 1;
            }
        }

        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }
}
