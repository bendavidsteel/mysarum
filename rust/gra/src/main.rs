#![allow(dead_code)]

mod barnes_hut;
mod gpu;
mod render;

use std::sync::{Arc, Mutex};

use nannou::prelude::*;
use serde::Deserialize;

use barnes_hut::{BarnesHut, brute_force_repulsion};
use gpu::{GpuCompute, GpuSimParams, RenderUniforms, MAX_NODES,
          create_gpu_compute, upload_topology, upload_forces, readback_bbox_and_positions,
          readback_states, update_render_uniforms, gpu_dispatch_frame,
          gpu_dispatch_physics_only, upload_discrete_states};
use render::RenderState;

const E: f32 = 2.718281828459045;
const MAX_CHEB_ORDER: usize = 20;

// Discrete CA: max 8 states × (0..6 neighbor sum) = 56 max configurations
const MAX_DISCRETE_STATES: u8 = 8;

#[derive(Clone, Copy, PartialEq)]
enum CaMode {
    Continuous,
    Discrete,
}

fn growth(x: f32, mu: f32, sigma: f32) -> f32 {
    2.0 * E.powf(-0.5 * ((x - mu) / sigma).powi(2)) - 1.0
}

/// Compute a division or death signal from u_values using a separate Gaussian.
/// Returns the mean signal across channels (0..1 range, where 1 = strong match).
fn config_signal(u: &[f32; 3], mu: &[f32; 3], sigma: &[f32; 3], num_channels: u32) -> f32 {
    let n = num_channels.min(3) as usize;
    let mut sum = 0.0f32;
    for ch in 0..n {
        let s = sigma[ch].max(0.01);
        // Gaussian: peaks at 1.0 when u == mu, falls off with sigma
        sum += E.powf(-0.5 * ((u[ch] - mu[ch]) / s).powi(2));
    }
    sum / n as f32
}

// ── Model ──────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Model {
    window: Entity,
    gpu: Option<GpuCompute>,
    render_state: Arc<Mutex<RenderState>>,

    // Node data (CPU-side for topology management)
    positions: Vec<(f32, f32)>,
    states: Vec<[f32; 3]>,
    u_values: Vec<[f32; 3]>,
    connections: Vec<(usize, usize)>,

    // Cached bounding box [min_x, min_y, max_x, max_y]
    cached_bbox: [f32; 4],

    // Barnes-Hut repulsion
    bh: BarnesHut,
    charge: f32,
    charge_max_dist: f32,
    theta: f32,
    use_brute_force: bool,

    // Simulation params
    spring_length: f32,
    spring_stiffness: f32,
    damping: f32,
    max_velocity: f32,
    // Per-channel Lenia-style kernel params
    kernel_mu: [f32; 3],
    kernel_sigma: [f32; 3],
    // Per-channel growth params
    growth_mu: [f32; 3],
    growth_sigma: [f32; 3],
    // Cross-channel coupling matrix (row i = how channels affect channel i)
    coupling: [[f32; 3]; 3],
    // Division/death: separate Gaussian functions of u (neighborhood config)
    div_mu: [f32; 3],
    div_sigma: [f32; 3],
    div_threshold: f32,
    div_prob: f32,
    death_mu: [f32; 3],
    death_sigma: [f32; 3],
    death_threshold: f32,
    death_prob: f32,
    state_dt: f32,
    node_radius: f32,
    num_channels: u32,
    // Per-channel Chebyshev expansion (max order across channels, coeffs as [f32;4] per step)
    cheb_order: usize,
    cheb_coeffs: [[f32; 4]; MAX_CHEB_ORDER],

    // CA mode
    ca_mode: CaMode,

    // Discrete CA state
    num_discrete_states: u8,          // number of discrete states (2..=MAX_DISCRETE_STATES)
    discrete_states: Vec<u8>,         // per-node state ∈ {0, ..., num_discrete_states-1}
    rule_state: Vec<u8>,              // R: config → new state
    rule_topo: Vec<u8>,               // R': config → topology action
    discrete_step_interval: u32,      // frames between discrete CA steps
    max_divisions_per_step: u32,      // cap on how many nodes can split per CA step

    frame: u64,
    hit_max_nodes: bool,

    // Camera (zoom & pan)
    zoom: f32,
    pan: Vec2,
    dragging_pan: bool,
    last_mouse: Vec2,
}

/// Number of discrete configurations for a given number of states.
fn num_discrete_configs(num_states: u8) -> usize {
    (num_states as usize) * 7
}

/// Compute configuration for a node: c = 7 * s(v) + min(sum_neighbors, 6)
fn discrete_config(own_state: u8, neighbor_sum: u32) -> usize {
    let capped_sum = neighbor_sum.min(6) as usize;
    (own_state as usize) * 7 + capped_sum
}

/// Generate a random discrete state rule table (R: config → {0..num_states-1}).
fn random_rule_table(num_states: u8) -> Vec<u8> {
    let n = num_discrete_configs(num_states);
    (0..n).map(|_| random_range(0u8, num_states)).collect()
}

/// Number of topology actions for a given number of states:
/// - 2 states: {stay, split}
/// - ≥3 states: {merge, stay, split}
fn num_topo_actions(num_states: u8) -> u8 {
    if num_states <= 2 { 2 } else { 3 }
}

/// Generate a random topology rule table (R': config → topo action).
fn random_topo_table(num_states: u8) -> Vec<u8> {
    let n = num_discrete_configs(num_states);
    let num_actions = num_topo_actions(num_states);
    (0..n).map(|_| random_range(0u8, num_actions)).collect()
}

/// Map topo table output to canonical action (0=merge, 1=stay, 2=split).
/// For 2-state systems the table outputs {0=stay, 1=split} which we remap.
fn canonical_topo_action(raw: u8, num_states: u8) -> u8 {
    if num_states <= 2 {
        // 0→stay(1), 1→split(2)
        if raw == 0 { 1 } else { 2 }
    } else {
        // 0→merge, 1→stay, 2→split (already canonical)
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

fn recompute_cheb_coeffs(model: &mut Model) {
    let mut max_order = 2usize;

    // Compute per-channel raw Gaussian coefficients and orders
    let mut raw = [[0.0f32; MAX_CHEB_ORDER]; 3];
    let mut ch_order = [2usize; 3];

    for ch in 0..3 {
        let mu = model.kernel_mu[ch];
        let sigma = model.kernel_sigma[ch];
        let mut total = 0.0f32;
        for k in 0..MAX_CHEB_ORDER {
            let c = (-((k as f32 - mu).powi(2)) / (2.0 * sigma * sigma)).exp();
            raw[ch][k] = c;
            total += c;
        }

        // Auto-select order: smallest N capturing >= 95% of total mass
        let threshold = 0.95 * total;
        let mut cumsum = 0.0f32;
        let mut order = MAX_CHEB_ORDER;
        for k in 0..MAX_CHEB_ORDER {
            cumsum += raw[ch][k];
            if cumsum >= threshold {
                order = k + 1;
                break;
            }
        }
        ch_order[ch] = order.max(2);
        max_order = max_order.max(ch_order[ch]);

        // Normalize
        let sum: f32 = raw[ch][..ch_order[ch]].iter().sum();
        if sum > 0.0 {
            for k in 0..ch_order[ch] {
                raw[ch][k] /= sum;
            }
        }
        for k in ch_order[ch]..MAX_CHEB_ORDER {
            raw[ch][k] = 0.0;
        }
    }

    model.cheb_order = max_order;

    // Pack into [f32; 4] per step (xyz = channels, w = 0)
    for k in 0..MAX_CHEB_ORDER {
        model.cheb_coeffs[k] = [raw[0][k], raw[1][k], raw[2][k], 0.0];
    }
}

// ── App ────────────────────────────────────────────────────────────────────────

fn main() {
    nannou::app(model)
        .update(update)
        .render(render::render)
        .run();
}

#[derive(Deserialize)]
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

fn random_graph(graphs: &[GraphEntry]) -> (Vec<(f32, f32)>, Vec<[f32; 3]>, Vec<(usize, usize)>) {
    let graph = &graphs[random_range(0, graphs.len())];
    let count = graph.state.len();
    let positions: Vec<(f32, f32)> = (0..count)
        .map(|_| (random_range(-200.0, 200.0), random_range(-200.0, 200.0)))
        .collect();
    // Initialize all 3 channels randomly
    let states: Vec<[f32; 3]> = (0..count)
        .map(|_| [random_f32(), random_f32(), random_f32()])
        .collect();
    let connections: Vec<(usize, usize)> = graph.edgelist.iter().map(|e| (e[0], e[1])).collect();
    (positions, states, connections)
}

/// Generate a random coupling matrix. Diagonal elements are stronger (self-coupling),
/// off-diagonal elements are weaker (cross-channel influence).
fn random_coupling() -> [[f32; 3]; 3] {
    let mut m = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                m[i][j] = 0.5 + random_f32() * 0.5; // 0.5..1.0
            } else {
                m[i][j] = random_f32() * 0.5 - 0.25; // -0.25..0.25
            }
        }
    }
    m
}

fn model(app: &App) -> Model {
    let w_id = app.new_window()
        .size(1000, 1000)
        .key_pressed(key_pressed)
        .mouse_pressed(mouse_pressed)
        .mouse_released(mouse_released)
        .mouse_moved(mouse_moved)
        .mouse_wheel(mouse_wheel)
        .build();

    let graphs = load_graphs();
    let (positions, states, connections) = random_graph(&graphs);
    let node_count = positions.len();

    let mut m = Model {
        window: w_id,
        gpu: None,
        render_state: Arc::new(Mutex::new(RenderState::default())),
        positions,
        states,
        u_values: vec![[0.0; 3]; node_count],
        connections,
        cached_bbox: [f32::NEG_INFINITY; 4],
        bh: BarnesHut::new(),
        charge: 3.0,
        charge_max_dist: 2000.0,
        theta: 0.9,
        use_brute_force: false,
        spring_length: 25.0,
        spring_stiffness: 1.0,
        damping: 0.1,
        max_velocity: 3.0,
        kernel_mu: [4.0; 3],
        kernel_sigma: [1.0; 3],
        growth_mu: [random_f32(), random_f32(), random_f32()],
        growth_sigma: [
            random_f32().max(0.1),
            random_f32().max(0.1),
            random_f32().max(0.1),
        ],
        coupling: random_coupling(),
        div_mu: [0.5, 0.5, 0.5],
        div_sigma: [0.2, 0.2, 0.2],
        div_threshold: 0.8,
        div_prob: 0.01,
        death_mu: [0.0, 0.0, 0.0],
        death_sigma: [0.2, 0.2, 0.2],
        death_threshold: 0.3,
        death_prob: 0.01,
        state_dt: random_f32() * 0.1,
        node_radius: 6.0,
        num_channels: 3,
        cheb_order: 10,
        cheb_coeffs: [[0.0; 4]; MAX_CHEB_ORDER],
        ca_mode: CaMode::Continuous,
        num_discrete_states: 3,
        discrete_states: vec![1; node_count],  // all start as state 1 (stay)
        rule_state: random_rule_table(3),
        rule_topo: random_topo_table(3),
        discrete_step_interval: 20,
        max_divisions_per_step: 3,
        frame: 0,
        hit_max_nodes: false,
        zoom: 1.0,
        pan: Vec2::ZERO,
        dragging_pan: false,
        last_mouse: Vec2::ZERO,
    };
    randomize_params(&mut m);
    m
}

fn randomize_params(model: &mut Model) {
    for i in 0..3 {
        model.kernel_mu[i] = 4.0 + random_f32() * 5.0;
        model.kernel_sigma[i] = 0.5 + random_f32() * 2.0;
        model.growth_mu[i] = random_f32();
        model.growth_sigma[i] = 0.1 + random_f32() * 0.5;
        // Division params: separate Gaussian over u-space
        model.div_mu[i] = random_f32();
        model.div_sigma[i] = 0.1 + random_f32() * 0.4;
        // Death params: tend toward low u regions
        model.death_mu[i] = random_f32();
        model.death_sigma[i] = 0.1 + random_f32() * 0.4;
    }
    model.div_threshold = 0.7 + random_f32() * 0.25;     // 0.7..0.95
    model.div_prob = 0.001 + random_f32() * 0.02;         // 0.001..0.021
    model.death_threshold = 0.7 + random_f32() * 0.25;    // 0.7..0.95
    model.death_prob = 0.001 + random_f32() * 0.02;       // 0.001..0.021
    model.coupling = random_coupling();
    model.rule_state = random_rule_table(model.num_discrete_states);
    model.rule_topo = random_topo_table(model.num_discrete_states);

    recompute_cheb_coeffs(model);
}

fn update(app: &App, model: &mut Model) {
    // egui settings panel
    let mut egui_ctx = app.egui_for_window(model.window);
    let ctx = egui_ctx.get_mut();

    let mut kernel_changed = false;
    egui::Window::new("Settings").show(&ctx, |ui| {
        // CA mode toggle
        ui.horizontal(|ui| {
            ui.label("CA Mode:");
            ui.selectable_value(&mut model.ca_mode, CaMode::Continuous, "Continuous");
            ui.selectable_value(&mut model.ca_mode, CaMode::Discrete, "Discrete");
        });
        ui.separator();

        match model.ca_mode {
            CaMode::Continuous => {
                // Channel mode toggle
                let mut use_3ch = model.num_channels == 3;
                if ui.checkbox(&mut use_3ch, "3-channel RGB Lenia").changed() {
                    model.num_channels = if use_3ch { 3 } else { 1 };
                }
                ui.separator();

                let colors = [
                    egui::Color32::from_rgb(255, 100, 100),
                    egui::Color32::from_rgb(100, 255, 100),
                    egui::Color32::from_rgb(100, 100, 255),
                ];
                let ch_labels = ["R", "G", "B"];
                let n_ch = (model.num_channels as usize).max(1);

                ui.label("Kernel (ring)");
                ui.horizontal(|ui| {
                    ui.label("μ ");
                    for ch in 0..n_ch {
                        kernel_changed |= ui.add(egui::DragValue::new(&mut model.kernel_mu[ch])
                            .range(0.0..=18.0).speed(0.1).prefix(ch_labels[ch])).changed();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("σ ");
                    for ch in 0..n_ch {
                        kernel_changed |= ui.add(egui::DragValue::new(&mut model.kernel_sigma[ch])
                            .range(0.1..=4.0).speed(0.05).prefix(ch_labels[ch])).changed();
                    }
                });
                ui.label(format!("order: {} (auto)", model.cheb_order));
                ui.separator();

                if model.num_channels == 1 {
                    ui.label("Growth function");
                    ui.add(egui::Slider::new(&mut model.growth_mu[0], 0.0..=1.0).text("growth mu"));
                    ui.add(egui::Slider::new(&mut model.growth_sigma[0], 0.01..=1.0).text("growth sigma"));
                } else {
                    ui.label("Growth function (per channel)");
                    let labels = ["R", "G", "B"];
                    for ch in 0..3 {
                        ui.colored_label(colors[ch], labels[ch]);
                        ui.add(egui::Slider::new(&mut model.growth_mu[ch], 0.0..=1.0).text("mu"));
                        ui.add(egui::Slider::new(&mut model.growth_sigma[ch], 0.01..=1.0).text("sigma"));
                    }
                    ui.separator();
                    ui.label("Coupling matrix");
                    let row_labels = ["R←", "G←", "B←"];
                    for i in 0..3 {
                        ui.horizontal(|ui| {
                            ui.colored_label(colors[i], row_labels[i]);
                            for j in 0..3 {
                                ui.add(egui::DragValue::new(&mut model.coupling[i][j])
                                    .range(-2.0..=2.0)
                                    .speed(0.01)
                                    .prefix(labels[j]));
                            }
                        });
                    }
                }
                ui.separator();
                ui.label("Division signal (R')");
                let ch_labels = ["R", "G", "B"];
                let n_ch = (model.num_channels as usize).max(1);
                ui.horizontal(|ui| {
                    ui.label("μ ");
                    for ch in 0..n_ch {
                        ui.add(egui::DragValue::new(&mut model.div_mu[ch])
                            .range(0.0..=1.0).speed(0.01).prefix(ch_labels[ch]));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("σ ");
                    for ch in 0..n_ch {
                        ui.add(egui::DragValue::new(&mut model.div_sigma[ch])
                            .range(0.01..=1.0).speed(0.01).prefix(ch_labels[ch]));
                    }
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut model.div_threshold)
                        .range(0.0..=1.0).speed(0.01).prefix("thr "));
                    ui.add(egui::DragValue::new(&mut model.div_prob)
                        .range(0.0..=0.1).speed(0.001).prefix("prob "));
                });
                ui.label("Death signal");
                ui.horizontal(|ui| {
                    ui.label("μ ");
                    for ch in 0..n_ch {
                        ui.add(egui::DragValue::new(&mut model.death_mu[ch])
                            .range(0.0..=1.0).speed(0.01).prefix(ch_labels[ch]));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("σ ");
                    for ch in 0..n_ch {
                        ui.add(egui::DragValue::new(&mut model.death_sigma[ch])
                            .range(0.01..=1.0).speed(0.01).prefix(ch_labels[ch]));
                    }
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut model.death_threshold)
                        .range(0.0..=1.0).speed(0.01).prefix("thr "));
                    ui.add(egui::DragValue::new(&mut model.death_prob)
                        .range(0.0..=0.1).speed(0.001).prefix("prob "));
                });
                ui.separator();
                ui.add(egui::Slider::new(&mut model.state_dt, 0.001..=0.2).text("state dt"));
            }
            CaMode::Discrete => {
                ui.label("Discrete CA GRA");
                let mut ns = model.num_discrete_states as i32;
                if ui.add(egui::Slider::new(&mut ns, 2..=(MAX_DISCRETE_STATES as i32)).text("states")).changed() {
                    model.num_discrete_states = ns as u8;
                    model.rule_state = random_rule_table(model.num_discrete_states);
                    model.rule_topo = random_topo_table(model.num_discrete_states);
                    for s in model.discrete_states.iter_mut() {
                        if *s >= model.num_discrete_states {
                            *s = random_range(0u8, model.num_discrete_states);
                        }
                    }
                }
                let ns = model.num_discrete_states;
                ui.label(if ns <= 2 {
                    "R': stay / split".to_string()
                } else {
                    "R': merge / stay / split".to_string()
                });
                ui.separator();
                ui.add(egui::Slider::new(&mut model.discrete_step_interval, 1..=60).text("step interval"));
                ui.add(egui::Slider::new(&mut model.max_divisions_per_step, 1..=20).text("max splits/step"));
                if ui.button("Randomize Rules").clicked() {
                    model.rule_state = random_rule_table(model.num_discrete_states);
                    model.rule_topo = random_topo_table(model.num_discrete_states);
                }
                ui.separator();

                // State distribution
                let mut counts = vec![0usize; model.num_discrete_states as usize];
                for &s in &model.discrete_states {
                    let idx = (s as usize).min(counts.len() - 1);
                    counts[idx] += 1;
                }
                // Generate colors for each state via hue rotation
                let state_colors: Vec<egui::Color32> = (0..model.num_discrete_states)
                    .map(|i| {
                        let t = (i as f32) / ((model.num_discrete_states.max(2) - 1) as f32);
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
                    for i in 0..model.num_discrete_states as usize {
                        ui.colored_label(state_colors[i], format!("{}: {}", i, counts[i]));
                    }
                });
            }
        }

        let n = model.positions.len();
        ui.label(format!("nodes: {} / {}", n, MAX_NODES));
        if model.hit_max_nodes {
            ui.colored_label(egui::Color32::YELLOW, "Max nodes reached");
        }
        ui.separator();
        ui.label("Physics");
        ui.add(egui::Slider::new(&mut model.charge, 0.0..=5.0).text("charge"));
        ui.add(egui::Slider::new(&mut model.charge_max_dist, 100.0..=5000.0).text("charge max dist"));
        ui.add(egui::Slider::new(&mut model.theta, 0.1..=1.0).text("theta (BH accuracy)"));
        ui.checkbox(&mut model.use_brute_force, "brute force O(N²)");
        ui.add(egui::Slider::new(&mut model.spring_stiffness, 0.001..=1.5).text("spring k"));
        ui.add(egui::Slider::new(&mut model.spring_length, 5.0..=100.0).text("spring len"));
        ui.add(egui::Slider::new(&mut model.damping, 0.01..=1.0).text("damping"));
        ui.add(egui::Slider::new(&mut model.max_velocity, 0.5..=10.0).text("max vel"));
        ui.separator();
        ui.label("R — Randomize & reset");
        ui.label("T — Reset graph (keep params)");
        ui.label("S — Save screenshot");
    });
    drop(egui_ctx);

    if kernel_changed {
        recompute_cheb_coeffs(model);
    }

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

    // Upload topology if dirty
    if model.gpu.as_ref().unwrap().topology_dirty {
        let window = app.window(model.window);
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        upload_topology(&queue, gpu, &model.positions, &model.states, &model.connections);
        // In discrete mode, overwrite state colors with discrete state mapping
        if model.ca_mode == CaMode::Discrete {
            upload_discrete_states(&queue, gpu, &model.discrete_states, model.num_discrete_states);
        }
        gpu.topology_dirty = false;
    }

    // CPU repulsion (BH or brute force)
    let forces = if model.use_brute_force {
        brute_force_repulsion(&model.positions, model.charge, model.charge_max_dist)
    } else {
        // Jitter tree origin by up to half a leaf cell to randomize cell boundaries.
        // This prevents the quadtree grid from imprinting on the force field.
        let jitter_scale = model.bh.last_leaf_extent() * 0.5;
        let jitter = (
            random_range(-jitter_scale, jitter_scale),
            random_range(-jitter_scale, jitter_scale),
        );
        model.bh.compute_repulsion(&model.positions, model.charge, model.charge_max_dist, model.theta, jitter)
    };

    // Upload forces + dispatch GPU (mode-dependent)
    let gpu_params = GpuSimParams {
        num_nodes: model.positions.len() as u32,
        num_connections: model.connections.len() as u32,
        spring_length: model.spring_length,
        spring_stiffness: model.spring_stiffness,
        damping: model.damping,
        max_velocity: model.max_velocity,
        state_dt: model.state_dt,
        cheb_order: model.cheb_order as u32,
        num_channels: if model.ca_mode == CaMode::Discrete { 3 } else { model.num_channels },
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        growth_mu: [model.growth_mu[0], model.growth_mu[1], model.growth_mu[2], 0.0],
        growth_sigma: [model.growth_sigma[0], model.growth_sigma[1], model.growth_sigma[2], 0.0],
        coupling_row0: [model.coupling[0][0], model.coupling[0][1], model.coupling[0][2], 0.0],
        coupling_row1: [model.coupling[1][0], model.coupling[1][1], model.coupling[1][2], 0.0],
        coupling_row2: [model.coupling[2][0], model.coupling[2][1], model.coupling[2][2], 0.0],
    };

    {
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        upload_forces(&queue, gpu, &forces);
        match model.ca_mode {
            CaMode::Continuous => {
                gpu_dispatch_frame(&device, &queue, gpu, &gpu_params, model.cheb_order, &model.cheb_coeffs);
            }
            CaMode::Discrete => {
                gpu_dispatch_physics_only(&device, &queue, gpu, &gpu_params);
            }
        }
    }

    // Read bbox + positions every frame (positions needed for BH tree accuracy)
    {
        let window = app.window(model.window);
        let device = window.device();
        let gpu = model.gpu.as_ref().unwrap();
        let (bbox, positions) = readback_bbox_and_positions(&device, gpu, model.positions.len());
        model.cached_bbox = bbox;
        model.positions = positions;
    }

    match model.ca_mode {
        CaMode::Continuous => {
            // Continuous: state readback + topology ops every 10 frames
            if model.frame % 10 == 0 {
                let window = app.window(model.window);
                let device = window.device();
                let queue = window.queue();
                let gpu = model.gpu.as_ref().unwrap();
                let data = readback_states(&device, &queue, gpu, model.positions.len());

                model.states = data.states;
                model.u_values = data.u_values;

                // Triangle merges (death)
                {
                    use std::collections::HashSet;
                    let n = model.positions.len();

                    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
                    for &(a, b) in &model.connections {
                        if a < n && b < n {
                            adj[a].insert(b);
                            adj[b].insert(a);
                        }
                    }

                    let mut merged: HashSet<usize> = HashSet::new();
                    let mut to_remove: Vec<usize> = Vec::new();

                    for i in 0..n {
                        if merged.contains(&i) { continue; }
                        let death_sig = config_signal(
                            &model.u_values[i], &model.death_mu, &model.death_sigma, model.num_channels,
                        );
                        if death_sig < model.death_threshold { continue; }
                        if random_f32() >= model.death_prob { continue; }

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

                        let avg_state = [
                            (model.states[i][0] + model.states[a][0] + model.states[b][0]) / 3.0,
                            (model.states[i][1] + model.states[a][1] + model.states[b][1]) / 3.0,
                            (model.states[i][2] + model.states[a][2] + model.states[b][2]) / 3.0,
                        ];
                        let avg_pos = (
                            (model.positions[i].0 + model.positions[a].0 + model.positions[b].0) / 3.0,
                            (model.positions[i].1 + model.positions[a].1 + model.positions[b].1) / 3.0,
                        );
                        model.states[a] = avg_state;
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
                            model.u_values.swap_remove(idx);

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

                        if let Some(ref mut gpu) = model.gpu {
                            let queue = window.queue();
                            upload_topology(&queue, gpu, &model.positions, &model.states, &model.connections);
                            if model.ca_mode == CaMode::Discrete {
                                upload_discrete_states(&queue, gpu, &model.discrete_states, model.num_discrete_states);
                            }
                        }
                    }
                }

                // Splits (division)
                let mut new_nodes = Vec::new();
                let mut new_connections = Vec::new();
                let n = model.positions.len();

                if n < MAX_NODES {
                    for i in 0..n {
                        if n + new_nodes.len() + 2 > MAX_NODES {
                            model.hit_max_nodes = true;
                            break;
                        }
                        let div_sig = config_signal(
                            &model.u_values[i], &model.div_mu, &model.div_sigma, model.num_channels,
                        );
                        if div_sig > model.div_threshold && random_f32() < model.div_prob {
                            let (px, py) = model.positions[i];
                            let state = model.states[i];

                            let na = (
                                px + random_range(-10.0, 10.0),
                                py + random_range(-10.0, 10.0),
                            );
                            let nb = (
                                px + random_range(-10.0, 10.0),
                                py + random_range(-10.0, 10.0),
                            );

                            let na_idx = n + new_nodes.len();
                            new_nodes.push((na, state));
                            let nb_idx = n + new_nodes.len();
                            new_nodes.push((nb, state));

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
                        }
                    }
                }

                if !new_nodes.is_empty() {
                    for ((x, y), state) in &new_nodes {
                        model.positions.push((*x, *y));
                        model.states.push(*state);
                        model.u_values.push([0.0; 3]);
                    }
                    model.connections.extend_from_slice(&new_connections);

                    let queue = window.queue();
                    let gpu = model.gpu.as_mut().unwrap();
                    upload_topology(&queue, gpu, &model.positions, &model.states, &model.connections);
                }
            }
        }
        CaMode::Discrete => {
            // Discrete CA: apply rule every N frames
            if model.frame % model.discrete_step_interval as u64 == 0 {
                use std::collections::HashSet;
                let n = model.positions.len();

                // Build adjacency
                let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
                for &(a, b) in &model.connections {
                    if a < n && b < n {
                        adj[a].push(b);
                        adj[b].push(a);
                    }
                }

                // Ensure discrete_states is correct length
                model.discrete_states.resize(n, 1);

                // Step 1: Compute new states and topology actions for all nodes simultaneously
                let mut new_states = vec![0u8; n];
                let mut topo_actions = vec![1u8; n]; // default: stay
                let ns = model.num_discrete_states;
                for i in 0..n {
                    let own = model.discrete_states[i];
                    let neighbor_sum: u32 = adj[i].iter()
                        .map(|&nb| model.discrete_states[nb] as u32)
                        .sum();
                    let cfg = discrete_config(own, neighbor_sum);
                    new_states[i] = if cfg < model.rule_state.len() {
                        model.rule_state[cfg]
                    } else {
                        own
                    };
                    let raw_topo = if cfg < model.rule_topo.len() {
                        model.rule_topo[cfg]
                    } else {
                        1 // stay
                    };
                    topo_actions[i] = canonical_topo_action(raw_topo, ns);
                }

                // Step 2: Apply new states
                model.discrete_states = new_states;

                // Step 3: Apply topology changes
                // First pass: merges (action == 0)
                // Find triangles to merge, similar to continuous mode
                let mut merged: HashSet<usize> = HashSet::new();
                let mut to_remove: Vec<usize> = Vec::new();

                for i in 0..n {
                    if topo_actions[i] != 0 { continue; }
                    if merged.contains(&i) { continue; }

                    // Find a triangle containing i
                    let neighbours: Vec<usize> = adj[i].iter().copied()
                        .filter(|&nb| !merged.contains(&nb))
                        .collect();
                    let mut found = None;
                    'tri_d: for ni in 0..neighbours.len() {
                        let a = neighbours[ni];
                        for nj in (ni + 1)..neighbours.len() {
                            let b = neighbours[nj];
                            if adj[a].contains(&b) {
                                found = Some((a, b));
                                break 'tri_d;
                            }
                        }
                    }

                    let (a, b) = match found {
                        Some(pair) => pair,
                        None => continue,
                    };

                    // Survivor = a. Average position.
                    let avg_pos = (
                        (model.positions[i].0 + model.positions[a].0 + model.positions[b].0) / 3.0,
                        (model.positions[i].1 + model.positions[a].1 + model.positions[b].1) / 3.0,
                    );
                    model.positions[a] = avg_pos;
                    // Survivor keeps its new discrete state (already updated)

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
                        model.discrete_states.swap_remove(idx);
                        model.states.swap_remove(idx);
                        model.u_values.swap_remove(idx);

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

                // Second pass: splits (action == 2)
                // Re-read n after merges
                let n = model.positions.len();
                let mut new_nodes: Vec<((f32, f32), u8)> = Vec::new();
                let mut new_connections = Vec::new();

                let max_div = model.max_divisions_per_step as usize;
                let mut div_count = 0usize;
                if n < MAX_NODES {
                    for i in 0..n {
                        if div_count >= max_div { break; }
                        // Only split nodes whose topo_action was 2 and weren't removed
                        if i >= topo_actions.len() || topo_actions[i] != 2 { continue; }
                        if merged.contains(&i) { continue; }
                        if n + new_nodes.len() + 2 > MAX_NODES {
                            model.hit_max_nodes = true;
                            break;
                        }

                        let (px, py) = model.positions[i];
                        let state = model.discrete_states[i];

                        let na = (
                            px + random_range(-10.0, 10.0),
                            py + random_range(-10.0, 10.0),
                        );
                        let nb = (
                            px + random_range(-10.0, 10.0),
                            py + random_range(-10.0, 10.0),
                        );

                        let na_idx = n + new_nodes.len();
                        new_nodes.push((na, state));
                        let nb_idx = n + new_nodes.len();
                        new_nodes.push((nb, state));

                        new_connections.push((i, na_idx));
                        new_connections.push((i, nb_idx));
                        new_connections.push((na_idx, nb_idx));

                        // Redistribute first 2 connections of parent to children
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
                        div_count += 1;
                    }
                }

                if !new_nodes.is_empty() {
                    for ((x, y), ds) in &new_nodes {
                        model.positions.push((*x, *y));
                        model.discrete_states.push(*ds);
                        model.states.push([0.0; 3]); // placeholder for continuous state
                        model.u_values.push([0.0; 3]);
                    }
                    model.connections.extend_from_slice(&new_connections);
                }

                // Immediately re-upload topology + discrete colors so the render
                // this frame sees consistent connection indices and node count.
                let window = app.window(model.window);
                let queue = window.queue();
                let gpu = model.gpu.as_mut().unwrap();
                if !to_remove.is_empty() || !new_nodes.is_empty() {
                    upload_topology(&queue, gpu, &model.positions, &model.states, &model.connections);
                }
                upload_discrete_states(&queue, gpu, &model.discrete_states, model.num_discrete_states);
            }
        }
    }

    // Update render uniforms every frame (apply zoom & pan)
    {
        let bbox = model.cached_bbox;
        let (base_min_x, base_min_y, base_max_x, base_max_y) = if bbox[0].is_finite() {
            let padding = 0.1;
            let dx = (bbox[2] - bbox[0]) * padding;
            let dy = (bbox[3] - bbox[1]) * padding;
            (bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy)
        } else {
            (-300.0, -300.0, 300.0, 300.0)
        };

        // Apply zoom and pan: shrink extents by zoom, shift center by pan
        let cx = (base_min_x + base_max_x) * 0.5 + model.pan.x;
        let cy = (base_min_y + base_max_y) * 0.5 + model.pan.y;
        let hw = (base_max_x - base_min_x) * 0.5 / model.zoom;
        let hh = (base_max_y - base_min_y) * 0.5 / model.zoom;

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
            num_channels: if model.ca_mode == CaMode::Discrete { 3 } else { model.num_channels },
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let queue = window.queue();
        let gpu = model.gpu.as_ref().unwrap();
        update_render_uniforms(&queue, gpu, &uniforms);
    }
}

fn mouse_pressed(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Right || button == MouseButton::Middle {
        model.dragging_pan = true;
    }
}

fn mouse_released(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Right || button == MouseButton::Middle {
        model.dragging_pan = false;
    }
}

fn mouse_moved(_app: &App, model: &mut Model, pos: Vec2) {
    if model.dragging_pan {
        let delta = pos - model.last_mouse;
        // Convert pixel delta to world-space offset (negate because moving mouse right
        // should shift the view left, i.e. pan right in world space)
        model.pan -= delta / model.zoom;
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
        randomize_params(model);
        let graphs = load_graphs();
        let (positions, states, connections) = random_graph(&graphs);
        let node_count = positions.len();
        let ns = model.num_discrete_states;
        model.discrete_states = (0..node_count).map(|_| random_range(0u8, ns)).collect();
        model.positions = positions;
        model.states = states;
        model.u_values = vec![[0.0; 3]; node_count];
        model.connections = connections;
        model.cached_bbox = [f32::NEG_INFINITY; 4];
        model.frame = 0;
        model.hit_max_nodes = false;
        model.zoom = 1.0;
        model.pan = Vec2::ZERO;
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }
    if key == KeyCode::KeyT {
        let graphs = load_graphs();
        let (positions, states, connections) = random_graph(&graphs);
        let node_count = positions.len();
        model.discrete_states = (0..node_count).map(|_| random_range(0u8, model.num_discrete_states)).collect();
        model.positions = positions;
        model.states = states;
        model.u_values = vec![[0.0; 3]; node_count];
        model.connections = connections;
        model.cached_bbox = [f32::NEG_INFINITY; 4];
        model.frame = 0;
        model.hit_max_nodes = false;
        model.zoom = 1.0;
        model.pan = Vec2::ZERO;
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }
}
