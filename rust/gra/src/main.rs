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
          readback_states, update_render_uniforms, gpu_dispatch_frame};
use render::RenderState;

const E: f32 = 2.718281828459045;
const MAX_CHEB_ORDER: usize = 20;

fn growth(x: f32, mu: f32, sigma: f32) -> f32 {
    2.0 * E.powf(-0.5 * ((x - mu) / sigma).powi(2)) - 1.0
}

// ── Model ──────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Model {
    window: Entity,
    gpu: Option<GpuCompute>,
    render_state: Arc<Mutex<RenderState>>,

    // Node data (CPU-side for topology management)
    positions: Vec<(f32, f32)>,
    states: Vec<f32>,
    u_values: Vec<f32>,
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
    // Lenia-style kernel params
    kernel_mu: f32,
    kernel_sigma: f32,
    growth_mu: f32,
    growth_sigma: f32,
    growth_threshold: f32,
    growth_prob: f32,
    state_dt: f32,
    node_radius: f32,
    // Chebyshev expansion
    cheb_order: usize,
    cheb_coeffs: [f32; MAX_CHEB_ORDER],

    frame: u64,
    hit_max_nodes: bool,
}

fn recompute_cheb_coeffs(model: &mut Model) {
    // Compute all raw Gaussian coefficients
    let mut raw = [0.0f32; MAX_CHEB_ORDER];
    let mut total = 0.0f32;
    for k in 0..MAX_CHEB_ORDER {
        let c = (-((k as f32 - model.kernel_mu).powi(2))
            / (2.0 * model.kernel_sigma * model.kernel_sigma))
            .exp();
        raw[k] = c;
        total += c;
    }

    // Auto-select order: smallest N capturing >= 95% of total mass
    let threshold = 0.95 * total;
    let mut cumsum = 0.0f32;
    let mut order = MAX_CHEB_ORDER;
    for k in 0..MAX_CHEB_ORDER {
        cumsum += raw[k];
        if cumsum >= threshold {
            order = k + 1;
            break;
        }
    }
    model.cheb_order = order.max(2);

    // Normalize the used coefficients
    let mut sum = 0.0f32;
    for k in 0..model.cheb_order {
        sum += raw[k];
    }
    for k in 0..model.cheb_order {
        model.cheb_coeffs[k] = raw[k] / sum;
    }
    for k in model.cheb_order..MAX_CHEB_ORDER {
        model.cheb_coeffs[k] = 0.0;
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

fn random_graph(graphs: &[GraphEntry]) -> (Vec<(f32, f32)>, Vec<f32>, Vec<(usize, usize)>) {
    let graph = &graphs[random_range(0, graphs.len())];
    let count = graph.state.len();
    let positions: Vec<(f32, f32)> = (0..count)
        .map(|_| (random_range(-200.0, 200.0), random_range(-200.0, 200.0)))
        .collect();
    let states: Vec<f32> = graph.state.iter().map(|&s| s as f32).collect();
    let connections: Vec<(usize, usize)> = graph.edgelist.iter().map(|e| (e[0], e[1])).collect();
    (positions, states, connections)
}

fn model(app: &App) -> Model {
    let w_id = app.new_window()
        .size(1000, 1000)
        .key_pressed(key_pressed)
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
        u_values: vec![0.0; node_count],
        connections,
        cached_bbox: [f32::NEG_INFINITY; 4],
        bh: BarnesHut::new(),
        charge: 3.0,
        charge_max_dist: 2000.0,
        theta: 0.9,
        use_brute_force: false,
        spring_length: 25.0,
        spring_stiffness: 0.5,
        damping: 0.1,
        max_velocity: 3.0,
        kernel_mu: 4.0,
        kernel_sigma: 1.0,
        growth_mu: random_f32(),
        growth_sigma: random_f32().max(0.1),
        growth_threshold: 0.98,
        growth_prob: 0.01,
        state_dt: random_f32() * 0.1,
        node_radius: 6.0,
        cheb_order: 10,
        cheb_coeffs: [0.0; MAX_CHEB_ORDER],
        frame: 0,
        hit_max_nodes: false,
    };
    randomize_params(&mut m);
    m
}

fn randomize_params(model: &mut Model) {
    model.kernel_mu = 4.0 + random_f32() * 5.0;
    model.kernel_sigma = 0.5 + random_f32() * 2.0;
    model.growth_mu = random_f32();
    model.growth_sigma = 0.1 + random_f32() * 0.5;

    recompute_cheb_coeffs(model);
}

fn update(app: &App, model: &mut Model) {
    // egui settings panel
    let mut egui_ctx = app.egui_for_window(model.window);
    let ctx = egui_ctx.get_mut();

    let mut kernel_changed = false;
    egui::Window::new("Settings").show(&ctx, |ui| {
        ui.label("Kernel (ring)");
        kernel_changed |= ui.add(egui::Slider::new(&mut model.kernel_mu, 0.0..=18.0).text("mu")).changed();
        kernel_changed |= ui.add(egui::Slider::new(&mut model.kernel_sigma, 0.1..=4.0).text("sigma")).changed();
        ui.label(format!("order: {} (auto)", model.cheb_order));
        ui.separator();
        ui.label("Growth function");
        ui.add(egui::Slider::new(&mut model.growth_mu, 0.0..=1.0).text("growth mu"));
        ui.add(egui::Slider::new(&mut model.growth_sigma, 0.01..=1.0).text("growth sigma"));
        ui.add(egui::Slider::new(&mut model.growth_threshold, 0.0..=1.0).text("growth threshold"));
        ui.add(egui::Slider::new(&mut model.growth_prob, 0.0..=1.0).text("growth prob"));
        ui.add(egui::Slider::new(&mut model.state_dt, 0.001..=0.2).text("state dt"));
        let n = model.positions.len();
        ui.label(format!("nodes: {} / {}", n, MAX_NODES));
        if model.hit_max_nodes {
            ui.colored_label(egui::Color32::YELLOW, "Max nodes reached");
        }
        ui.separator();
        ui.label("Physics");
        ui.add(egui::Slider::new(&mut model.charge, 0.0..=100.0).text("charge"));
        ui.add(egui::Slider::new(&mut model.charge_max_dist, 100.0..=5000.0).text("charge max dist"));
        ui.add(egui::Slider::new(&mut model.theta, 0.1..=1.0).text("theta (BH accuracy)"));
        ui.checkbox(&mut model.use_brute_force, "brute force O(N²)");
        ui.add(egui::Slider::new(&mut model.spring_stiffness, 0.001..=0.5).text("spring k"));
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

    // Upload forces + dispatch GPU
    let gpu_params = GpuSimParams {
        num_nodes: model.positions.len() as u32,
        num_connections: model.connections.len() as u32,
        spring_length: model.spring_length,
        spring_stiffness: model.spring_stiffness,
        damping: model.damping,
        max_velocity: model.max_velocity,
        growth_mu: model.growth_mu,
        growth_sigma: model.growth_sigma,
        state_dt: model.state_dt,
        cheb_order: model.cheb_order as u32,
        _pad0: 0.0,
        _pad1: 0.0,
    };

    {
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        upload_forces(&queue, gpu, &forces);
        gpu_dispatch_frame(&device, &queue, gpu, &gpu_params, model.cheb_order, &model.cheb_coeffs);
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

    // State readback + topology ops every 10 frames
    if model.frame % 10 == 0 {
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        let gpu = model.gpu.as_ref().unwrap();
        let data = readback_states(&device, &queue, gpu, model.positions.len());

        // Update CPU-side data
        model.states = data.states;
        model.u_values = data.u_values;

        // Check for splits
        let mut new_nodes = Vec::new();
        let mut new_connections = Vec::new();
        let n = model.positions.len();

        if n < MAX_NODES {
            for i in 0..n {
                if n + new_nodes.len() + 2 > MAX_NODES {
                    model.hit_max_nodes = true;
                    break;
                }
                let u = model.u_values[i];
                if model.states[i] > model.growth_threshold
                    && random_f32() < model.growth_prob
                {
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
                }
            }
        }

        if !new_nodes.is_empty() {
            for ((x, y), state) in &new_nodes {
                model.positions.push((*x, *y));
                model.states.push(*state);
                model.u_values.push(0.0);
            }
            model.connections.extend_from_slice(&new_connections);

            // Re-upload topology
            let queue = window.queue();
            let gpu = model.gpu.as_mut().unwrap();
            upload_topology(&queue, gpu, &model.positions, &model.states, &model.connections);
        }
    }

    // Update render uniforms every frame
    {
        let bbox = model.cached_bbox;
        let (min_x, min_y, max_x, max_y) = if bbox[0].is_finite() {
            let padding = 0.1;
            let dx = (bbox[2] - bbox[0]) * padding;
            let dy = (bbox[3] - bbox[1]) * padding;
            (bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy)
        } else {
            (-300.0, -300.0, 300.0, 300.0)
        };

        let window = app.window(model.window);
        let win_rect = window.rect();
        let window_aspect = win_rect.w() / win_rect.h();

        let uniforms = RenderUniforms {
            min_x, min_y, max_x, max_y,
            node_radius: model.node_radius,
            num_nodes: model.positions.len() as u32,
            num_connections: model.connections.len() as u32,
            window_aspect,
        };
        let queue = window.queue();
        let gpu = model.gpu.as_ref().unwrap();
        update_render_uniforms(&queue, gpu, &uniforms);
    }
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
        model.positions = positions;
        model.states = states;
        model.u_values = vec![0.0; node_count];
        model.connections = connections;
        model.cached_bbox = [f32::NEG_INFINITY; 4];
        model.frame = 0;
        model.hit_max_nodes = false;
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }
    if key == KeyCode::KeyT {
        let graphs = load_graphs();
        let (positions, states, connections) = random_graph(&graphs);
        let node_count = positions.len();
        model.positions = positions;
        model.states = states;
        model.u_values = vec![0.0; node_count];
        model.connections = connections;
        model.cached_bbox = [f32::NEG_INFINITY; 4];
        model.frame = 0;
        model.hit_max_nodes = false;
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }
}
