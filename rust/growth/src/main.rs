#![allow(dead_code)]

mod mesh;
mod mesh_builders;
mod gpu;
mod render;

use std::sync::{Arc, Mutex};

use nannou::prelude::*;
use bevy::input::mouse::MouseWheel;

use mesh::HalfEdgeMesh;
use mesh_builders::{StartShape, make_start_mesh};
use gpu::{GpuCompute, GpuSimParams, create_gpu_compute, upload_mesh_to_gpu, readback_from_gpu,
          readback_bbox_only, rebuild_render_indices, update_render_uniforms, gpu_dispatch_frame};
use render::RenderState;

const MAX_BINS_PER_DIM: u32 = 64;

// ── Model ──────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Model {
    window: Entity,
    mesh: Arc<HalfEdgeMesh>,
    gpu: Option<GpuCompute>,
    render_state: Arc<Mutex<RenderState>>,
    // Cached bounding box for rendering (updated on readback frames)
    cached_center: Vec3,
    cached_scale: f32,
    // Cached spatial hash bounding box from GPU reduction [min_x, min_y, max_x, max_y, min_z, max_z]
    cached_spatial_bbox: [f32; 6],
    spring_len: f32,
    elastic_constant: f32,
    repulsion_distance: f32,
    repulsion_strength: f32,
    bulge_strength: f32,
    planar_strength: f32,
    dt: f32,
    state_dt: f32,
    damping: f32,
    // Lenia-style params
    kernel_mu: f32,
    kernel_sigma: f32,
    growth_mu: f32,
    growth_sigma: f32,
    split_threshold: f32,
    split_chance: f32,
    cheb_order: usize,
    cheb_coeffs: [f32; 20],
    frame: u64,
    // Camera rotation
    camera_yaw: f32,
    camera_pitch: f32,
    zoom: f32,
    dragging: bool,
    last_mouse: Vec2,
    render_mode: u32,
    show_wireframe: bool,
    start_shape: StartShape,
    ico_nu: usize,
}

fn recompute_cheb_coeffs(model: &mut Model) {
    const MAX_ORDER: usize = 20;

    // Compute all raw Gaussian coefficients
    let mut raw = [0.0f32; MAX_ORDER];
    let mut total = 0.0f32;
    for k in 0..MAX_ORDER {
        let c = (-((k as f32 - model.kernel_mu).powi(2)) / (2.0 * model.kernel_sigma * model.kernel_sigma)).exp();
        raw[k] = c;
        total += c;
    }

    // Auto-select order: smallest N capturing >= 95% of total mass
    let threshold = 0.95 * total;
    let mut cumsum = 0.0f32;
    let mut order = MAX_ORDER;
    for k in 0..MAX_ORDER {
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
    for k in model.cheb_order..MAX_ORDER {
        model.cheb_coeffs[k] = 0.0;
    }
}

fn randomize_params(model: &mut Model) {
    // Lenia-style params
    model.kernel_mu = 4.0 + random_f32() * 5.0;
    model.kernel_sigma = 0.5 + random_f32() * 2.0;
    model.growth_mu = random_f32();
    model.growth_sigma = 0.1 + random_f32() * 0.5;

    recompute_cheb_coeffs(model);
}

// ── App ────────────────────────────────────────────────────────────────────────

fn main() {
    nannou::app(model)
        .update(update)
        .render(render::render)
        .run();
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

    let mut m = Model {
        window: w_id,
        mesh: Arc::from(HalfEdgeMesh::new()),
        gpu: None,
        render_state: Arc::new(Mutex::new(RenderState::default())),
        cached_center: Vec3::ZERO,
        cached_scale: 1.0,
        cached_spatial_bbox: [f32::NEG_INFINITY; 6],
        spring_len: 30.0,
        elastic_constant: 0.01,
        repulsion_distance: 150.0,
        repulsion_strength: 8.0,
        bulge_strength: 10.0,
        planar_strength: 0.4,
        dt: 0.05,
        state_dt: 0.05,
        damping: 0.5,
        kernel_mu: 8.0,
        kernel_sigma: 1.0,
        growth_mu: 0.5,
        growth_sigma: 0.3,
        split_threshold: 0.95,
        split_chance: 0.001,
        cheb_order: 10,
        cheb_coeffs: [0.0; 20],
        frame: 0,
        camera_yaw: 0.0,
        camera_pitch: 0.0,
        zoom: 1.0,
        dragging: false,
        last_mouse: Vec2::ZERO,
        render_mode: 0,
        show_wireframe: false,
        start_shape: StartShape::Sphere,
        ico_nu: 32,
    };
    randomize_params(&mut m);
    m.mesh = Arc::from(make_start_mesh(m.start_shape, m.spring_len, m.ico_nu));
    m
}

fn update(app: &App, model: &mut Model) {
    // egui settings panel
    let mut egui_ctx = app.egui_for_window(model.window);
    let ctx = egui_ctx.get_mut();

    let mut kernel_changed = false;
    let mut shape_changed = false;
    egui::Window::new("Settings").show(&ctx, |ui| {
        ui.label("Start shape");
        let prev_shape = model.start_shape;
        egui::ComboBox::from_id_salt("start_shape")
            .selected_text(match model.start_shape {
                StartShape::Circle => "Circle",
                StartShape::Sphere => "Sphere",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut model.start_shape, StartShape::Circle, "Circle");
                ui.selectable_value(&mut model.start_shape, StartShape::Sphere, "Sphere");
            });
        if model.start_shape == StartShape::Sphere {
            let prev_nu = model.ico_nu;
            ui.add(egui::Slider::new(&mut model.ico_nu, 2..=64).text("subdivisions"));
            if model.ico_nu != prev_nu {
                shape_changed = true;
            }
        }
        if model.start_shape != prev_shape {
            shape_changed = true;
        }
        ui.separator();
        ui.label("Kernel (ring)");
        kernel_changed |= ui.add(egui::Slider::new(&mut model.kernel_mu, 0.0..=18.0).text("mu")).changed();
        kernel_changed |= ui.add(egui::Slider::new(&mut model.kernel_sigma, 0.1..=4.0).text("sigma")).changed();
        ui.label(format!("order: {} (auto)", model.cheb_order));
        ui.separator();
        ui.label("Growth function");
        ui.add(egui::Slider::new(&mut model.growth_mu, 0.0..=1.0).text("mu"));
        ui.add(egui::Slider::new(&mut model.growth_sigma, 0.01..=1.0).text("sigma"));
        ui.separator();
        ui.label("Split");
        ui.add(egui::Slider::new(&mut model.split_threshold, 0.0..=1.0).text("threshold"));
        ui.add(egui::Slider::new(&mut model.split_chance, 0.0..=0.1).text("chance").logarithmic(true));
        ui.separator();
        ui.label("Physics");
        ui.add(egui::Slider::new(&mut model.elastic_constant, 0.01..=0.5).text("elastic"));
        ui.add(egui::Slider::new(&mut model.repulsion_strength, 0.0..=10.0).text("repulsion"));
        ui.add(egui::Slider::new(&mut model.bulge_strength, 0.0..=30.0).text("bulge"));
        ui.add(egui::Slider::new(&mut model.planar_strength, 0.0..=0.5).text("planar"));
        ui.add(egui::Slider::new(&mut model.dt, 0.01..=0.3).text("force dt"));
        ui.add(egui::Slider::new(&mut model.damping, 0.01..=1.0).text("damping"));
        ui.add(egui::Slider::new(&mut model.state_dt, 0.01..=0.5).text("state dt"));
        ui.separator();
        ui.label("Render");
        ui.checkbox(&mut model.show_wireframe, "Wireframe");
        ui.separator();
        ui.label("Controls");
        ui.label("R — Randomize params & reset");
        ui.label("T — Reset mesh (keep params)");
        ui.label("S — Save screenshot");
        ui.label("M — Toggle render mode");
        ui.label("W — Toggle wireframe");
        ui.label("Drag — Rotate camera");
        ui.label("Scroll — Zoom");
    });
    drop(egui_ctx);

    if kernel_changed {
        recompute_cheb_coeffs(model);
    }

    if shape_changed {
        model.mesh = Arc::from(make_start_mesh(model.start_shape, model.spring_len, model.ico_nu));
        model.frame = 0;
        model.camera_yaw = 0.0;
        model.camera_pitch = 0.0;
        model.zoom = 1.0;
        model.cached_spatial_bbox = [f32::NEG_INFINITY; 6];
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }

    model.frame += 1;

    // Initialize GPU on second frame (give window time to fully initialize)
    if model.gpu.is_none() {
        if model.frame < 2 { return; }
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        model.gpu = Some(create_gpu_compute(&device, &queue));
        // Mark topology dirty so initial upload happens
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }

    // Spatial hash grid params from cached GPU bounding box (3D)
    let mesh = &model.mesh;
    let bin_size = model.repulsion_distance;
    let bbox = model.cached_spatial_bbox; // [min_x, min_y, max_x, max_y, min_z, max_z]
    let (origin_x, origin_y, origin_z, num_bins_x, num_bins_y, num_bins_z) = if bbox[0].is_finite() {
        let ox = bbox[0] - bin_size;
        let oy = bbox[1] - bin_size;
        let oz = bbox[4] - bin_size;
        let nbx = (((bbox[2] - ox) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        let nby = (((bbox[3] - oy) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        let nbz = (((bbox[5] - oz) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        (ox, oy, oz, nbx, nby, nbz)
    } else {
        // First frame: compute from CPU mesh (GPU bbox not yet available)
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        for v in 0..mesh.next_vertex {
            if mesh.vertex_idx[v] < 0 { continue; }
            min_x = min_x.min(mesh.vertex_pos[v].x);
            min_y = min_y.min(mesh.vertex_pos[v].y);
            min_z = min_z.min(mesh.vertex_pos[v].z);
            max_x = max_x.max(mesh.vertex_pos[v].x);
            max_y = max_y.max(mesh.vertex_pos[v].y);
            max_z = max_z.max(mesh.vertex_pos[v].z);
        }
        let ox = min_x - bin_size;
        let oy = min_y - bin_size;
        let oz = min_z - bin_size;
        let nbx = (((max_x - ox) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        let nby = (((max_y - oy) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        let nbz = (((max_z - oz) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
        (ox, oy, oz, nbx, nby, nbz)
    };

    let gpu_params = GpuSimParams {
        num_vertices: mesh.next_vertex as u32,
        num_half_edges: mesh.next_half_edge as u32,
        repulsion_distance: model.repulsion_distance,
        spring_len: model.spring_len,
        elastic_constant: model.elastic_constant,
        bulge_strength: model.bulge_strength,
        planar_strength: model.planar_strength,
        dt: model.dt,
        origin_x,
        origin_y,
        origin_z,
        bin_size,
        num_bins_x,
        num_bins_y,
        num_bins_z,
        growth_mu: model.growth_mu,
        growth_sigma: model.growth_sigma,
        cheb_order: model.cheb_order as u32,
        repulsion_strength: model.repulsion_strength,
        state_dt: model.state_dt,
        damping: model.damping,
        _pad: [0.0; 3],
    };

    // GPU dispatch: physics + state evolution
    {
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        gpu_dispatch_frame(
            &device,
            &queue,
            gpu,
            &model.mesh,
            &gpu_params,
            model.cheb_order,
            &model.cheb_coeffs,
        );
    }

    // Read bbox every frame so spatial hash grid stays current
    {
        let window = app.window(model.window);
        let device = window.device();
        let gpu = model.gpu.as_ref().unwrap();
        let bbox = readback_bbox_only(&device, gpu);
        model.cached_spatial_bbox = bbox;

        // Update cached bounding box for render uniforms using 3D bbox from GPU
        if bbox[0].is_finite() {
            let min_pos = Vec3::new(bbox[0], bbox[1], bbox[4]);
            let max_pos = Vec3::new(bbox[2], bbox[3], bbox[5]);
            model.cached_center = (min_pos + max_pos) * 0.5;
            let half_extent = (max_pos - min_pos) * 0.5;
            let max_radius = half_extent.length() * 1.15;
            model.cached_scale = 8.0 / max_radius.max(1.0);
        }
    }

    // Full readback + topology ops only every 10/20 frames
    let needs_topology_ops = model.frame % 10 == 0;
    if needs_topology_ops || model.frame <= 1 {
        let window = app.window(model.window);
        let device = window.device();
        let queue = window.queue();
        let gpu = model.gpu.as_mut().unwrap();
        let mesh = Arc::make_mut(&mut model.mesh);
        readback_from_gpu(&device, &queue, gpu, mesh);

        // Growth (every 10 frames)
        mesh::generate_new_triangles(mesh, model.split_threshold, model.split_chance);

        // Mesh refinement (every 20 frames)
        if model.frame % 20 == 0 {
            mesh.refine_mesh();
        }

        // Upload new mesh data to GPU immediately so render indices stay in sync
        upload_mesh_to_gpu(&queue, gpu, mesh);
        gpu.topology_dirty = false;

        // Rebuild render index buffer
        rebuild_render_indices(&queue, gpu, mesh);
    }

    // Update render uniforms every frame (camera may have changed)
    {
        let window = app.window(model.window);
        let queue = window.queue();
        let gpu = model.gpu.as_ref().unwrap();
        let sz = window.size_pixels();
        let aspect = sz.x as f32 / sz.y.max(1) as f32;
        update_render_uniforms(
            &queue, gpu,
            model.cached_center, model.cached_scale,
            model.camera_yaw, model.camera_pitch,
            model.zoom, model.render_mode, model.show_wireframe, aspect,
        );
    }

    #[cfg(debug_assertions)]
    if model.frame % 100 == 0 {
        mesh::validate_mesh(&model.mesh);
    }
}

fn key_pressed(app: &App, model: &mut Model, key: KeyCode) {
    if key == KeyCode::KeyS {
        app.window(model.window)
            .save_screenshot(app.exe_name().unwrap() + ".png");
    }
    if key == KeyCode::KeyM {
        model.render_mode = (model.render_mode + 1) % 2;
    }
    if key == KeyCode::KeyW {
        model.show_wireframe = !model.show_wireframe;
    }
    if key == KeyCode::KeyR {
        randomize_params(model);
        model.mesh = Arc::from(make_start_mesh(model.start_shape, model.spring_len, model.ico_nu));
        model.frame = 0;
        model.camera_yaw = 0.0;
        model.camera_pitch = 0.0;
        model.zoom = 1.0;
        model.cached_spatial_bbox = [f32::NEG_INFINITY; 6];
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }
    if key == KeyCode::KeyT {
        model.mesh = Arc::from(make_start_mesh(model.start_shape, model.spring_len, model.ico_nu));
        model.frame = 0;
        model.camera_yaw = 0.0;
        model.camera_pitch = 0.0;
        model.zoom = 1.0;
        model.cached_spatial_bbox = [f32::NEG_INFINITY; 6];
        if let Some(ref mut gpu) = model.gpu {
            gpu.topology_dirty = true;
        }
    }
}

fn mouse_pressed(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Left {
        model.dragging = true;
    }
}

fn mouse_released(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Left {
        model.dragging = false;
    }
}

fn mouse_moved(_app: &App, model: &mut Model, pos: Vec2) {
    if model.dragging {
        let delta = pos - model.last_mouse;
        model.camera_yaw += delta.x * 0.005;
        model.camera_pitch += delta.y * 0.005;
    }
    model.last_mouse = pos;
}

fn mouse_wheel(_app: &App, model: &mut Model, wheel: MouseWheel) {
    let factor = 1.0 + wheel.y * 0.1;
    model.zoom = (model.zoom * factor).clamp(0.1, 20.0);
}
