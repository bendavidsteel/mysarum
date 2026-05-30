mod camera;
mod gpu;
mod render;
mod trees;

use nannou::prelude::*;
use std::sync::{Arc, Mutex};

use gpu::{ComputeParams, GpuCompute, NUM_BOIDS, VOL_W, VOL_H, VOL_D};
use trees::Trees;

const WIDTH:  u32 = 1000;
const HEIGHT: u32 = 1000;

#[derive(Clone)]
pub struct Model {
    window: Entity,
    pub gpu: Option<GpuCompute>,
    pub render_state: Arc<Mutex<render::RenderState>>,
    trees: Arc<Mutex<Trees>>,

    frame: u64,
    pub time: f32,

    // wind
    pub wind_strength: f32,
    pub wind_dir: Vec2,
    wind_angle: f32,

    // activity ramps
    pub boid_activity: f32,
    pub physarum_activity: f32,
    pub tree_activity: f32,

    // random-walked boid/physarum params
    attraction: f32,
    repulsion: f32,
    fov: f32,
    sensor_angle: f32,
    sensor_offset: f32,

    rng: u32,
}

fn main() {
    nannou::app(model).update(update).render(render::render).run();
}

fn model(app: &App) -> Model {
    let window = app.new_window()
        .size(WIDTH, HEIGHT)
        .title("The Grove")
        .key_pressed(key_pressed)
        .build();

    Model {
        window,
        gpu: None,
        render_state: Arc::new(Mutex::new(render::RenderState::default())),
        trees: Arc::new(Mutex::new(Trees::new())),
        frame: 0,
        time: 0.0,
        wind_strength: 0.0,
        wind_dir: Vec2::new(1.0, 0.0),
        wind_angle: 0.0,
        boid_activity: 0.0,
        physarum_activity: 0.0,
        tree_activity: 0.0,
        attraction: 0.02,
        repulsion: 0.05,
        fov: 0.0,
        sensor_angle: 15.0_f32.to_radians(),
        sensor_offset: 5.0,
        rng: 0x5151_2A3B,
    }
}

fn rng_next(state: &mut u32) -> f32 {
    let mut x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    (x as f32 / u32::MAX as f32) * 2.0 - 1.0   // [-1,1]
}

fn update(app: &App, model: &mut Model) {
    model.frame += 1;
    model.time += 1.0 / 60.0;

    if model.gpu.is_none() {
        if model.frame < 2 { return; }
        let window = app.window(model.window);
        model.gpu = Some(GpuCompute::new(&window.device()));
    }

    // ── Wind random-walk (ofApp::update) ────────────────────────────────────
    model.wind_strength += rng_next(&mut model.rng) * 0.005;
    model.wind_strength = model.wind_strength.clamp(0.0, 0.1);
    model.wind_angle += rng_next(&mut model.rng) * 0.1;
    model.wind_dir = Vec2::new(model.wind_angle.cos(), model.wind_angle.sin());

    // ── Activity ramps ──────────────────────────────────────────────────────
    if model.time > 0.0  && model.boid_activity     < 1.0 { model.boid_activity     += 0.01; }
    if model.time > 10.0 && model.physarum_activity < 1.0 { model.physarum_activity += 0.01; }
    if model.time > 20.0 && model.tree_activity     < 1.0 { model.tree_activity     += 0.01; }

    // ── Random-walked params ────────────────────────────────────────────────
    model.fov = (model.fov + rng_next(&mut model.rng) * 0.01).clamp(-0.8, 0.8);
    model.attraction = (model.attraction + rng_next(&mut model.rng) * 0.001).clamp(0.01, 0.1);
    model.repulsion  = (model.repulsion  + rng_next(&mut model.rng) * 0.001).clamp(0.01, 0.1);
    model.sensor_angle = (model.sensor_angle + rng_next(&mut model.rng) * 0.001)
        .clamp(15.0_f32.to_radians(), 90.0_f32.to_radians());
    model.sensor_offset = (model.sensor_offset + rng_next(&mut model.rng) * 0.1).clamp(1.0, 20.0);

    // ── Tree growth (CPU) ───────────────────────────────────────────────────
    let segs = {
        let mut t = model.trees.lock().unwrap();
        let grew = t.update(model.wind_strength, model.wind_dir, model.tree_activity);
        if grew { Some(t.segments()) } else { None }
    };

    let window = app.window(model.window);
    let device = window.device();
    let queue = window.queue();
    let gpu = model.gpu.as_mut().unwrap();
    if let Some(segs) = segs {
        gpu.upload_segments(&device, &queue, &segs);
    }

    // ── ComputeParams ───────────────────────────────────────────────────────
    let params = ComputeParams {
        vol_res:   [VOL_W as f32, VOL_H as f32, VOL_D as f32, 0.0],
        world_res: [camera::WORLD, camera::WORLD, camera::WORLD, NUM_BOIDS as f32],
        timing:    [model.time, 1.0, model.wind_strength, model.boid_activity],
        wind:      [model.wind_dir.x, model.wind_dir.y, model.physarum_activity, 0.0],
        boid_a:    [model.attraction, 300.0, 0.05, 100.0],
        boid_b:    [model.repulsion, 25.0, model.boid_activity * 2.0, 0.1],
        boid_c:    [model.fov, model.boid_activity * 0.005, 100.0, 0.0],
        phys:      [1.0, model.sensor_angle, model.sensor_offset, model.physarum_activity],
        phys2:     [1.0, 0.05, 0.98, 0.0],
        blur_dir:  [0.0, 0.0, 0.0, 0.0],
    };
    gpu.step(&device, &queue, params);
}

fn key_pressed(app: &App, model: &mut Model, key: KeyCode) {
    match key {
        KeyCode::KeyS => {
            app.window(model.window).save_screenshot(app.exe_name().unwrap() + ".png");
        }
        KeyCode::KeyR => {
            model.frame = 0;
            model.time = 0.0;
            model.boid_activity = 0.0;
            model.physarum_activity = 0.0;
            model.tree_activity = 0.0;
            model.gpu = None;
            *model.trees.lock().unwrap() = Trees::new();
        }
        _ => {}
    }
}
