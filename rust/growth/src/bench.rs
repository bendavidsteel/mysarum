// Headless benchmark for the per-frame GPU simulation step.
//
// Gated behind the GROWTH_BENCH env var so it never runs in the normal app.
// It reuses the *real* create_gpu_compute + gpu_dispatch_frame so the numbers
// reflect production code, not a reimplementation.
//
//   GROWTH_BENCH=1 cargo run --release
//
// It answers one question: how long does one gpu_dispatch_frame take on this
// machine, relative to the ~16.7 ms vsync budget? If it is a small fraction,
// running multiple sim substeps per presented frame ("decouple from vsync")
// scales sim throughput nearly linearly until the GPU saturates.

use std::time::Instant;

use crate::gpu::{GpuSimParams, StepProfiler, create_gpu_compute, gpu_dispatch_frame};
use crate::mesh::HalfEdgeMesh;
use crate::mesh_builders::{StartShape, make_start_mesh};

const MAX_BINS_PER_DIM: u32 = 64;
const SPRING_LEN: f32 = 30.0;
const REPULSION_DISTANCE: f32 = 80.0;
const VSYNC_BUDGET_MS: f64 = 1000.0 / 60.0;

/// Gaussian Chebyshev coefficients + auto-selected order (mirrors
/// recompute_cheb_coeffs in main.rs with the default kernel mu/sigma).
fn cheb_coeffs(kernel_mu: f32, kernel_sigma: f32) -> (usize, [f32; 20]) {
    const MAX_ORDER: usize = 20;
    let mut raw = [0.0f32; MAX_ORDER];
    let mut total = 0.0f32;
    for k in 0..MAX_ORDER {
        let c = (-((k as f32 - kernel_mu).powi(2)) / (2.0 * kernel_sigma * kernel_sigma)).exp();
        raw[k] = c;
        total += c;
    }
    let threshold = 0.95 * total;
    let mut cumsum = 0.0f32;
    let mut order = MAX_ORDER;
    for k in 0..MAX_ORDER {
        cumsum += raw[k];
        if cumsum >= threshold { order = k + 1; break; }
    }
    let order = order.max(2);
    let mut coeffs = [0.0f32; 20];
    let sum: f32 = raw[..order].iter().sum();
    for k in 0..order { coeffs[k] = raw[k] / sum; }
    (order, coeffs)
}

/// Build sim params for a mesh using the same CPU-bbox path update() uses on
/// the first frame (before a GPU bbox is available).
fn build_params(mesh: &HalfEdgeMesh, cheb_order: usize) -> GpuSimParams {
    let bin_size = REPULSION_DISTANCE;
    let (mut min_x, mut min_y, mut min_z) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y, mut max_z) =
        (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in 0..mesh.next_vertex {
        if mesh.vertex_idx[v] < 0 { continue; }
        let p = mesh.vertex_pos[v];
        min_x = min_x.min(p.x); min_y = min_y.min(p.y); min_z = min_z.min(p.z);
        max_x = max_x.max(p.x); max_y = max_y.max(p.y); max_z = max_z.max(p.z);
    }
    let ox = min_x - bin_size;
    let oy = min_y - bin_size;
    let oz = min_z - bin_size;
    let nbx = (((max_x - ox) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
    let nby = (((max_y - oy) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);
    let nbz = (((max_z - oz) / bin_size).ceil() as u32 + 2).min(MAX_BINS_PER_DIM);

    GpuSimParams {
        num_vertices: mesh.next_vertex as u32,
        num_half_edges: mesh.next_half_edge as u32,
        repulsion_distance: REPULSION_DISTANCE,
        spring_len: SPRING_LEN,
        compliance: 0.0,
        bulge_strength: 5.0,
        smoothing_strength: 400.0,
        dt: 0.02,
        origin_x: ox,
        origin_y: oy,
        origin_z: oz,
        bin_size,
        num_bins_x: nbx,
        num_bins_y: nby,
        num_bins_z: nbz,
        growth_mu: 0.5,
        growth_sigma: 0.2,
        cheb_order: cheb_order as u32,
        state_dt: 0.02,
        damping: 10.0,
        growth_rate: 0.3,
        xpbd_iterations: 20,
        bending_compliance: 2.5,
        relaxation: 0.7,
        growth_mode: 0, // Lenia
        frame_seed: 0,
        floor_enabled: 0.0,
        floor_z: 0.0,
        anisotropy_axis: 2,
        anisotropy_strength: 0.0,
        dot_diffusion: 0.8,
        dot_decay: 0.02,
        inner_radius: 0.0,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    }
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench_size(device: &wgpu::Device, queue: &wgpu::Queue, ico_nu: usize) {
    let mesh = make_start_mesh(StartShape::Sphere, SPRING_LEN, ico_nu);
    let n_verts = mesh.active_vertex_count();
    let (order, coeffs) = cheb_coeffs(6.0, 1.5);
    let params = build_params(&mesh, order);

    let mut gpu = create_gpu_compute(device, queue);
    gpu.topology_dirty = true; // force initial upload on first dispatch

    // Warmup (compile pipelines, populate caches, first upload).
    for f in 0..15 {
        let mut p = params;
        p.frame_seed = f;
        gpu_dispatch_frame(device, queue, &mut gpu, &mesh, &p, order, &coeffs, None);
    }
    device.poll(wgpu::PollType::Wait).unwrap();

    // (A) Per-step latency: dispatch + sync each step.
    let n_lat = 60u32;
    let mut lat = Vec::with_capacity(n_lat as usize);
    for f in 0..n_lat {
        let mut p = params;
        p.frame_seed = 1000 + f;
        let t = Instant::now();
        gpu_dispatch_frame(device, queue, &mut gpu, &mesh, &p, order, &coeffs, None);
        device.poll(wgpu::PollType::Wait).unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let lat_med = median(&mut lat);

    // (B) Pipelined throughput: K dispatches, single sync. This is exactly the
    // cost model of running K substeps per presented frame.
    let k = 200u32;
    let t = Instant::now();
    for f in 0..k {
        let mut p = params;
        p.frame_seed = 2000 + f;
        gpu_dispatch_frame(device, queue, &mut gpu, &mesh, &p, order, &coeffs, None);
    }
    device.poll(wgpu::PollType::Wait).unwrap();
    let per_step_pipelined = t.elapsed().as_secs_f64() * 1000.0 / k as f64;

    let substeps = (VSYNC_BUDGET_MS / per_step_pipelined).floor().max(1.0);
    println!(
        "  nu={:>3}  verts={:>6}  he={:>6}  | latency/step {:>7.3} ms | pipelined/step {:>7.3} ms | substeps@60fps ~{:>4.0}  ({:>5.1}% of 16.7ms)",
        ico_nu, n_verts, mesh.next_half_edge,
        lat_med, per_step_pipelined, substeps,
        per_step_pipelined / VSYNC_BUDGET_MS * 100.0,
    );
}

/// Per-pass GPU-time breakdown for one mesh size, via timestamp queries.
/// Accumulates each segment's total over many steps (XPBD segments repeat
/// per iteration, so they sum into one springs+bending and one collision
/// total per step) and reports the average step's composition.
fn profile_size(device: &wgpu::Device, queue: &wgpu::Queue, ico_nu: usize) {
    let mesh = make_start_mesh(StartShape::Sphere, SPRING_LEN, ico_nu);
    let n_verts = mesh.active_vertex_count();
    let (order, coeffs) = cheb_coeffs(6.0, 1.5);
    let params = build_params(&mesh, order);

    let mut gpu = create_gpu_compute(device, queue);
    gpu.topology_dirty = true;
    let mut prof = StepProfiler::new(device, queue, 64);

    // Warmup.
    for f in 0..15 {
        let mut p = params;
        p.frame_seed = f;
        gpu_dispatch_frame(device, queue, &mut gpu, &mesh, &p, order, &coeffs, None);
    }
    device.poll(wgpu::PollType::Wait).unwrap();

    // Profiled steps: accumulate per-label totals (label order preserved).
    let m = 40u32;
    let mut totals: Vec<(&'static str, f64)> = Vec::new();
    for f in 0..m {
        let mut p = params;
        p.frame_seed = 5000 + f;
        gpu_dispatch_frame(device, queue, &mut gpu, &mesh, &p, order, &coeffs, Some(&mut prof));
        device.poll(wgpu::PollType::Wait).unwrap();
        for (label, ms) in prof.read_segments_ms(device) {
            match totals.iter_mut().find(|(l, _)| *l == label) {
                Some(entry) => entry.1 += ms,
                None => totals.push((label, ms)),
            }
        }
    }

    let step_total: f64 = totals.iter().map(|(_, ms)| ms).sum::<f64>() / m as f64;
    println!(
        "\n── per-pass breakdown: nu={} ({} verts) — {:.2} ms/step (timestamp sum) ──",
        ico_nu, n_verts, step_total,
    );
    let mut sorted: Vec<(&str, f64)> = totals.iter().map(|(l, ms)| (*l, ms / m as f64)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (label, ms) in sorted {
        println!("   {:<24} {:>7.3} ms  ({:>5.1}%)", label, ms, ms / step_total * 100.0);
    }
}

pub fn run() {
    println!("=== GROWTH headless GPU step benchmark ===");
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).expect("no GPU adapter");
    let info = adapter.get_info();
    println!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

    // CommandEncoder::write_timestamp (used by StepProfiler to mark between
    // passes) needs TIMESTAMP_QUERY_INSIDE_ENCODERS in addition to the base
    // TIMESTAMP_QUERY feature.
    let ts_feats = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    let has_ts = adapter.features().contains(ts_feats);
    let features = if has_ts { ts_feats } else { wgpu::Features::empty() };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("bench_device"),
        required_features: features,
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    })).expect("request_device failed");

    println!("vsync budget @60fps = {:.2} ms/frame\n", VSYNC_BUDGET_MS);
    for &nu in &[16usize, 24, 32, 40, 48, 56] {
        bench_size(&device, &queue, nu);
    }
    println!("\nInterpretation: 'substeps@60fps' = how many sim steps fit in one");
    println!("displayed frame. >1 means the sim is vsync-bound and decoupling wins.");

    if has_ts {
        for &nu in &[32usize, 56] {
            profile_size(&device, &queue, nu);
        }
    } else {
        println!("\n(TIMESTAMP_QUERY unsupported on this adapter — skipping per-pass breakdown)");
    }
}
