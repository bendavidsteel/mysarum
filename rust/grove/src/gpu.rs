use bytemuck::{Pod, Zeroable};
use nannou::prelude::*;

use crate::camera::WORLD;

// ── Shader sources ──────────────────────────────────────────────────────────
macro_rules! with_common {
    ($file:expr) => {
        concat!(include_str!("shaders/common.wgsl"), include_str!($file))
    };
}
const BOIDS_WGSL:   &str = with_common!("shaders/boids.wgsl");
const AGENTS_WGSL:  &str = with_common!("shaders/agents.wgsl");
const DIFFUSE_WGSL: &str = with_common!("shaders/diffuse.wgsl");
const DECAY_WGSL:   &str = with_common!("shaders/decay.wgsl");

// ── Simulation sizes ──────────────────────────────────────────────────────────
pub const NUM_BOIDS:  u32 = 1024 * 8;
pub const NUM_AGENTS: u32 = 1024 * 8;
pub const VOL_W: u32 = 256;
pub const VOL_H: u32 = 32;
pub const VOL_D: u32 = 256;

const TRAIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

// Physarum volume world-space placement: a flat slab sitting on the ground.
pub const VOL_MIN: [f32; 3] = [0.0, 0.0, 0.0];
pub const VOL_MAX: [f32; 3] = [WORLD, WORLD * 0.1, WORLD];

// ── Particle / Agent / Segment as raw 16-float (or 12-float) blocks ───────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Particle {
    pub pos:   [f32; 4],
    pub vel:   [f32; 4],
    pub attr:  [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Agent {
    pub pos:   [f32; 4],
    pub vel:   [f32; 4],
    pub attr:  [f32; 4],
    pub state: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Segment {
    pub p0:    [f32; 4],
    pub p1:    [f32; 4],
    pub color: [f32; 4],
}

// ── ComputeParams (mirrors WGSL ComputeParams: 10 × vec4) ─────────────────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ComputeParams {
    pub vol_res:   [f32; 4],
    pub world_res: [f32; 4],
    pub timing:    [f32; 4],
    pub wind:      [f32; 4],
    pub boid_a:    [f32; 4],
    pub boid_b:    [f32; 4],
    pub boid_c:    [f32; 4],
    pub phys:      [f32; 4],
    pub phys2:     [f32; 4],
    pub blur_dir:  [f32; 4],
}

// ── Tiny deterministic RNG for CPU-side init ──────────────────────────────────
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32) / (u32::MAX as f32)
    }
    fn range(&mut self, a: f32, b: f32) -> f32 { a + (b - a) * self.next() }
}

// ── GPU state ─────────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct GpuCompute {
    // boids ping-pong
    pub boid_bufs: [wgpu::Buffer; 2],
    boid_bgs:      [wgpu::BindGroup; 2],
    boid_pipeline: wgpu::ComputePipeline,
    /// index of the buffer holding the latest boid state (for render)
    pub boid_cur:  usize,

    // physarum agents ping-pong (buffers kept alive for the bind groups below)
    #[allow(dead_code)]
    agent_bufs: [wgpu::Buffer; 2],
    agent_bgs:  [wgpu::BindGroup; 2],
    agent_pipeline: wgpu::ComputePipeline,
    agent_cur:  usize,

    // trail volume ping-pong. `trail_cur` is the texture holding the current
    // field after a step (what the renderer samples); it flips each frame.
    pub trail_views: [wgpu::TextureViewHandle; 2],
    pub trail_cur:   usize,
    diffuse_pipeline: wgpu::ComputePipeline,
    decay_pipeline:   wgpu::ComputePipeline,
    /// trail bind groups keyed by (read_idx): bg[r] reads tex[r], writes tex[1-r]
    trail_bgs_diffuse: [wgpu::BindGroup; 2],
    trail_textures: [wgpu::TextureHandle; 2],

    // shared compute params
    pub params_buf: wgpu::Buffer,
    params_bg:      wgpu::BindGroup,

    // tree segments (uploaded from CPU)
    pub segment_buf:   wgpu::Buffer,
    pub segment_count: u32,
    segment_capacity:  u32,

    flip: bool,
}

impl GpuCompute {
    pub fn new(device: &wgpu::Device) -> Self {
        let cs = wgpu::ShaderStages::COMPUTE;

        // ── CPU init ───────────────────────────────────────────────────────
        let mut rng = Rng(0x1234_5678);
        let mut boids = vec![Particle::zeroed(); NUM_BOIDS as usize];
        for p in boids.iter_mut() {
            p.pos = [rng.range(0.0, WORLD),
                     rng.range(0.25 * WORLD, WORLD),
                     rng.range(0.0, WORLD), 1.0];
            p.vel = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 1.0];
            p.attr = [0.1, rng.range(0.0, TAU), 0.0, 0.0];
            p.color = [1.0, 1.0, 1.0, 0.0];
        }

        let mut agents = vec![Agent::zeroed(); NUM_AGENTS as usize];
        let (vw, vh, vd) = (VOL_W as f32, VOL_H as f32, VOL_D as f32);
        for a in agents.iter_mut() {
            // EDGES spawn (species 0)
            let mut x; let mut z; let y = vh;
            match (rng.range(0.0, 4.0)) as i32 {
                0 => { x = rng.range(0.0, vw); z = 0.0; }
                1 => { x = vw; z = rng.range(0.0, vd); }
                2 => { x = rng.range(0.0, vw); z = vd; }
                _ => { x = 0.0; z = rng.range(0.0, vd); }
            }
            x = x.clamp(0.5, vw - 0.5);
            z = z.clamp(0.5, vd - 0.5);
            let mut v = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt().max(1e-4);
            v = [v[0]/len * 1.1, v[1]/len * 1.1, v[2]/len * 1.1];
            a.pos = [x, y - 0.5, z, 1.0];
            a.vel = [v[0], v[1], v[2], 1.0];
            a.attr = [0.0, 0.0, 0.0, 0.0];
        }

        let boid_bufs = [
            create_storage_init(device, "boid0", bytemuck::cast_slice(&boids)),
            create_storage_init(device, "boid1", bytemuck::cast_slice(&boids)),
        ];
        let agent_bufs = [
            create_storage_init(device, "agent0", bytemuck::cast_slice(&agents)),
            create_storage_init(device, "agent1", bytemuck::cast_slice(&agents)),
        ];

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compute_params"),
            size:  std::mem::size_of::<ComputeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // segment buffer — preallocated to the metamer cap so its render bind
        // group never goes stale (trees cap at MAX_METAMERS*0.5 metamers).
        let segment_capacity = 16384u32;
        let segment_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segments"),
            size:  (segment_capacity as u64) * std::mem::size_of::<Segment>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Trail volume textures ──────────────────────────────────────────
        let make_trail = |label: &str| device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: VOL_W, height: VOL_H, depth_or_array_layers: VOL_D },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: TRAIL_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_SRC
                 | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let trail_textures = [make_trail("trail0"), make_trail("trail1")];
        let trail_views = [
            trail_textures[0].create_view(&Default::default()),
            trail_textures[1].create_view(&Default::default()),
        ];

        // ── Bind group layouts ─────────────────────────────────────────────
        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("params_bgl"),
            entries: &[uniform_entry(0, cs)],
        });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("params_bg"),
            layout: &params_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() }],
        });

        // boids: 2 storage buffers
        let boid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("boid_bgl"),
            entries: &[storage_entry(0, cs, true), storage_entry(1, cs, false)],
        });
        let boid_bgs = [
            make_two_buf_bg(device, &boid_bgl, &boid_bufs[0], &boid_bufs[1], "boid_bg0"),
            make_two_buf_bg(device, &boid_bgl, &boid_bufs[1], &boid_bufs[0], "boid_bg1"),
        ];

        // agents: 2 storage buffers + trail_read(sampled) + trail_write(storage)
        let agent_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agent_bgl"),
            entries: &[
                storage_entry(0, cs, true),
                storage_entry(1, cs, false),
                tex3d_entry(2, cs),
                storage_tex_entry(3, cs),
            ],
        });
        // agent_bgs[f]: read agent[f] + trail[f], write agent[1-f] + trail[1-f].
        // The trail ping-pongs in lockstep with `flip`, so the deposit target is
        // always the freshly-seeded "other" texture (see step()).
        let agent_bgs = [
            make_agent_bg(device, &agent_bgl, &agent_bufs[0], &agent_bufs[1], &trail_views[0], &trail_views[1], "agent_bg0"),
            make_agent_bg(device, &agent_bgl, &agent_bufs[1], &agent_bufs[0], &trail_views[1], &trail_views[0], "agent_bg1"),
        ];

        // diffuse/decay: trail_read(sampled) + trail_write(storage)
        let trail_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trail_bgl"),
            entries: &[tex3d_entry(0, cs), storage_tex_entry(1, cs)],
        });
        let trail_bgs_diffuse = [
            make_trail_bg(device, &trail_bgl, &trail_views[0], &trail_views[1], "trail_bg_0to1"),
            make_trail_bg(device, &trail_bgl, &trail_views[1], &trail_views[0], "trail_bg_1to0"),
        ];

        // ── Pipelines ───────────────────────────────────────────────────────
        let boid_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("boid_pl"), bind_group_layouts: &[&boid_bgl, &params_bgl], push_constant_ranges: &[],
        });
        let agent_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agent_pl"), bind_group_layouts: &[&agent_bgl, &params_bgl], push_constant_ranges: &[],
        });
        let trail_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trail_pl"), bind_group_layouts: &[&trail_bgl, &params_bgl], push_constant_ranges: &[],
        });

        let boid_pipeline    = make_pipeline(device, &boid_pl,  BOIDS_WGSL,   "boids");
        let agent_pipeline   = make_pipeline(device, &agent_pl, AGENTS_WGSL,  "agents");
        let diffuse_pipeline = make_pipeline(device, &trail_pl, DIFFUSE_WGSL, "diffuse");
        let decay_pipeline   = make_pipeline(device, &trail_pl, DECAY_WGSL,   "decay");

        Self {
            boid_bufs, boid_bgs, boid_pipeline, boid_cur: 0,
            agent_bufs, agent_bgs, agent_pipeline, agent_cur: 0,
            trail_views, trail_cur: 0, diffuse_pipeline, decay_pipeline, trail_bgs_diffuse, trail_textures,
            params_buf, params_bg,
            segment_buf, segment_count: 0, segment_capacity,
            flip: false,
        }
    }

    /// Upload the latest tree segments (called only when the tree grew).
    pub fn upload_segments(&mut self, _device: &wgpu::Device, queue: &wgpu::Queue, segs: &[Segment]) {
        if segs.is_empty() { self.segment_count = 0; return; }
        let n = (segs.len() as u32).min(self.segment_capacity);
        queue.write_buffer(&self.segment_buf, 0, bytemuck::cast_slice(&segs[..n as usize]));
        self.segment_count = n;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, params: ComputeParams) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("grove_compute"),
        });

        let src = self.flip as usize;        // boid/agent read index
        let dst = 1 - src;

        // ── Boids ──────────────────────────────────────────────────────────
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("boids"), timestamp_writes: None });
            pass.set_pipeline(&self.boid_pipeline);
            pass.set_bind_group(0, &self.boid_bgs[src], &[]);
            pass.set_bind_group(1, &self.params_bg, &[]);
            pass.dispatch_workgroups((NUM_BOIDS + 255) / 256, 1, 1);
        }
        self.boid_cur = dst;

        // ── Physarum (ping-pong, current field starts in trail[src]) ───────────
        // The agent deposit is a sparse scatter into a write-only storage texture,
        // so its target must already hold the current field — hence one seed copy
        // (trail[src] → trail[dst]) is structurally required. The old second copy
        // (re-canonicalising to a fixed tex0) is gone: we just track trail_cur.
        copy_tex(&mut enc, &self.trail_textures[src], &self.trail_textures[dst]);

        {
            // agents: read trail[src], deposit into the seeded trail[dst]
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("agents"), timestamp_writes: None });
            pass.set_pipeline(&self.agent_pipeline);
            pass.set_bind_group(0, &self.agent_bgs[src], &[]);
            pass.set_bind_group(1, &self.params_bg, &[]);
            pass.dispatch_workgroups((NUM_AGENTS + 255) / 256, 1, 1);
        }
        self.agent_cur = dst;

        // diffuse (read dst → write src), decay (read src → write dst).
        // trail_bgs_diffuse[r] reads trail[r] and writes trail[1-r].
        let (gx, gy, gz) = ((VOL_W + 7) / 8, (VOL_H + 7) / 8, (VOL_D + 7) / 8);
        run_trail(&mut enc, &self.diffuse_pipeline, &self.trail_bgs_diffuse[dst], &self.params_bg, gx, gy, gz, "diffuse");
        run_trail(&mut enc, &self.decay_pipeline,   &self.trail_bgs_diffuse[src], &self.params_bg, gx, gy, gz, "decay");
        // field now lives in trail[dst]; it becomes next frame's trail[src].
        self.trail_cur = dst;

        queue.submit(Some(enc.finish()));
        self.flip = !self.flip;
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────
fn copy_tex(enc: &mut wgpu::CommandEncoder, src: &wgpu::TextureHandle, dst: &wgpu::TextureHandle) {
    enc.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo { texture: src, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyTextureInfo { texture: dst, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::Extent3d { width: VOL_W, height: VOL_H, depth_or_array_layers: VOL_D },
    );
}

#[allow(clippy::too_many_arguments)]
fn run_trail(enc: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup,
             params_bg: &wgpu::BindGroup, gx: u32, gy: u32, gz: u32, label: &str) {
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg, &[]);
    pass.set_bind_group(1, params_bg, &[]);
    pass.dispatch_workgroups(gx, gy, gz);
}

fn create_storage_init(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    })
}

fn uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
fn storage_entry(binding: u32, vis: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
fn tex3d_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}
fn storage_tex_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: TRAIL_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D3,
        },
        count: None,
    }
}

fn make_two_buf_bg(device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
                   a: &wgpu::Buffer, b: &wgpu::Buffer, label: &str) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label), layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn make_agent_bg(device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
                 a_in: &wgpu::Buffer, a_out: &wgpu::Buffer,
                 trail_r: &wgpu::TextureViewHandle, trail_w: &wgpu::TextureViewHandle, label: &str) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label), layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: a_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(trail_r) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(trail_w) },
        ],
    })
}

fn make_trail_bg(device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
                 trail_r: &wgpu::TextureViewHandle, trail_w: &wgpu::TextureViewHandle, label: &str) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label), layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(trail_r) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(trail_w) },
        ],
    })
}

fn make_pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout, src: &str, label: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label), layout: Some(layout), module: &module,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    })
}
