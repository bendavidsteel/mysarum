use bytemuck::{Pod, Zeroable};
use nannou::prelude::*;

use crate::camera::Camera;
use crate::gpu::{GpuCompute, VOL_W, VOL_H, VOL_D, VOL_MIN, VOL_MAX, NUM_BOIDS};

macro_rules! with_common {
    ($file:expr) => {
        concat!(include_str!("shaders/common.wgsl"), include_str!($file))
    };
}
const GROUND_WGSL:  &str = with_common!("shaders/ground.wgsl");
const BOX_WGSL:     &str = with_common!("shaders/box.wgsl");
const BOIDS_WGSL:   &str = with_common!("shaders/boids_render.wgsl");
const TREES_WGSL:   &str = with_common!("shaders/trees_render.wgsl");
const RAYCAST_WGSL: &str = with_common!("shaders/raycast.wgsl");

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// ── Uniforms (mirrors WGSL `Uniforms`) ────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj:     [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub cam_pos:       [f32; 4],
    pub world_dim:     [f32; 4],
    pub vol_min:       [f32; 4],
    pub vol_max:       [f32; 4],
    pub vol_res:       [f32; 4],
    pub misc:          [f32; 4],
    pub activity:      [f32; 4],
    pub render_params: [f32; 4],
}

pub struct RenderState {
    uniform_buf: Option<wgpu::Buffer>,
    ground_pipeline:  Option<wgpu::RenderPipeline>,
    box_pipeline:     Option<wgpu::RenderPipeline>,
    boids_pipeline:   Option<wgpu::RenderPipeline>,
    trees_pipeline:   Option<wgpu::RenderPipeline>,
    raycast_pipeline: Option<wgpu::RenderPipeline>,
    ubo_bg:     Option<wgpu::BindGroup>,
    boid_bgs:   Option<[wgpu::BindGroup; 2]>,
    trees_bg:   Option<wgpu::BindGroup>,
    raycast_bg: Option<wgpu::BindGroup>,
    depth_view: Option<wgpu::TextureViewHandle>,
    depth_size: [u32; 2],
    depth_samples: u32,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            uniform_buf: None,
            ground_pipeline: None, box_pipeline: None, boids_pipeline: None,
            trees_pipeline: None, raycast_pipeline: None,
            ubo_bg: None, boid_bgs: None, trees_bg: None, raycast_bg: None,
            depth_view: None, depth_size: [0, 0], depth_samples: 1,
        }
    }
}

fn screen_blend() -> Option<wgpu::BlendState> {
    Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrc,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrc,
            operation: wgpu::BlendOperation::Add,
        },
    })
}
fn additive_blend() -> Option<wgpu::BlendState> {
    Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn make_pipeline(
    device: &wgpu::Device, layout: &wgpu::PipelineLayout, src: &str, label: &str,
    fmt: wgpu::TextureFormat, msaa: u32, topology: wgpu::PrimitiveTopology,
    blend: Option<wgpu::BlendState>, depth_write: bool, depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label), layout: Some(layout),
        vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &module, entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format: fmt, blend, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT, depth_write_enabled: depth_write, depth_compare,
            stencil: Default::default(), bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: msaa, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None, cache: None,
    })
}

fn init(device: &wgpu::Device, gpu: &GpuCompute, rs: &mut RenderState, fmt: wgpu::TextureFormat, msaa: u32) {
    let vsf = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let vs = wgpu::ShaderStages::VERTEX;
    let fs = wgpu::ShaderStages::FRAGMENT;

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // layouts
    let ubo_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ubo_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0, visibility: vsf,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
    });
    let storage_ubo_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("storage_ubo_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: vs,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: vsf,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    let raycast_bgl = wgpu::BindGroupLayoutBuilder::new()
        .texture(fs, false, wgpu::TextureViewDimension::D3, wgpu::TextureSampleType::Float { filterable: true })
        .sampler(fs, true)
        .uniform_buffer(fs, false)
        .build(device);

    // bind groups
    let ubo_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ubo_bg"), layout: &ubo_bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
    });
    let boid_bgs = [0usize, 1usize].map(|i| device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("boid_render_bg"), layout: &storage_ubo_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gpu.boid_bufs[i].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: uniform_buf.as_entire_binding() },
        ],
    }));
    let trees_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trees_render_bg"), layout: &storage_ubo_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gpu.segment_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: uniform_buf.as_entire_binding() },
        ],
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("trail_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest, ..Default::default()
    });
    let raycast_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raycast_bg"), layout: &raycast_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&gpu.trail_views[0]) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: uniform_buf.as_entire_binding() },
        ],
    });

    // pipeline layouts
    let ubo_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ubo_pl"), bind_group_layouts: &[&ubo_bgl], push_constant_ranges: &[] });
    let storage_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("storage_pl"), bind_group_layouts: &[&storage_ubo_bgl], push_constant_ranges: &[] });
    let raycast_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("raycast_pl"), bind_group_layouts: &[&raycast_bgl], push_constant_ranges: &[] });

    use wgpu::PrimitiveTopology::{TriangleList, LineList};
    use wgpu::CompareFunction::{Less, Always};

    rs.ground_pipeline  = Some(make_pipeline(device, &ubo_pl, GROUND_WGSL, "ground", fmt, msaa, TriangleList, None, true, Less));
    rs.box_pipeline     = Some(make_pipeline(device, &ubo_pl, BOX_WGSL, "box", fmt, msaa, LineList, None, true, Less));
    rs.trees_pipeline   = Some(make_pipeline(device, &storage_pl, TREES_WGSL, "trees", fmt, msaa, TriangleList, None, true, Less));
    rs.boids_pipeline   = Some(make_pipeline(device, &storage_pl, BOIDS_WGSL, "boids", fmt, msaa, TriangleList, additive_blend(), false, Less));
    rs.raycast_pipeline = Some(make_pipeline(device, &raycast_pl, RAYCAST_WGSL, "raycast", fmt, msaa, TriangleList, screen_blend(), false, Always));

    rs.uniform_buf = Some(uniform_buf);
    rs.ubo_bg = Some(ubo_bg);
    rs.boid_bgs = Some(boid_bgs);
    rs.trees_bg = Some(trees_bg);
    rs.raycast_bg = Some(raycast_bg);
}

fn ensure_depth(device: &wgpu::Device, rs: &mut RenderState, size: [u32; 2], samples: u32) {
    if rs.depth_view.is_some() && rs.depth_size == size && rs.depth_samples == samples { return; }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: samples, dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    rs.depth_view = Some(tex.create_view(&Default::default()));
    rs.depth_size = size;
    rs.depth_samples = samples;
}

pub fn render(_render_app: &RenderApp, model: &crate::Model, mut frame: Frame) {
    frame.clear(Color::BLACK);
    let Some(ref gpu) = model.gpu else { return; };

    let mut rs = model.render_state.lock().unwrap();
    let device = frame.device();
    if rs.uniform_buf.is_none() {
        init(device, gpu, &mut rs, frame.texture_format(), frame.texture_msaa_samples());
    }
    let size = frame.texture_size();
    ensure_depth(device, &mut rs, size, frame.texture_msaa_samples());

    // ── Build uniforms ────────────────────────────────────────────────────
    let aspect = size[0] as f32 / size[1].max(1) as f32;
    let cam = Camera::orbit(model.time);
    let vp = cam.view_proj(aspect);
    let inv = vp.inverse();
    let u = Uniforms {
        view_proj: vp.to_cols_array_2d(),
        inv_view_proj: inv.to_cols_array_2d(),
        cam_pos: [cam.eye.x, cam.eye.y, cam.eye.z, 0.0],
        world_dim: [crate::camera::WORLD, crate::camera::WORLD, crate::camera::WORLD, 0.0],
        vol_min: [VOL_MIN[0], VOL_MIN[1], VOL_MIN[2], 0.0],
        vol_max: [VOL_MAX[0], VOL_MAX[1], VOL_MAX[2], 0.0],
        vol_res: [VOL_W as f32, VOL_H as f32, VOL_D as f32, 0.0],
        misc: [model.time, model.wind_strength, model.wind_dir.x, model.wind_dir.y],
        activity: [model.boid_activity, model.physarum_activity, model.tree_activity, aspect],
        render_params: [3.0, 0.85, 0.008, 0.006],
    };
    frame.queue().write_buffer(rs.uniform_buf.as_ref().unwrap(), 0, bytemuck::bytes_of(&u));

    let depth_view = rs.depth_view.clone().unwrap();
    let texture_view = frame.texture_view();
    let mut enc = frame.command_encoder();
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("grove_render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture_view, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None, occlusion_query_set: None,
        });

        // 1. ground
        pass.set_pipeline(rs.ground_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, rs.ubo_bg.as_ref().unwrap(), &[]);
        pass.draw(0..(96u32 * 96 * 6), 0..1);

        // 2. trees
        if gpu.segment_count > 0 {
            pass.set_pipeline(rs.trees_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, rs.trees_bg.as_ref().unwrap(), &[]);
            pass.draw(0..6, 0..gpu.segment_count);
        }

        // 3. boids (additive)
        pass.set_pipeline(rs.boids_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, &rs.boid_bgs.as_ref().unwrap()[gpu.boid_cur], &[]);
        pass.draw(0..6, 0..NUM_BOIDS);

        // 4. physarum volume (screen blend)
        pass.set_pipeline(rs.raycast_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, rs.raycast_bg.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);

        // 5. wireframe box
        pass.set_pipeline(rs.box_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, rs.ubo_bg.as_ref().unwrap(), &[]);
        pass.draw(0..24, 0..1);
    }
}
