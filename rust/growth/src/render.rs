use nannou::prelude::*;

use crate::gpu::GpuCompute;

const MESH_RENDER_WGSL: &str = include_str!("shaders/mesh_render.wgsl");
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub(crate) struct RenderState {
    pub(crate) pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(crate) bind_group: Option<wgpu::BindGroup>,
    pub(crate) depth_view: Option<wgpu::TextureViewHandle>,
    pub(crate) depth_size: [u32; 2],
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            pipeline: None, bind_group_layout: None, bind_group: None,
            depth_view: None, depth_size: [0, 0],
        }
    }
}

pub(crate) fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32, msaa_samples: u32) -> wgpu::TextureViewHandle {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

pub(crate) fn init_render_state(
    device: &wgpu::Device,
    gpu: &GpuCompute,
    rs: &mut RenderState,
    texture_format: wgpu::TextureFormat,
    msaa_samples: u32,
    texture_size: [u32; 2],
) {
    let vs = wgpu::ShaderStages::VERTEX;
    let fs = wgpu::ShaderStages::VERTEX_FRAGMENT;

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: vs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: vs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: fs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gpu.vertex_pos_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gpu.vertex_state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gpu.render_index_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gpu.render_uniform_buf.as_entire_binding() },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh_render"),
        source: wgpu::ShaderSource::Wgsl(MESH_RENDER_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pl"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mesh_render"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None, // double-sided, use front_facing in shader
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    let depth_view = create_depth_texture(device, texture_size[0], texture_size[1], msaa_samples);

    rs.bind_group_layout = Some(layout);
    rs.bind_group = Some(bind_group);
    rs.pipeline = Some(pipeline);
    rs.depth_view = Some(depth_view);
    rs.depth_size = texture_size;
}

pub(crate) fn render(_render_app: &RenderApp, model: &super::Model, mut frame: Frame) {
    frame.clear(Color::BLACK);

    let Some(ref gpu) = model.gpu else { return; };
    if gpu.num_render_tris == 0 { return; }

    let mut rs = model.render_state.lock().unwrap();
    let texture_size = frame.texture_size();
    let msaa_samples = frame.texture_msaa_samples();

    if rs.pipeline.is_none() {
        let device = frame.device();
        let texture_format = frame.texture_format();
        init_render_state(device, gpu, &mut rs, texture_format, msaa_samples, texture_size);
    }

    // Recreate depth texture on resize
    if rs.depth_size != texture_size {
        let device = frame.device();
        rs.depth_view = Some(create_depth_texture(device, texture_size[0], texture_size[1], msaa_samples));
        rs.depth_size = texture_size;
    }

    let pipeline = rs.pipeline.as_ref().unwrap();
    let bind_group = rs.bind_group.as_ref().unwrap();
    let depth_view = rs.depth_view.as_ref().unwrap();
    let num_verts = gpu.num_render_tris * 3;

    let texture_view = frame.texture_view();
    let mut encoder = frame.command_encoder();

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..num_verts, 0..1);
    }
}
