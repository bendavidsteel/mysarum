use nannou::prelude::*;

use crate::gpu::GpuCompute;

const RENDER_WGSL: &str = include_str!("shaders/render.wgsl");

pub(crate) struct RenderState {
    pub(crate) line_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) node_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) core_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) bind_group: Option<wgpu::BindGroup>,
    pub(crate) quad_vertex_buf: Option<wgpu::Buffer>,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            line_pipeline: None,
            node_pipeline: None,
            core_pipeline: None,
            bind_group: None,
            quad_vertex_buf: None,
        }
    }
}

pub(crate) fn init_render_state(
    device: &wgpu::Device,
    gpu: &GpuCompute,
    rs: &mut RenderState,
    texture_format: wgpu::TextureFormat,
    msaa_samples: u32,
) {
    let vs = wgpu::ShaderStages::VERTEX;
    let fs = wgpu::ShaderStages::VERTEX_FRAGMENT;

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: vs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: vs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: vs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3, visibility: fs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gpu.node_pos_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gpu.node_state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gpu.connection_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gpu.render_uniform_buf.as_entire_binding() },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render"),
        source: wgpu::ShaderSource::Wgsl(RENDER_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pl"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    // Additive blend for node glow
    let additive_blend = Some(wgpu::BlendState {
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
    });

    let multisample = wgpu::MultisampleState {
        count: msaa_samples, mask: !0, alpha_to_coverage_enabled: false,
    };

    // Line pipeline
    let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("line_render"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_line"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_line"),
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample,
        multiview: None,
        cache: None,
    });

    // Node pipeline (instanced quads)
    let node_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("node_render"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_node"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_node"),
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: additive_blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample,
        multiview: None,
        cache: None,
    });

    // Core pipeline (opaque cores, standard alpha blend, drawn on top)
    let alpha_blend = Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    });

    let core_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("core_render"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_core"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_core"),
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: alpha_blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample,
        multiview: None,
        cache: None,
    });

    // Quad vertex buffer (6 vertices: 2 triangles)
    #[rustfmt::skip]
    let quad_data: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 0.0],
        [ 1.0, -1.0, 1.0, 0.0],
        [ 1.0,  1.0, 1.0, 1.0],
        [-1.0, -1.0, 0.0, 0.0],
        [ 1.0,  1.0, 1.0, 1.0],
        [-1.0,  1.0, 0.0, 1.0],
    ];
    let quad_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("quad_vertices"),
        contents: bytemuck::cast_slice(&quad_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    rs.line_pipeline = Some(line_pipeline);
    rs.node_pipeline = Some(node_pipeline);
    rs.core_pipeline = Some(core_pipeline);
    rs.bind_group = Some(bind_group);
    rs.quad_vertex_buf = Some(quad_vertex_buf);
}

pub(crate) fn render(_render_app: &RenderApp, model: &super::Model, mut frame: Frame) {
    frame.clear(Color::BLACK);

    let Some(ref gpu) = model.gpu else { return; };
    if gpu.num_nodes == 0 { return; }

    let mut rs = model.render_state.lock().unwrap();

    if rs.node_pipeline.is_none() {
        let device = frame.device();
        let texture_format = frame.texture_format();
        let msaa_samples = frame.texture_msaa_samples();
        init_render_state(device, gpu, &mut rs, texture_format, msaa_samples);
    }

    let line_pipeline = rs.line_pipeline.as_ref().unwrap();
    let node_pipeline = rs.node_pipeline.as_ref().unwrap();
    let core_pipeline = rs.core_pipeline.as_ref().unwrap();
    let bind_group = rs.bind_group.as_ref().unwrap();
    let quad_buf = rs.quad_vertex_buf.as_ref().unwrap();

    let texture_view = frame.texture_view();
    let mut encoder = frame.command_encoder();

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gra_render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Draw connections (lines)
        if gpu.num_connections > 0 {
            pass.set_pipeline(line_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..gpu.num_connections * 2, 0..1);
        }

        // Draw bloom (additive blending)
        pass.set_pipeline(node_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, quad_buf.slice(..));
        pass.draw(0..6, 0..gpu.num_nodes);

        // Draw opaque cores on top (standard alpha blending)
        pass.set_pipeline(core_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, quad_buf.slice(..));
        pass.draw(0..6, 0..gpu.num_nodes);
    }
}
