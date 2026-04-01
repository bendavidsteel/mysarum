use nannou::prelude::*;

use crate::gpu::GpuCompute;

const RENDER_PARTICLE_WGSL: &str = include_str!("shaders/render_particle.wgsl");
const RENDER_GRA_WGSL: &str = include_str!("shaders/render_gra.wgsl");

pub(crate) struct RenderState {
    // Particle rendering
    pub(crate) particle_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) particle_bgs: Option<[wgpu::BindGroup; 2]>,

    // GRA rendering
    pub(crate) gra_line_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) gra_node_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) gra_core_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) gra_bg: Option<wgpu::BindGroup>,

    pub(crate) quad_vertex_buf: Option<wgpu::Buffer>,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            particle_pipeline: None,
            particle_bgs: None,
            gra_line_pipeline: None,
            gra_node_pipeline: None,
            gra_core_pipeline: None,
            gra_bg: None,
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

    let multisample = wgpu::MultisampleState {
        count: msaa_samples, mask: !0, alpha_to_coverage_enabled: false,
    };

    // ── Shared quad vertex buffer ────────────────────────────────────────────
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

    let quad_layout = wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
        ],
    };

    // ── Additive blend for particles and GRA glow ────────────────────────────
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

    // ── Particle rendering ───────────────────────────────────────────────────
    let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render_particle"),
        source: wgpu::ShaderSource::Wgsl(RENDER_PARTICLE_WGSL.into()),
    });

    // Bind group: particles(R) + uniforms(U)
    let particle_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("particle_render_bgl"),
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
                binding: 1, visibility: fs,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let particle_bgs = [0, 1].map(|i| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(if i == 0 { "particle_render_bg_0" } else { "particle_render_bg_1" }),
            layout: &particle_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gpu.particle_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.render_uniform_buf.as_entire_binding() },
            ],
        })
    });

    let particle_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle_render_pl"),
        bind_group_layouts: &[&particle_bgl],
        push_constant_ranges: &[],
    });

    let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("particle_render"),
        layout: Some(&particle_pl),
        vertex: wgpu::VertexState {
            module: &particle_shader,
            entry_point: Some("vs_main"),
            buffers: &[quad_layout.clone()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &particle_shader,
            entry_point: Some("fs_main"),
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

    // ── GRA rendering ────────────────────────────────────────────────────────
    let gra_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render_gra"),
        source: wgpu::ShaderSource::Wgsl(RENDER_GRA_WGSL.into()),
    });

    // Bind group: gra_pos(R) + gra_state(R) + connections(R) + uniforms(U)
    let gra_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gra_render_bgl"),
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

    let gra_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gra_render_bg"),
        layout: &gra_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gpu.gra_pos_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gpu.gra_state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gpu.connection_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gpu.render_uniform_buf.as_entire_binding() },
        ],
    });

    let gra_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gra_render_pl"),
        bind_group_layouts: &[&gra_bgl],
        push_constant_ranges: &[],
    });

    // Edge pipeline (quad strips with Gaussian falloff)
    let gra_line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gra_line_render"),
        layout: Some(&gra_pl),
        vertex: wgpu::VertexState {
            module: &gra_shader,
            entry_point: Some("vs_line"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &gra_shader,
            entry_point: Some("fs_line"),
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

    // Node glow pipeline (alpha blend so nodes render on top of edges)
    let gra_node_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gra_node_render"),
        layout: Some(&gra_pl),
        vertex: wgpu::VertexState {
            module: &gra_shader,
            entry_point: Some("vs_node"),
            buffers: &[quad_layout.clone()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &gra_shader,
            entry_point: Some("fs_node"),
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

    // Core pipeline
    let gra_core_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gra_core_render"),
        layout: Some(&gra_pl),
        vertex: wgpu::VertexState {
            module: &gra_shader,
            entry_point: Some("vs_core"),
            buffers: &[quad_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &gra_shader,
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

    rs.particle_pipeline = Some(particle_pipeline);
    rs.particle_bgs = Some(particle_bgs);
    rs.gra_line_pipeline = Some(gra_line_pipeline);
    rs.gra_node_pipeline = Some(gra_node_pipeline);
    rs.gra_core_pipeline = Some(gra_core_pipeline);
    rs.gra_bg = Some(gra_bg);
    rs.quad_vertex_buf = Some(quad_vertex_buf);
}

pub(crate) fn render(_render_app: &RenderApp, model: &super::Model, mut frame: Frame) {
    frame.clear(Color::BLACK);

    let Some(ref gpu) = model.gpu else { return; };

    let mut rs = model.render_state.lock().unwrap();

    if rs.particle_pipeline.is_none() {
        let device = frame.device();
        let texture_format = frame.texture_format();
        let msaa_samples = frame.texture_msaa_samples();
        init_render_state(device, gpu, &mut rs, texture_format, msaa_samples);
    }

    let texture_view = frame.texture_view();
    let mut encoder = frame.command_encoder();

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pond_render"),
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

        let quad_buf = rs.quad_vertex_buf.as_ref().unwrap();

        // ── Draw GRA edges ───────────────────────────────────────────────
        if gpu.num_gra_connections > 0 {
            let gra_bg = rs.gra_bg.as_ref().unwrap();
            pass.set_pipeline(rs.gra_line_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, gra_bg, &[]);
            pass.draw(0..gpu.num_gra_connections * 12, 0..1);
        }

        // ── Draw GRA node glow ───────────────────────────────────────────
        if gpu.num_gra_nodes > 0 {
            let gra_bg = rs.gra_bg.as_ref().unwrap();
            pass.set_pipeline(rs.gra_node_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, gra_bg, &[]);
            pass.set_vertex_buffer(0, quad_buf.slice(..));
            pass.draw(0..6, 0..gpu.num_gra_nodes);

            // ── Draw GRA cores ───────────────────────────────────────────
            pass.set_pipeline(rs.gra_core_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, gra_bg, &[]);
            pass.set_vertex_buffer(0, quad_buf.slice(..));
            pass.draw(0..6, 0..gpu.num_gra_nodes);
        }

        // ── Draw particles ───────────────────────────────────────────────
        if gpu.num_particles > 0 {
            let particle_bg = &rs.particle_bgs.as_ref().unwrap()[gpu.particle_frame];
            pass.set_pipeline(rs.particle_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, particle_bg, &[]);
            pass.set_vertex_buffer(0, quad_buf.slice(..));
            pass.draw(0..6, 0..gpu.num_particles);
        }
    }
}
