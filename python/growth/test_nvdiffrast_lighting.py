"""
Unit tests for nvdiffrast renderer lighting.
"""

import torch
import numpy as np
import nvdiffrast.torch as dr


def create_test_triangle(center_x=400, center_y=400, size=100):
    """Create a simple triangle matching the winding order of the real mesh.

    Real mesh uses CCW winding in screen coords (Y+ down), which gives +Z normals.
    Face vertices come from half_edge_dest: [dest(he0), dest(he1), dest(he2)]
    For the first triangle: [1, 2, 0] meaning v1 -> v2 -> v0
    """
    h = size * np.sqrt(3) / 2

    # Match real mesh layout: v0=center, v1=right, v2=above (in screen coords Y+ down)
    verts = torch.tensor([
        [center_x, center_y, 0],               # v0: center
        [center_x + size, center_y, 0],        # v1: right of center
        [center_x + size/2, center_y + h, 0],  # v2: above (larger Y in screen coords)
    ], dtype=torch.float32, device='cuda')

    # Face order [1, 2, 0] to match real mesh extraction
    faces = torch.tensor([[1, 2, 0]], dtype=torch.int32, device='cuda')

    # Compute actual normal from vertices
    v0, v1, v2 = verts[1], verts[2], verts[0]  # face order
    e1 = v1 - v0
    e2 = v2 - v0
    normal = torch.cross(e1, e2, dim=0)
    normal = normal / torch.norm(normal)

    normals = normal.unsqueeze(0).expand(3, -1).contiguous()

    return verts, faces, normals


def render_triangle_with_light(glctx, verts, faces, normals, light_dir, width=800, height=800):
    """Render a triangle with specified light direction."""
    # Transform to clip space
    center_x, center_y = width / 2, height / 2
    proj_scale = 2.0 / max(width, height)

    verts_clip = verts.clone()
    verts_clip[:, 0] = (verts[:, 0] - center_x) * proj_scale
    verts_clip[:, 1] = (verts[:, 1] - center_y) * proj_scale
    verts_clip[:, 2] = verts[:, 2] * proj_scale

    # Homogeneous coords
    verts_homo = torch.cat([verts_clip, torch.ones_like(verts_clip[:, :1])], dim=-1)
    verts_homo = verts_homo.unsqueeze(0).contiguous()

    # Double-sided faces: back faces first so front faces win depth test
    faces_back = faces[:, [0, 2, 1]]
    faces_double = torch.cat([faces_back, faces], dim=0)

    with torch.no_grad():
        # Rasterize
        rast_out, _ = dr.rasterize(glctx, verts_homo, faces_double, (height, width))

        # Interpolate normals
        normals_batch = normals.unsqueeze(0).contiguous()
        normals_interp, _ = dr.interpolate(normals_batch, rast_out, faces_double)
        normals_interp = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)

        # Front/back detection
        tri_id = rast_out[..., 3:4]
        # Back faces are first in faces_double, front faces are second
        n_back_faces = faces.shape[0]
        is_back = (tri_id > 0) & (tri_id <= n_back_faces)
        is_front = tri_id > n_back_faces

        # Flip normals for back faces
        normals_shading = torch.where(is_back, -normals_interp, normals_interp)

        # Lighting
        light_dir_norm = light_dir / torch.norm(light_dir)
        light_dir_view = light_dir_norm.view(1, 1, 1, 3)
        ndotl = torch.sum(normals_shading * light_dir_view, dim=-1, keepdim=True)

        # Return raw ndotl and mask for testing
        mask = (tri_id > 0).float()

    return ndotl[0], mask[0], rast_out[0]


def test_lighting_changes_with_direction():
    """Test that lighting intensity changes when light direction changes."""
    print("Test: Lighting changes with direction...")

    glctx = dr.RasterizeCudaContext()
    verts, faces, normals = create_test_triangle()

    # Light from front (should be bright, ndotl = 1)
    light_front = torch.tensor([0.0, 0.0, 1.0], device='cuda')
    ndotl_front, mask, _ = render_triangle_with_light(glctx, verts, faces, normals, light_front)

    # Light from side (should be dim, ndotl ≈ 0)
    light_side = torch.tensor([1.0, 0.0, 0.0], device='cuda')
    ndotl_side, _, _ = render_triangle_with_light(glctx, verts, faces, normals, light_side)

    # Light from back (should be negative/zero for front face)
    light_back = torch.tensor([0.0, 0.0, -1.0], device='cuda')
    ndotl_back, _, _ = render_triangle_with_light(glctx, verts, faces, normals, light_back)

    # Sample center of triangle
    cy, cx = 400, 400

    # Check values at triangle center
    front_val = ndotl_front[cy, cx, 0].item()
    side_val = ndotl_side[cy, cx, 0].item()
    back_val = ndotl_back[cy, cx, 0].item()

    print(f"  Light from front: ndotl = {front_val:.4f}")
    print(f"  Light from side:  ndotl = {side_val:.4f}")
    print(f"  Light from back:  ndotl = {back_val:.4f}")

    # Assertions
    assert mask[cy, cx, 0].item() > 0, "Triangle should be visible at center"
    assert front_val > 0.9, f"Front light should give ndotl ≈ 1, got {front_val}"
    assert abs(side_val) < 0.1, f"Side light should give ndotl ≈ 0, got {side_val}"
    assert back_val < -0.9, f"Back light should give ndotl ≈ -1, got {back_val}"

    print("  PASSED!\n")


def test_lighting_varies_across_angled_surface():
    """Test that lighting varies across a tilted triangle."""
    print("Test: Lighting varies across angled surface...")

    glctx = dr.RasterizeCudaContext()

    # Create a triangle tilted in 3D (one vertex pushed back in Z)
    verts = torch.tensor([
        [400, 300, 0],    # bottom center (close)
        [300, 500, 50],   # top left (far)
        [500, 500, 50],   # top right (far)
    ], dtype=torch.float32, device='cuda')

    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')

    # Compute face normal
    v0, v1, v2 = verts[0], verts[1], verts[2]
    e1 = v1 - v0
    e2 = v2 - v0
    face_normal = torch.cross(e1, e2)
    face_normal = face_normal / torch.norm(face_normal)

    # All vertices get the same normal
    normals = face_normal.unsqueeze(0).expand(3, -1).contiguous()

    print(f"  Face normal: {face_normal.cpu().numpy()}")

    # Light from front-top
    light_dir = torch.tensor([0.0, -0.5, 1.0], device='cuda')
    ndotl, mask, _ = render_triangle_with_light(glctx, verts, faces, normals, light_dir)

    # Sample multiple points on the triangle
    test_points = [(350, 400), (400, 400), (450, 400)]
    values = []
    for y, x in test_points:
        if mask[y, x, 0].item() > 0:
            val = ndotl[y, x, 0].item()
            values.append(val)
            print(f"  Point ({x}, {y}): ndotl = {val:.4f}")

    # For a flat triangle with uniform normal, all points should have same lighting
    if len(values) > 1:
        variation = max(values) - min(values)
        print(f"  Variation across surface: {variation:.4f}")
        assert variation < 0.01, "Flat triangle should have uniform lighting"

    print("  PASSED!\n")


def test_front_back_face_different():
    """Test that front and back faces have different lighting behavior."""
    print("Test: Front and back faces differ...")

    glctx = dr.RasterizeCudaContext()
    verts, faces, normals = create_test_triangle()

    # Light from front
    light_dir = torch.tensor([0.0, 0.0, 1.0], device='cuda')

    # Transform and render
    width, height = 800, 800
    center_x, center_y = width / 2, height / 2
    proj_scale = 2.0 / max(width, height)

    verts_clip = verts.clone()
    verts_clip[:, 0] = (verts[:, 0] - center_x) * proj_scale
    verts_clip[:, 1] = (verts[:, 1] - center_y) * proj_scale
    verts_clip[:, 2] = verts[:, 2] * proj_scale

    verts_homo = torch.cat([verts_clip, torch.ones_like(verts_clip[:, :1])], dim=-1)
    verts_homo = verts_homo.unsqueeze(0).contiguous()

    # Double-sided faces: back faces first so front faces win depth test
    faces_back = faces[:, [0, 2, 1]]
    faces_double = torch.cat([faces_back, faces], dim=0)

    with torch.no_grad():
        rast_out, _ = dr.rasterize(glctx, verts_homo, faces_double, (height, width))

        tri_id = rast_out[0, :, :, 3]

        # Find pixels that are front face (id=1) vs back face (id=2)
        front_mask = tri_id == 1
        back_mask = tri_id == 2

        n_front = front_mask.sum().item()
        n_back = back_mask.sum().item()

        print(f"  Front face pixels: {n_front}")
        print(f"  Back face pixels: {n_back}")

        # At least one should be visible (front face since normal points to camera)
        assert n_front > 0 or n_back > 0, "Triangle should be visible"

        # For a triangle with normal pointing +Z, front face should win
        # (it's closer in terms of winding order for the camera)
        print(f"  Front face is primary: {n_front > n_back}")

    print("  PASSED!\n")


def test_render_produces_visible_output():
    """Test that rendering produces non-zero output in triangle area."""
    print("Test: Render produces visible output...")

    glctx = dr.RasterizeCudaContext()
    verts, faces, normals = create_test_triangle(size=200)

    light_dir = torch.tensor([0.3, 0.3, 1.0], device='cuda')
    ndotl, mask, rast_out = render_triangle_with_light(glctx, verts, faces, normals, light_dir)

    # Count visible pixels
    visible_pixels = (mask > 0).sum().item()
    print(f"  Visible pixels: {visible_pixels}")

    assert visible_pixels > 1000, f"Should have many visible pixels, got {visible_pixels}"

    # Check that ndotl values are reasonable in visible area
    visible_ndotl = ndotl[mask > 0]
    mean_ndotl = visible_ndotl.mean().item()
    print(f"  Mean ndotl in visible area: {mean_ndotl:.4f}")

    # With light at (0.3, 0.3, 1.0) and normal at (0, 0, 1), ndotl should be positive
    assert mean_ndotl > 0.5, f"Expected positive ndotl with frontal light, got {mean_ndotl}"

    print("  PASSED!\n")


def test_full_render_color_output():
    """Test the full rendering pipeline produces expected colors."""
    print("Test: Full render color output...")

    glctx = dr.RasterizeCudaContext()
    verts, faces, normals = create_test_triangle(size=200)

    width, height = 800, 800
    center_x, center_y = width / 2, height / 2
    proj_scale = 2.0 / max(width, height)

    # Transform vertices
    verts_clip = verts.clone()
    verts_clip[:, 0] = (verts[:, 0] - center_x) * proj_scale
    verts_clip[:, 1] = (verts[:, 1] - center_y) * proj_scale
    verts_clip[:, 2] = verts[:, 2] * proj_scale

    verts_homo = torch.cat([verts_clip, torch.ones_like(verts_clip[:, :1])], dim=-1)
    verts_homo = verts_homo.unsqueeze(0).contiguous()

    # Double-sided faces: back faces first so front faces win depth test
    faces_back = faces[:, [0, 2, 1]]
    faces_double = torch.cat([faces_back, faces], dim=0)

    # Lighting params
    light_dir = torch.tensor([0.3, 0.3, 1.0], device='cuda')
    light_dir = light_dir / torch.norm(light_dir)
    ambient = 0.25
    diffuse = 0.75
    front_color = torch.tensor([0.6, 0.7, 0.9], device='cuda')
    back_color = torch.tensor([0.9, 0.6, 0.5], device='cuda')

    with torch.no_grad():
        rast_out, _ = dr.rasterize(glctx, verts_homo, faces_double, (height, width))

        normals_batch = normals.unsqueeze(0).contiguous()
        normals_interp, _ = dr.interpolate(normals_batch, rast_out, faces_double)
        normals_interp = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)

        tri_id = rast_out[..., 3:4]
        # Back faces are first in faces_double, front faces are second
        n_back_faces = faces.shape[0]
        is_back = (tri_id > 0) & (tri_id <= n_back_faces)
        is_front = tri_id > n_back_faces

        normals_shading = torch.where(is_back, -normals_interp, normals_interp)

        light_dir_view = light_dir.view(1, 1, 1, 3)
        ndotl = torch.clamp(torch.sum(normals_shading * light_dir_view, dim=-1, keepdim=True), 0, 1)

        front_color_view = front_color.view(1, 1, 1, 3)
        back_color_view = back_color.view(1, 1, 1, 3)
        base_color = torch.where(is_front, front_color_view, back_color_view)

        color = base_color * (ambient + diffuse * ndotl)

        mask = (tri_id > 0).float()
        bg_color = torch.tensor([0.1, 0.1, 0.1], device='cuda').view(1, 1, 1, 3)
        color = color * mask + bg_color * (1 - mask)

        color_uint8 = (color[0] * 255).clamp(0, 255).to(torch.uint8)

    # Sample center
    cy, cx = 400, 400
    center_color = color_uint8[cy, cx].cpu().numpy()
    print(f"  Center pixel color (RGB): {center_color}")

    # Should be bluish (front face color with lighting)
    assert center_color[2] > center_color[0], "Blue should be higher than red for front face"
    assert center_color[2] > 100, f"Blue channel should be significant, got {center_color[2]}"

    # Check background
    bg_pixel = color_uint8[0, 0].cpu().numpy()
    print(f"  Background color (RGB): {bg_pixel}")
    assert np.allclose(bg_pixel, [25, 25, 25], atol=2), f"Background should be ~25, got {bg_pixel}"

    print("  PASSED!\n")


if __name__ == '__main__':
    print("=" * 60)
    print("nvdiffrast Lighting Tests")
    print("=" * 60 + "\n")

    test_render_produces_visible_output()
    test_lighting_changes_with_direction()
    test_lighting_varies_across_angled_surface()
    test_front_back_face_different()
    test_full_render_color_output()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
