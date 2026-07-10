"""Printability metrics for grown floraform meshes (FDM, flat side down).

The mesh always starts as a hemisphere with its flat cap on the ground plane,
so the physical build direction is +z (the dome grows up, away from the bed).
These metrics estimate how support-hungry / slow a mesh is to print, for use as
a MAP-Elites fitness (lower = more printable).

All functions are pure NumPy and take the extracted, re-indexed mesh
(verts (V,3) float, faces (F,3) int referencing only valid verts) — the same
arrays `nca_shape_descriptors.mesh_descriptors` consumes.

Face-normal orientation is assumed outward-consistent (the mesh builders wind
faces outward and the growth/subdivision ops preserve winding), so a negative
z-component means a downward-facing surface.
"""

import numpy as np


def _face_geometry(verts, faces):
    """Return (face_normals_unit, face_areas, face_centroids)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    twice_area = np.linalg.norm(cross, axis=1)
    area = 0.5 * twice_area
    normals = cross / (twice_area[:, None] + 1e-12)
    centroids = (v0 + v1 + v2) / 3.0
    return normals, area, centroids


def print_metrics(verts, faces, *, overhang_angle_deg=45.0, layer_height=1.0,
                  build_up=(0.0, 0.0, 1.0), ground_z=None):
    """Raw printability metrics for one mesh.

    Returns a dict:
      overhang_fraction  area-weighted fraction of faces whose downward tilt
                         exceeds the self-support angle (build_up = +z).
      height             extent along the build axis (n layers = height/layer_height).
      layers             height / layer_height.
      support_volume     Σ over overhang faces of (horizontal-projected area ·
                         height above ground) — a support-material proxy.
      surface_area       total triangle area (shell/perimeter time proxy).
      footprint_area     xy bounding-box area (bed footprint).
    """
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    up = np.asarray(build_up, np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)

    normals, area, centroids = _face_geometry(verts, faces)
    total_area = float(area.sum()) + 1e-12

    # Overhang: face is an unsupported overhang when the angle between its
    # (downward-pointing) normal and straight-down exceeds the support angle,
    # i.e. n·up < -cos(overhang_angle). n·up == 0 is a vertical wall (fine).
    ndotu = normals @ up
    cos_crit = np.cos(np.radians(overhang_angle_deg))
    is_overhang = ndotu < -cos_crit
    overhang_fraction = float(area[is_overhang].sum() / total_area)

    proj = verts @ up
    height = float(proj.max() - proj.min())
    z0 = float(proj.min()) if ground_z is None else float(ground_z)

    # Support volume ≈ Σ (horizontal projection of overhang face) · (height of
    # the face above the bed). Horizontal projection area = area · |n·up|.
    face_h = np.maximum(centroids @ up - z0, 0.0)
    support_col = area * np.abs(ndotu) * face_h
    support_volume = float(support_col[is_overhang].sum())

    # Footprint: bbox area in the plane orthogonal to the build axis. For the
    # canonical +z build this is just the xy bounding box.
    ax = int(np.argmax(np.abs(up)))
    planar = [i for i in range(3) if i != ax]
    span = verts[:, planar].max(0) - verts[:, planar].min(0)
    footprint_area = float(span[0] * span[1])

    # Enclosed volume via the divergence theorem (Σ v0·(v1×v2)/6).
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    mesh_volume = abs(float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0))

    return {
        "overhang_fraction": overhang_fraction,
        "height": height,
        "layers": height / max(layer_height, 1e-9),
        "support_volume": support_volume,
        "surface_area": total_area,
        "footprint_area": footprint_area,
        "mesh_volume": mesh_volume,
        "compactness": total_area / max(mesh_volume, 1e-6) ** (2.0 / 3.0),
    }


# Reference scales that map each raw metric to ~O(1) before weighting, so the
# fitness weights are interpretable and comparable. Overridable via config.
DEFAULT_REFS = dict(
    overhang_fraction=1.0,   # already a fraction in [0, 1]
    layers=200.0,            # ~a few-hundred-layer print
    support_volume=1.0e6,    # world units^3 (spring_len ~ 30)
    surface_area=1.0e6,      # world units^2
)


def printability_penalty(metrics, weights, refs=None):
    """Weighted, reference-normalised penalty (lower = more printable).

    penalty = Σ_k weights[k] · metrics[k] / refs[k]   over keys present in both.
    `weights` picks which metrics count and how much; a metric with weight 0 or
    absent from `weights` is ignored. Returns (penalty, normalised_components).
    """
    refs = {**DEFAULT_REFS, **(refs or {})}
    components = {}
    penalty = 0.0
    for k, w in weights.items():
        if not w:
            continue
        raw = metrics.get(k)
        if raw is None:
            continue
        norm = raw / refs.get(k, 1.0)
        components[k] = float(norm)
        penalty += float(w) * float(norm)
    return float(penalty), components
