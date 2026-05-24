"""Mesh shape descriptors for measuring diversity across NCA growth runs.

For each finished mesh we compute a fixed-length feature vector composed of:
  - Laplace-Beltrami spectrum (first K nontrivial cotangent-Laplacian eigenvalues, scaled by area)
  - Discrete Gaussian-curvature distribution moments (mean, std, skew, kurt) at interior verts
  - Discrete mean-curvature distribution moments
  - Principal-moment ratios (3 normalized PCA eigenvalues of vertex positions)
  - Compactness: surface-area / volume^(2/3), and 1 - mesh_vol / hull_vol
  - Scale features: log(n_verts), log(n_faces)

These together cover scale, shape (oblate/prolate/sphere), bumpiness, lobedness,
and frequency content — the macroscopic axes that NCA-rule re-seeding alone
does NOT explore.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError


def _cot_laplacian(verts, faces):
    """Cotangent-Laplacian (sparse) + Voronoi-area diagonal mass matrix.

    Returns (L, M) where L is V x V (positive semidefinite, L = D - W with
    w_ij = 0.5 (cot a_ij + cot b_ij)) and M is V x V diagonal Voronoi area.
    """
    nv = verts.shape[0]
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    e0 = v2 - v1  # opposite v0
    e1 = v0 - v2
    e2 = v1 - v0

    cross = np.cross(e0, -e1)
    twice_area = np.linalg.norm(cross, axis=1) + 1e-12

    cot0 = (-(e1 * e2).sum(1)) / twice_area
    cot1 = (-(e2 * e0).sum(1)) / twice_area
    cot2 = (-(e0 * e1).sum(1)) / twice_area

    I = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1]])
    J = np.concatenate([faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0]])
    V = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

    W = sp.coo_matrix((V, (I, J)), shape=(nv, nv)).tocsr()
    d = np.asarray(W.sum(axis=1)).ravel()
    L = sp.diags(d) - W

    face_area = 0.5 * twice_area
    voronoi = np.zeros(nv, np.float64)
    for k in range(3):
        np.add.at(voronoi, faces[:, k], face_area / 3.0)
    voronoi = np.maximum(voronoi, 1e-12)
    M = sp.diags(voronoi)
    return L, M, voronoi, face_area


def _spectrum(L, M, k=20):
    """First k nontrivial generalized eigenvalues of L v = λ M v.

    Smallest-magnitude eigenpairs, dropped first (~zero from connected component).
    Result has length k; padded with NaN if eigensolver fails.
    """
    try:
        n = L.shape[0]
        if n <= k + 4:
            return np.full(k, np.nan)
        vals = spla.eigsh(
            L, k=k + 1, M=M, sigma=-1e-6, which="LM", return_eigenvectors=False,
        )
        vals = np.sort(np.real(vals))
        vals = vals[1:k + 1]
        return vals
    except Exception:
        return np.full(k, np.nan)


def _curvature_moments(verts, faces, voronoi, face_area, L, M):
    """Return (gauss_moments, mean_moments), each = (mean, std, skew, kurt)."""

    nv = verts.shape[0]
    angles = np.zeros((faces.shape[0], 3), np.float64)
    for i, (a, b, c) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
        va = verts[faces[:, a]]
        vb = verts[faces[:, b]]
        vc = verts[faces[:, c]]
        u = vb - va
        w = vc - va
        cos = (u * w).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-12)
        cos = np.clip(cos, -1.0, 1.0)
        angles[:, i] = np.arccos(cos)

    angle_sum = np.zeros(nv, np.float64)
    for k in range(3):
        np.add.at(angle_sum, faces[:, k], angles[:, k])

    boundary_mask = _boundary_vertex_mask(faces, nv)
    gauss = (2 * np.pi - angle_sum) / voronoi
    gauss = gauss[~boundary_mask]

    Lp = L @ verts
    Hvec = Lp / voronoi[:, None]
    H = 0.5 * np.linalg.norm(Hvec, axis=1)
    H = H[~boundary_mask]

    def moments(x):
        x = x[np.isfinite(x)]
        if x.size < 4:
            return np.full(4, np.nan)
        m = x.mean()
        s = x.std() + 1e-12
        z = (x - m) / s
        return np.array([m, s, (z ** 3).mean(), (z ** 4).mean() - 3.0])

    return moments(gauss), moments(H)


def _boundary_vertex_mask(faces, nv):
    """True if vertex is on a boundary (edge belongs to only one face)."""
    e = np.concatenate([
        np.stack([faces[:, 0], faces[:, 1]], 1),
        np.stack([faces[:, 1], faces[:, 2]], 1),
        np.stack([faces[:, 2], faces[:, 0]], 1),
    ])
    ek = np.sort(e, axis=1)
    keys = ek[:, 0].astype(np.int64) * (nv + 1) + ek[:, 1].astype(np.int64)
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    boundary_edge = counts[inv] == 1
    mask = np.zeros(nv, bool)
    mask[e[boundary_edge].ravel()] = True
    return mask


def _pca_ratios(verts):
    c = verts - verts.mean(0, keepdims=True)
    cov = c.T @ c / max(verts.shape[0] - 1, 1)
    w = np.linalg.eigvalsh(cov)
    w = np.sort(w)[::-1]
    w = np.maximum(w, 0.0)
    s = w.sum() + 1e-12
    return w / s


def _compactness(verts, faces, face_area):
    surface_area = float(face_area.sum())
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    mesh_vol = abs(float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0))
    try:
        hull = ConvexHull(verts)
        hull_vol = float(hull.volume)
    except (QhullError, Exception):
        hull_vol = mesh_vol + 1e-6
    sa_v23 = surface_area / max(mesh_vol, 1e-6) ** (2.0 / 3.0)
    concavity = 1.0 - mesh_vol / max(hull_vol, 1e-6)
    return np.array([sa_v23, concavity, np.log10(max(surface_area, 1e-6)),
                     np.log10(max(mesh_vol, 1e-6))])


def mesh_descriptors(verts, faces, *, k_spectrum=20):
    """Compute the full descriptor vector for a single mesh.

    verts: (V, 3) float
    faces: (F, 3) int  (must reference only valid verts)
    Returns dict with named subvectors AND a flat concatenated 'features' array.
    """
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)

    L, M, voronoi, face_area = _cot_laplacian(verts, faces)
    spec = _spectrum(L, M, k=k_spectrum)

    total_area = float(face_area.sum())
    spec_norm = spec * total_area

    g_mom, h_mom = _curvature_moments(verts, faces, voronoi, face_area, L, M)
    pca_r = _pca_ratios(verts)
    comp = _compactness(verts, faces, face_area)
    scale = np.array([
        np.log10(max(verts.shape[0], 1)),
        np.log10(max(faces.shape[0], 1)),
    ])

    flat = np.concatenate([spec_norm, g_mom, h_mom, pca_r, comp, scale])

    return {
        "spectrum": spec_norm,
        "gauss_moments": g_mom,
        "mean_moments": h_mom,
        "pca_ratios": pca_r,
        "compactness": comp,
        "scale": scale,
        "features": flat,
    }


def vendi_score(features, sigma=None):
    """Vendi score from a feature matrix using an RBF kernel.

    Effective number of distinct items. Higher = more diverse.
    """
    X = np.asarray(features, np.float64)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True) + 1e-9
    Z = X / std
    Z = np.where(np.isfinite(Z), Z, 0.0)

    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    if sigma is None:
        sigma = np.sqrt(max(np.median(d2), 1e-6))
    K = np.exp(-d2 / (2 * sigma ** 2))
    K = K / K.shape[0]
    w = np.linalg.eigvalsh(K)
    w = np.clip(w, 1e-12, None)
    H = -(w * np.log(w)).sum()
    return float(np.exp(H))


def diversity_pick(features, n):
    """Greedy farthest-point selection on the feature matrix.

    Picks n items that maximize spread. Returns list of selected indices.
    """
    X = np.asarray(features, np.float64)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True) + 1e-9
    Z = X / std
    Z = np.where(np.isfinite(Z), Z, 0.0)
    N = Z.shape[0]
    n = min(n, N)
    norms = (Z ** 2).sum(1)
    first = int(np.argmax(norms))
    picked = [first]
    dist = ((Z - Z[first]) ** 2).sum(1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dist))
        picked.append(nxt)
        dist = np.minimum(dist, ((Z - Z[nxt]) ** 2).sum(1))
    return picked


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    phi = np.arccos(1 - 2 * rng.random(n))
    theta = 2 * np.pi * rng.random(n)
    verts = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], 1)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts)
    faces = hull.simplices
    d = mesh_descriptors(verts, faces, k_spectrum=10)
    print("sphere spectrum (scaled):", np.round(d["spectrum"], 2))
    print("sphere pca ratios:", np.round(d["pca_ratios"], 3))
    print("sphere compactness:", np.round(d["compactness"], 3))
    print("flat feature vector length:", d["features"].shape[0])
