"""Reconstruct a per-pixel ERP depth map from a camera-centered 3DGS PLY.

Both da360 (spag_converter) and UniSHARP (after unisharp_format.convert_
unisharp_ply_to_spag's reorient step) write gaussians in the SAME
camera-at-origin, Y-up convention used by spag4d/spherical_grid.py to build
the panorama's spherical grid in the first place (see that module's
docstring). So the inverse mapping here — gaussian xyz -> (theta, phi) ->
ERP pixel — is exactly the geometric inverse of create_spherical_grid, and
applies unchanged to gaussians from either generator. This lets us measure
depth temporal-stability for a generator (UniSHARP) that never exposes a
raw depth map itself, only reconstructed splats.
"""
from __future__ import annotations

import math

import numpy as np


def render_depth_from_ply(ply_path: str, H: int, W: int) -> np.ndarray:
    """Z-buffer rasterize gaussian centers into an (H, W) ERP depth map.

    Depth = radial distance from camera (matches da360's depth convention).
    Each pixel takes the NEAREST point that lands in it (front-most surface,
    like a real depth buffer); pixels with no point are NaN.
    """
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    vtx = next(el for el in ply.elements if el.name == "vertex")
    x = np.asarray(vtx["x"], dtype=np.float64)
    y = np.asarray(vtx["y"], dtype=np.float64)
    z = np.asarray(vtx["z"], dtype=np.float64)

    r = np.sqrt(x * x + y * y + z * z)
    valid = r > 1e-6
    x, y, z, r = x[valid], y[valid], z[valid], r[valid]

    # Inverse of spherical_grid.create_spherical_grid:
    #   X = sin(phi)*cos(theta), Y = cos(phi), Z = -sin(phi)*sin(theta)
    #   theta = (1 - u/W) * 2*pi,  phi = v/H * pi
    phi = np.arccos(np.clip(y / r, -1.0, 1.0))  # [0, pi]
    theta = np.mod(np.arctan2(-z, x), 2 * math.pi)  # [0, 2pi)

    u = W * (1.0 - theta / (2 * math.pi))
    v = phi / math.pi * H
    px = np.clip(u.astype(np.int64), 0, W - 1)
    py = np.clip(v.astype(np.int64), 0, H - 1)

    depth_flat = np.full(H * W, np.inf, dtype=np.float64)
    np.minimum.at(depth_flat, py * W + px, r)
    depth_flat[np.isinf(depth_flat)] = np.nan
    return depth_flat.reshape(H, W)
