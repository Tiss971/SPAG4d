# spag4d/spherical_grid.py
"""
Spherical grid computation for equirectangular panoramas.

Coordinate System (Y-up, right-handed):
- θ (azimuth): 0 at +X, increases counter-clockwise viewed from +Y
- φ (elevation): 0 at +Y (north pole), π at -Y (south pole)
"""

import torch
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class SphericalGrid:
    """
    Precomputed spherical grid for ERP → Gaussian conversion.
    
    Attributes:
        theta: Azimuth angles [H, W] in radians [0, 2π]
        phi: Elevation angles [H, W] in radians [0, π]
        rhat: Unit direction vectors [H, W, 3] (Y-up frame)
        tangent_right: Tangent basis right vectors [H, W, 3]
        tangent_up: Tangent basis up vectors [H, W, 3]
        device: Torch device
        stride: Downsampling factor used
    """
    theta: torch.Tensor
    phi: torch.Tensor
    rhat: torch.Tensor
    tangent_right: torch.Tensor
    tangent_up: torch.Tensor
    device: torch.device
    stride: int
    original_H: int
    original_W: int


def create_spherical_grid(
    H: int,
    W: int,
    device: torch.device,
    stride: int = 1,
    pole_rows: int = 3
) -> SphericalGrid:
    """
    Create spherical grid for an equirectangular image.
    
    Args:
        H: Original image height
        W: Original image width
        device: Torch device
        stride: Spatial downsampling factor (1, 2, 4, 8)
        pole_rows: Number of rows to exclude at poles
    
    Returns:
        SphericalGrid with precomputed geometry
    """
    # Compute strided dimensions
    H_strided = H // stride
    W_strided = W // stride
    
    # Create pixel indices (center of each strided cell)
    # u: horizontal [0, W), v: vertical [0, H)
    u = torch.arange(W_strided, device=device, dtype=torch.float32) * stride + stride / 2
    v = torch.arange(H_strided, device=device, dtype=torch.float32) * stride + stride / 2
    
    # Meshgrid: v varies along dim 0, u varies along dim 1
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    
    # ERP pixel → spherical angles
    # θ: azimuth [0, 2π], decreasing left-to-right so θ=0 at image center-right
    # Note: uu already represents cell centers (stride/2 offset applied), so no additional +0.5
    theta = (1 - uu / W) * 2 * math.pi
    
    # φ: elevation [0, π], 0 at top (north pole), π at bottom (south pole)
    phi = vv / H * math.pi
    
    # Spherical → Cartesian direction (Y-up frame)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    # Unit direction vector r̂ = [sin(φ)cos(θ), cos(φ), -sin(φ)sin(θ)]
    # Y is up (= cos(φ)):
    # φ=0 (north pole) -> cos(0)=1 -> +Y
    # φ=π (south pole) -> cos(π)=-1 -> -Y
    rhat = torch.stack([
        sin_phi * cos_theta,   # X
        cos_phi,               # Y (up)
        -sin_phi * sin_theta   # Z
    ], dim=-1)
    
    # Compute tangent basis
    # Normal points toward camera (origin) = -rhat
    normal = -rhat
    
    # World up reference
    up_world = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    # Expand for broadcasting
    up_world_expanded = up_world.view(1, 1, 3).expand(H_strided, W_strided, 3)
    
    # right = normalize(up_world × normal)
    right = torch.cross(up_world_expanded, normal, dim=-1)
    right_norm = right.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    right = right / right_norm
    
    # Handle poles where normal ≈ ±Y (cross product degenerates)
    # At poles, use Z as alternative up reference
    pole_mask = (torch.abs(cos_phi) > 0.99).unsqueeze(-1).expand(-1, -1, 3)
    z_world = torch.tensor([0.0, 0.0, 1.0], device=device)
    z_world_expanded = z_world.view(1, 1, 3).expand(H_strided, W_strided, 3)
    
    right_pole = torch.cross(z_world_expanded, normal, dim=-1)
    right_pole_norm = right_pole.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    right_pole = right_pole / right_pole_norm
    
    right = torch.where(pole_mask, right_pole, right)
    
    # up = normal × right
    up = torch.cross(normal, right, dim=-1)
    
    return SphericalGrid(
        theta=theta,
        phi=phi,
        rhat=rhat,
        tangent_right=right,
        tangent_up=up,
        device=device,
        stride=stride,
        original_H=H,
        original_W=W
    )


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion using Shepperd's method.
    
    This method is numerically stable for all orientations, unlike
    simpler methods that can have singularities.
    
    Args:
        R: Rotation matrices [..., 3, 3]
    
    Returns:
        Quaternions [..., 4] in XYZW order
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    N = R.shape[0]
    
    # Shepperd's method: choose the largest diagonal element
    # to avoid division by small numbers
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Preallocate output
    quat = torch.zeros(N, 4, device=R.device, dtype=R.dtype)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        quat[mask1, 3] = 0.25 * s  # W
        quat[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # X
        quat[mask1, 1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # Y
        quat[mask1, 2] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # Z
    
    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        quat[mask2, 3] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        quat[mask2, 0] = 0.25 * s
        quat[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        quat[mask2, 2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
    
    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        quat[mask3, 3] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        quat[mask3, 0] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        quat[mask3, 1] = 0.25 * s
        quat[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
    
    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        quat[mask4, 3] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        quat[mask4, 0] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        quat[mask4, 1] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        quat[mask4, 2] = 0.25 * s
    
    # Normalize
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    return quat.reshape(*batch_shape, 4)
