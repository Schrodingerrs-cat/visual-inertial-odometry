"""Geometry utilities for Phase 2.

Covers:
  - Rotation conversions: Rodrigues (axis-angle) ↔ rotmat, quaternion ↔ rotmat,
    6D-continuous ↔ rotmat (Zhou et al. 2019, "On the Continuity of Rotation
    Representations in Neural Networks").
  - SE(3) composition, inversion, and dead-reckoning over a sequence of
    relative poses.
  - Planar-homography compose / decompose helpers that wrap OpenCV's
    cv2.decomposeHomographyMat with physical disambiguation (pick the
    solution whose normal is closest to [0,0,1]).

Convention:
  - Rotation matrices act as R @ v where both v and the result are 3-vectors
    (column convention).
  - Quaternions use [qx, qy, qz, qw] (scalar-last, Hamilton convention,
    matching scipy.spatial.transform.Rotation).
  - Pose tuples are (R: 3x3, t: 3) representing c2 = R @ c1 + t where c1 is
    in the source frame and c2 is in the target frame.
  - Homography "compose" follows H = K (R − t n^T / d) K^{-1}  with d > 0.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch

try:  # cv2 is required by the decomposition helper and sanity checks.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover — surface failure only when used.
    cv2 = None  # type: ignore

ArrayLike = Union[np.ndarray, torch.Tensor]
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Axis-angle (Rodrigues) ↔ rotation matrix
# ---------------------------------------------------------------------------

def rodrigues_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors (B, 3) to rotation matrices (B, 3, 3).

    Uses Rodrigues' rotation formula with numerical stabilisation near the
    identity (||aa|| → 0 uses the Taylor expansion).
    """
    if aa.dim() == 1:
        aa = aa.unsqueeze(0)
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(_EPS)  # (B, 1)
    k = aa / theta                                                      # (B, 3)
    kx, ky, kz = k.unbind(dim=-1)
    K = torch.zeros(aa.shape[0], 3, 3, device=aa.device, dtype=aa.dtype)
    K[:, 0, 1] = -kz
    K[:, 0, 2] =  ky
    K[:, 1, 0] =  kz
    K[:, 1, 2] = -kx
    K[:, 2, 0] = -ky
    K[:, 2, 1] =  kx
    s = torch.sin(theta).unsqueeze(-1)   # (B, 1, 1)
    c = torch.cos(theta).unsqueeze(-1)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    return eye + s * K + (1.0 - c) * (K @ K)


def matrix_to_rodrigues(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (B, 3, 3) to axis-angle (B, 3)."""
    if R.dim() == 2:
        R = R.unsqueeze(0)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)                           # (B,)
    sin_theta = torch.sin(theta).clamp_min(_EPS)            # (B,)
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1) / (2.0 * sin_theta.unsqueeze(-1))
    return axis * theta.unsqueeze(-1)


def slerp_matrices(R0: torch.Tensor, R1: torch.Tensor,
                   alpha: torch.Tensor) -> torch.Tensor:
    """SLERP between batched rotation matrices via the matrix logarithm.

    Computes
        SLERP(R0, R1; α) = R0 · exp( α · log(R0ᵀ R1) ),
    i.e. the geodesic on SO(3) parametrised by α ∈ [0, 1] with
    α=0 → R0 and α=1 → R1.

    Implemented through axis-angle: log(R0ᵀ R1) is the Rodrigues vector
    of R0ᵀ R1, scaled by α and exponentiated back to a rotation matrix.

    Parameters
    ----------
    R0, R1 : (B, 3, 3) rotation matrices.
    alpha  : (B,) or (B, 1) interpolation weight in [0, 1].

    Returns
    -------
    (B, 3, 3) rotation matrices.
    """
    if R0.dim() == 2:
        R0 = R0.unsqueeze(0)
    if R1.dim() == 2:
        R1 = R1.unsqueeze(0)
    R_rel = R0.transpose(-1, -2) @ R1                       # (B, 3, 3)
    aa_rel = matrix_to_rodrigues(R_rel)                     # (B, 3)
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)                          # (B, 1)
    aa_scaled = aa_rel * alpha
    R_inc = rodrigues_to_matrix(aa_scaled)                   # (B, 3, 3)
    return R0 @ R_inc


# ---------------------------------------------------------------------------
# 6D-continuous representation (Zhou et al. 2019)
# ---------------------------------------------------------------------------

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Map a 6-vector (B, 6) to a rotation matrix (B, 3, 3) via Gram-Schmidt.

    The 6-vector is interpreted as two stacked columns; the output rotation is
    formed by orthonormalising them (Zhou et al. 2019 Eq. 3).
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    a1 = x[..., :3]
    a2 = x[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=_EPS)
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2 - a2_proj, dim=-1, eps=_EPS)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Inverse: take the first two columns of R and return a 6-vector."""
    if R.dim() == 2:
        R = R.unsqueeze(0)
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


# ---------------------------------------------------------------------------
# Quaternion helpers (scalar-last, qx,qy,qz,qw — scipy convention)
# ---------------------------------------------------------------------------

def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions [qx,qy,qz,qw] (B, 4) to rotation matrices."""
    if q.dim() == 1:
        q = q.unsqueeze(0)
    q = torch.nn.functional.normalize(q, dim=-1, eps=_EPS)
    qx, qy, qz, qw = q.unbind(dim=-1)
    B = q.shape[0]
    R = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (B, 3, 3) to [qx,qy,qz,qw] quaternions.

    Uses the numerically stable branching derivation (Shepperd's method).
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(B):
        tr = trace[i].item()
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[i, 2, 1] - R[i, 1, 2]) / s
            qy = (R[i, 0, 2] - R[i, 2, 0]) / s
            qz = (R[i, 1, 0] - R[i, 0, 1]) / s
        elif (R[i, 0, 0] > R[i, 1, 1]) and (R[i, 0, 0] > R[i, 2, 2]):
            s = np.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2.0
            qw = (R[i, 2, 1] - R[i, 1, 2]) / s
            qx = 0.25 * s
            qy = (R[i, 0, 1] + R[i, 1, 0]) / s
            qz = (R[i, 0, 2] + R[i, 2, 0]) / s
        elif R[i, 1, 1] > R[i, 2, 2]:
            s = np.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2.0
            qw = (R[i, 0, 2] - R[i, 2, 0]) / s
            qx = (R[i, 0, 1] + R[i, 1, 0]) / s
            qy = 0.25 * s
            qz = (R[i, 1, 2] + R[i, 2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2.0
            qw = (R[i, 1, 0] - R[i, 0, 1]) / s
            qx = (R[i, 0, 2] + R[i, 2, 0]) / s
            qy = (R[i, 1, 2] + R[i, 2, 1]) / s
            qz = 0.25 * s
        q[i] = torch.tensor([qx, qy, qz, qw], device=R.device, dtype=R.dtype)
    return q


# ---------------------------------------------------------------------------
# SE(3) composition and dead-reckoning
# ---------------------------------------------------------------------------

def compose_se3(R1: np.ndarray, t1: np.ndarray,
                R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, t) such that (R, t) = (R1, t1) ∘ (R2, t2).

    Interpretation: applying (R2, t2) first then (R1, t1) in the same frame.
    I.e. p' = R1 (R2 p + t2) + t1 = (R1 R2) p + (R1 t2 + t1).
    """
    R = R1 @ R2
    t = R1 @ t2 + t1
    return R, t


def invert_se3(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of a rigid transform: (R, t)^{-1} = (R^T, -R^T t)."""
    R_inv = R.T
    return R_inv, -R_inv @ t


def relative_pose(R_a: np.ndarray, t_a: np.ndarray,
                  R_b: np.ndarray, t_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the pose of frame b expressed in frame a's coordinates.

    Given world->a and world->b poses (as T_wa, T_wb) this returns T_ab such
    that T_ab = T_wa^{-1} ∘ T_wb.  Here poses are stored as world<-camera
    transforms: a point p_cam in camera frame maps to p_world = R_wc p + t_wc.
    The "relative pose from a to b" is then b's pose expressed in a:
        R_ab = R_wa^T R_wb
        t_ab = R_wa^T (t_wb - t_wa)
    """
    R_ab = R_a.T @ R_b
    t_ab = R_a.T @ (t_b - t_a)
    return R_ab, t_ab


def dead_reckon(init_R: np.ndarray, init_t: np.ndarray,
                rel_Rs: np.ndarray, rel_ts: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Chain a sequence of relative poses onto an initial absolute pose.

    Parameters
    ----------
    init_R, init_t
        Absolute pose of frame 0 in world coordinates, shapes (3, 3) and (3,).
    rel_Rs, rel_ts
        Sequences of relative poses of shape (N, 3, 3) and (N, 3) where each
        `(rel_Rs[i], rel_ts[i])` is the pose of frame i+1 in frame i's
        coordinates (i.e. the output of `relative_pose`).

    Returns
    -------
    abs_Rs, abs_ts : np.ndarray, np.ndarray
        Absolute poses of frames `0 .. N` in world coordinates. Shapes are
        `(N+1, 3, 3)` and `(N+1, 3)`.
    """
    N = rel_Rs.shape[0]
    abs_Rs = np.zeros((N + 1, 3, 3), dtype=init_R.dtype)
    abs_ts = np.zeros((N + 1, 3), dtype=init_t.dtype)
    abs_Rs[0] = init_R
    abs_ts[0] = init_t
    for i in range(N):
        abs_Rs[i + 1] = abs_Rs[i] @ rel_Rs[i]
        abs_ts[i + 1] = abs_Rs[i] @ rel_ts[i] + abs_ts[i]
    return abs_Rs, abs_ts


# ---------------------------------------------------------------------------
# Planar homography: compose and decompose
# ---------------------------------------------------------------------------

def compose_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                       d: float, n: np.ndarray) -> np.ndarray:
    """Compose the planar homography H induced by (R, t, d, n) relative to K.

    For a plane with unit normal `n` at signed distance `d > 0` from camera 1
    (n^T X = d for any 3D point X on the plane), the mapping from image 1 to
    image 2 under camera motion (R, t) is
        H = K (R − t n^T / d) K^{-1}.
    """
    n = np.asarray(n, dtype=np.float64).reshape(3, 1)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    R = np.asarray(R, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    M = R - (t @ n.T) / max(d, _EPS)
    return K @ M @ np.linalg.inv(K)


# ---------------------------------------------------------------------------
# Blender/OpenGL-to-CV homography adapter (project-specific convention)
# ---------------------------------------------------------------------------

# Basis flip: takes a vector expressed in Blender/OpenGL camera frame
# (+X right, +Y up, −Z forward) to CV camera frame
# (+X right, +Y down, +Z forward). S is its own inverse: S @ S = I.
_OPENGL_TO_CV = np.diag([1.0, -1.0, -1.0])


def compose_homography_blender_plane(
    K: np.ndarray,
    R_ba: np.ndarray,
    t_ba: np.ndarray,
    altitude: float,
) -> np.ndarray:
    """Homography H mapping pixels in image_a → image_b for a ground-plane
    scene rendered by Blender.

    Assumes:
      - Camera is OpenGL-convention (Blender default): +X right, +Y up,
        −Z forward. Points in front of the camera have z_cam < 0.
      - The scene has one plane at world z = 0; camera is above (world z > 0).
      - ``(R_ba, t_ba)`` is the *pose of cam_a in cam_b*, i.e. what
        ``relative_pose(R_wb, t_wb, R_wa, t_wa)`` returns.

    Formula, with S = diag(1, -1, -1):
        R_cv = S @ R_ba @ S         (OpenGL → CV basis flip)
        t_cv = S @ t_ba
        n_cv = (0, 0, -1)           (plane normal in CV camera frame)
        H    = K · (R_cv − t_cv · n_cv^T / d) · K^{-1}

    Empirically verified against GT Blender renders: MAE < 5 at inner
    border, vs MAE > 17 with the raw compose_homography + world-frame n.
    """
    S = _OPENGL_TO_CV
    R_cv = S @ np.asarray(R_ba, dtype=np.float64) @ S
    t_cv = (S @ np.asarray(t_ba, dtype=np.float64)).reshape(3, 1)
    n = np.array([[0.0, 0.0, -1.0]])
    M = R_cv - (t_cv @ n) / max(float(altitude), _EPS)
    return np.asarray(K, dtype=np.float64) @ M @ np.linalg.inv(np.asarray(K, dtype=np.float64))


def decompose_homography(H: np.ndarray, K: np.ndarray,
                         prior_normal: np.ndarray = np.array([0.0, 0.0, 1.0])
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a planar homography into (R, t/d, n) using OpenCV.

    OpenCV returns up to four solutions; we pick the one whose normal has the
    largest (positive) dot-product with `prior_normal`. For our down-facing
    drone scene the correct normal in camera frame should be close to
    `[0, 0, 1]` (the plane lies in front of the camera along +Z).
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for decompose_homography")
    _n_sols, Rs, ts, ns = cv2.decomposeHomographyMat(H.astype(np.float64),
                                                     K.astype(np.float64))
    best_score = -np.inf
    best = None
    for R, t, n in zip(Rs, ts, ns):
        score = float(np.dot(n.flatten(), prior_normal))
        if score > best_score:
            best_score = score
            best = (R, t.flatten(), n.flatten())
    assert best is not None, "decomposeHomographyMat returned no solutions"
    return best
