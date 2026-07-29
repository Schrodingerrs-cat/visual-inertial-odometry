"""Frame-pair dataset and dataloaders for Phase 2.

A sample consists of consecutive rendered frames `(t, t+1)`, the 10 IMU samples
between them, the ground-truth relative pose from camera `t` to camera `t+1`,
the absolute pose of both frames, and the plane depth + normal in frame `t`.

Trajectory-directory naming is ``traj_{id:04d}`` (matches generate_data.py).
Training mixes all trajectories in ``config.data.train_ids = [0, 1, 2, 3]``.
Test trajectories are ``config.data.test_ids = [100, 101, 102]``.

Sampler options (see ``build_train_loader``):
  * ``mode="weighted"`` (default, Tier-1 B.2) — WeightedRandomSampler with
    per-sample weight ``(1 / bin_count) ** alpha``. ``alpha=0`` is uniform,
    ``alpha=1`` is strict balancing, ``alpha=0.5`` is the default.
  * ``mode="naive"`` — plain uniform random (shuffle=True, no sampler).
  * ``mode="small_only"`` — draw only from the smallest |Δt| quintile, used
    by the motion-magnitude study to reproduce the zero-predictor failure.

Photometric augmentation (Tier-1 texture-invariance): pass a
``PhotometricAugment`` instance to ``FramePairDataset`` for the training split.
The same random jitter is applied to both frames of a pair so geometry is
preserved. Augmentation is never applied in val / test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .geometry import relative_pose
from .stratified_sampler import (
    build_small_only_sampler,
    build_weighted_motion_sampler,
)

_IMU_WINDOW = 10
_IMAGE_HZ = 100
_IMU_HZ = 1000

# Canonical Phase 2 splits (4 train, 3 test).
DEFAULT_TRAIN_IDS: Tuple[int, ...] = (0, 1, 2, 3)
DEFAULT_TEST_IDS:  Tuple[int, ...] = (100, 101, 102)


def traj_dir_name(traj_id: int) -> str:
    """Canonical on-disk folder name for a trajectory id (4-digit zero-pad)."""
    return f"traj_{int(traj_id):04d}"


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """[qx,qy,qz,qw] → rotation matrix."""
    qx, qy, qz, qw = q
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _compute_frames_sha1(traj) -> str:
    """SHA-1 hash of concatenated per-frame SHA-1s for one trajectory.

    Used by Step K's RAFT cache to detect stale precompute when frames
    have been re-rendered (R3). Hashes the *bytes* of every PNG under
    ``traj.root/frames/`` in numeric order, then SHA-1's the
    concatenation. ~1 s per trajectory at 3000 frames.
    """
    import hashlib
    h_outer = hashlib.sha1()
    frames_dir = Path(traj.root) / "frames"
    paths = sorted(frames_dir.glob("*.png"))
    for p in paths:
        h_inner = hashlib.sha1()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h_inner.update(chunk)
        h_outer.update(h_inner.digest())
    return h_outer.hexdigest()


def _load_image(path: Path, img_size: Tuple[int, int]) -> torch.Tensor:
    """Load a PNG as a (3, H, W) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    H, W = img_size
    if img.size != (W, H):
        img = img.resize((W, H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


@dataclass
class TrajectoryData:
    """Lazily-accessible cache of one trajectory's GT + metadata."""
    traj_id: int
    root: Path
    poses_full: np.ndarray            # (N, 7)
    imu: np.ndarray                   # (N, 6)
    metadata: Dict

    @property
    def n_frames(self) -> int:
        n_samples = self.poses_full.shape[0]
        return (n_samples - 1) // _IMU_WINDOW + 1

    @classmethod
    def from_dir(cls, traj_dir: Path) -> "TrajectoryData":
        metadata = json.loads((traj_dir / "metadata.json").read_text())
        poses = np.load(traj_dir / "poses_full.npy")
        imu = np.load(traj_dir / "imu.npy")
        return cls(traj_id=int(metadata["trajectory_id"]), root=traj_dir,
                   poses_full=poses, imu=imu, metadata=metadata)

    def rel_translation_norm_at_frame(self, k: int) -> float:
        """Magnitude of camera-frame translation between kept frame k and k+1."""
        a = k * _IMU_WINDOW
        b = (k + 1) * _IMU_WINDOW
        t_a = self.poses_full[a, :3]
        t_b = self.poses_full[b, :3]
        q_a = self.poses_full[a, 3:]
        R_a = _quat_to_R(q_a)
        return float(np.linalg.norm(R_a.T @ (t_b - t_a)))


# ---------------------------------------------------------------------------
# Photometric augmentation (Tier-1 texture-invariance)
# ---------------------------------------------------------------------------

try:
    import torchvision.transforms.v2 as _TV2  # type: ignore
    _HAS_TV2 = True
except ImportError:  # pragma: no cover — fall back to torchvision.transforms
    _TV2 = None
    _HAS_TV2 = False
    import torchvision.transforms as _TV  # type: ignore


class PhotometricAugment:
    """Per-frame-pair photometric augmentation applied in training only.

    Applies the SAME random jitter to both frames of a pair so that the
    geometric relationship (and the homography ground truth) is unchanged.
    Includes ColorJitter, additive Gaussian noise, and stochastic grayscale.
    """

    def __init__(self,
                 brightness: float = 0.3,
                 contrast: float = 0.3,
                 saturation: float = 0.2,
                 hue: float = 0.05,
                 gaussian_noise_std: float = 0.01,
                 grayscale_prob: float = 0.1) -> None:
        jitter_cls = _TV2.ColorJitter if _HAS_TV2 else _TV.ColorJitter
        self.color_jitter = jitter_cls(brightness, contrast, saturation, hue)
        self.noise_std = float(gaussian_noise_std)
        self.grayscale_prob = float(grayscale_prob)

    def __call__(self, frame_t: torch.Tensor, frame_tp1: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        torch.manual_seed(seed)
        t_out = self.color_jitter(frame_t)
        torch.manual_seed(seed)
        tp1_out = self.color_jitter(frame_tp1)

        if self.noise_std > 0.0:
            t_out = t_out + torch.randn_like(t_out) * self.noise_std
            tp1_out = tp1_out + torch.randn_like(tp1_out) * self.noise_std

        if self.grayscale_prob > 0.0 and torch.rand(1).item() < self.grayscale_prob:
            t_out = t_out.mean(dim=0, keepdim=True).expand_as(t_out)
            tp1_out = tp1_out.mean(dim=0, keepdim=True).expand_as(tp1_out)

        return torch.clamp(t_out, 0.0, 1.0), torch.clamp(tp1_out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FramePairDataset
# ---------------------------------------------------------------------------

class FramePairDataset(Dataset):
    """Samples pairs of consecutive rendered frames with the IMU between them.

    Mixes all ``traj_ids`` into a single flat index; the motion-magnitude
    samplers operate across the concatenated dataset.
    """

    def __init__(self, raw_dir: Path, traj_ids: Sequence[int],
                 img_size: Tuple[int, int] = (480, 640),
                 imu_window: int = _IMU_WINDOW,
                 imu_context_length: Optional[int] = None,
                 preload_metadata: bool = True,
                 mode: str = "train",
                 photometric_augment: Optional[PhotometricAugment] = None,
                 raft_cache: bool = False,
                 raft_cache_strict: bool = True) -> None:
        self.raw_dir = Path(raw_dir)
        self.traj_ids = list(traj_ids)
        self.img_size = tuple(img_size)
        self.imu_window = int(imu_window)
        # Phase 2: optional longer causal context window for the I branch.
        # Default = imu_window (length 10) → legacy behavior, imu_context
        # is aliased to imu in __getitem__.
        self.imu_context_length = int(imu_context_length) if imu_context_length is not None \
                                  else int(imu_window)
        if self.imu_context_length < self.imu_window:
            raise ValueError(
                f"imu_context_length ({self.imu_context_length}) must be >= "
                f"imu_window ({self.imu_window}) — context can extend before "
                f"the frame pair, but cannot be shorter than the inter-frame "
                f"interval.")
        self.mode = str(mode)
        self.photometric_augment = photometric_augment
        self._trajectories: Dict[int, TrajectoryData] = {}
        self._index: List[Tuple[int, int]] = []  # (traj_id, frame_idx_k)
        self._motion_cache: Optional[np.ndarray] = None
        # Step K (knowledge distillation): optional per-traj precomputed RAFT
        # teacher predictions. ``self._raft_cache`` is keyed by traj_id; each
        # value holds R_b, t_b, sigma_b_px, valid_mask numpy arrays. Empty
        # dict when ``raft_cache=False``.
        self.raft_cache_enabled = bool(raft_cache)
        self.raft_cache_strict = bool(raft_cache_strict)
        self._raft_cache: Dict[int, Dict[str, np.ndarray]] = {}

        if preload_metadata:
            for tid in self.traj_ids:
                self._load_traj(tid)
            self._build_index()
            if self.raft_cache_enabled:
                for tid in self.traj_ids:
                    self._load_raft_cache(tid)

    def _load_traj(self, traj_id: int) -> TrajectoryData:
        if traj_id not in self._trajectories:
            traj_dir = self.raw_dir / traj_dir_name(traj_id)
            self._trajectories[traj_id] = TrajectoryData.from_dir(traj_dir)
        return self._trajectories[traj_id]

    # ------------------------------------------------------------------
    # Step K — RAFT teacher cache loading
    # ------------------------------------------------------------------

    def _load_raft_cache(self, traj_id: int) -> None:
        """Load `data/raw_g/traj_{N}/raft_predictions.npz` for one trajectory.

        On strict mode, raises if the cache is missing or its frame hash
        doesn't match the on-disk frames (R3 mitigation: stale cache
        detection). On non-strict mode (eval / forensic loaders) we silently
        skip — useful for inspecting other-step trajectories that don't
        have a cache.
        """
        traj = self._trajectories[traj_id]
        cache_path = traj.root / "raft_predictions.npz"
        if not cache_path.exists():
            if self.raft_cache_strict:
                raise FileNotFoundError(
                    f"RAFT cache missing for traj {traj_id}: {cache_path}. "
                    f"Run scripts/precompute_raft.py before training."
                )
            return
        with np.load(str(cache_path), allow_pickle=True) as blob:
            R_b = np.asarray(blob["R_b"], dtype=np.float32)
            t_b = np.asarray(blob["t_b"], dtype=np.float32)
            sigma_b = np.asarray(blob["sigma_b"], dtype=np.float32)
            valid = np.asarray(blob["valid_mask"], dtype=bool)
            meta_arr = blob.get("meta", None)
            meta = {} if meta_arr is None else dict(meta_arr.item())
        # Sanity: shape consistency vs the trajectory's frame count.
        n_pairs_expected = traj.n_frames - 1
        if R_b.shape[0] != n_pairs_expected:
            raise ValueError(
                f"RAFT cache for traj {traj_id} has {R_b.shape[0]} pairs "
                f"but trajectory has {n_pairs_expected}. Cache stale — "
                f"re-run scripts/precompute_raft.py."
            )
        # R3: hash verification. ``meta['frames_sha1']`` was computed by
        # precompute_raft over concat(frame_a_sha1, frame_b_sha1) for every
        # pair; we recompute and compare.
        expected_hash = meta.get("frames_sha1", None)
        if expected_hash is not None:
            current_hash = _compute_frames_sha1(traj)
            if str(expected_hash) != current_hash:
                msg = (f"RAFT cache hash mismatch for traj {traj_id}: "
                       f"expected={expected_hash}, current={current_hash}. "
                       f"Frames have changed since precompute — re-run "
                       f"scripts/precompute_raft.py.")
                if self.raft_cache_strict:
                    raise ValueError(msg)
                print(f"[dataloader] WARNING: {msg}")
        self._raft_cache[traj_id] = {
            "R_b": R_b, "t_b": t_b,
            "sigma_b": sigma_b, "valid_mask": valid,
        }

    def _build_index(self) -> None:
        self._index.clear()
        for tid in self.traj_ids:
            traj = self._trajectories[tid]
            for k in range(traj.n_frames - 1):
                self._index.append((tid, k))

    # ---- Public helpers --------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    @property
    def motion_magnitudes(self) -> np.ndarray:
        """Per-sample camera-frame |Δt| (metres), cached after first call."""
        if self._motion_cache is None:
            vals = np.zeros(len(self._index), dtype=np.float64)
            for i, (tid, k) in enumerate(self._index):
                vals[i] = self._trajectories[tid].rel_translation_norm_at_frame(k)
            self._motion_cache = vals
        return self._motion_cache

    def build_weighted_sampler(self, alpha: float = 0.5,
                               num_bins: int = 5,
                               bin_edges: Optional[Sequence[float]] = None,
                               num_samples: Optional[int] = None,
                               seed: int = 0):
        return build_weighted_motion_sampler(
            motion_magnitudes=self.motion_magnitudes,
            alpha=alpha, num_bins=num_bins, bin_edges=bin_edges,
            num_samples=num_samples or len(self), seed=seed)

    def build_small_only_sampler(self, threshold_m: Optional[float] = None,
                                 quantile: float = 0.2,
                                 seed: int = 0):
        return build_small_only_sampler(
            motion_magnitudes=self.motion_magnitudes,
            threshold_m=threshold_m, quantile=quantile, seed=seed)

    # ---- Sample access ----------------------------------------------------

    def _pose_abs(self, traj: TrajectoryData, sample_idx: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
        row = traj.poses_full[sample_idx]
        return _quat_to_R(row[3:]), row[:3]

    @staticmethod
    def _per_frame_altitude(traj: TrajectoryData, sample_idx: int) -> float:
        """World-frame Z of the camera at `sample_idx` in poses_full.

        Replaces the previous scalar `altitude_initial` (S5 fix). Training
        trajectories 0–2 had 9–17% altitude variation across 100 s; using
        the initial-frame altitude as the plane depth introduced p95 24–33%
        systematic bias into GT (t/d). Per-frame lookup removes that bias.
        """
        return float(traj.poses_full[sample_idx, 2])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_id, k = self._index[idx]
        traj = self._load_traj(traj_id)
        a = k * self.imu_window
        b = (k + 1) * self.imu_window
        R_wa, t_wa = self._pose_abs(traj, a)
        R_wb, t_wb = self._pose_abs(traj, b)
        R_ab, t_ab = relative_pose(R_wa, t_wa, R_wb, t_wb)
        # S5 fix: plane depth = camera's world-frame z at frame_a, not the
        # static altitude_initial.
        d = self._per_frame_altitude(traj, a)
        n_cam = np.array([0.0, 0.0, 1.0])
        imu_seg = traj.imu[a + 1: a + 1 + self.imu_window]
        assert imu_seg.shape == (self.imu_window, 6), \
            f"bad IMU shape {imu_seg.shape} at idx {idx}"

        # Phase 2 (optional): causal context window of length
        # imu_context_length, ending just before frame b. Used by the I
        # branch only; V / VI continue to consume `imu` (length imu_window).
        # Window indices: [10(k+1) - context_length, 10(k+1)) per spec.
        # Left-pad with zeros when the window extends before the
        # trajectory start (preserves k → frame-pair alignment so eval
        # comparison with prior phases remains honest).
        ctx_len = self.imu_context_length
        if ctx_len == self.imu_window:
            # No-op alias: same tensor, no extra memory or copy.
            imu_context = imu_seg
        else:
            ctx_end = (k + 1) * self.imu_window           # 10(k+1)
            ctx_start = ctx_end - ctx_len
            if ctx_start >= 0:
                imu_context = traj.imu[ctx_start:ctx_end]
            else:
                n_pad = -ctx_start
                head = np.zeros((n_pad, 6), dtype=traj.imu.dtype)
                tail = traj.imu[0:ctx_end]
                imu_context = np.concatenate([head, tail], axis=0)
            assert imu_context.shape == (ctx_len, 6), \
                f"bad imu_context shape {imu_context.shape} at idx {idx}"

        frame_a_path = traj.root / "frames" / f"{k:05d}.png"
        frame_b_path = traj.root / "frames" / f"{k + 1:05d}.png"
        img_a = _load_image(frame_a_path, self.img_size)
        img_b = _load_image(frame_b_path, self.img_size)

        if self.mode == "train" and self.photometric_augment is not None:
            img_a, img_b = self.photometric_augment(img_a, img_b)

        out = {
            "img_a": img_a,
            "img_b": img_b,
            "imu": torch.from_numpy(imu_seg.astype(np.float32)),
            "imu_context": torch.from_numpy(imu_context.astype(np.float32)),
            "R_rel": torch.from_numpy(R_ab.astype(np.float32)),
            "t_rel": torch.from_numpy(t_ab.astype(np.float32)),
            "depth": torch.tensor(float(d), dtype=torch.float32),
            "normal": torch.from_numpy(n_cam.astype(np.float32)),
            "R_wa": torch.from_numpy(R_wa.astype(np.float32)),
            "t_wa": torch.from_numpy(t_wa.astype(np.float32)),
            "R_wb": torch.from_numpy(R_wb.astype(np.float32)),
            "t_wb": torch.from_numpy(t_wb.astype(np.float32)),
            "traj_id": torch.tensor(traj_id, dtype=torch.long),
            "frame_idx": torch.tensor(k, dtype=torch.long),
        }
        if self.raft_cache_enabled and traj_id in self._raft_cache:
            cache = self._raft_cache[traj_id]
            out["raft_R"] = torch.from_numpy(cache["R_b"][k].astype(np.float32))
            out["raft_t"] = torch.from_numpy(cache["t_b"][k].astype(np.float32))
            out["raft_sigma"] = torch.tensor(float(cache["sigma_b"][k]),
                                             dtype=torch.float32)
            out["raft_valid"] = torch.tensor(bool(cache["valid_mask"][k]),
                                             dtype=torch.bool)
        return out


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

_VALID_MODES = {"weighted", "naive", "small_only"}


def _resolve_sampling_mode(sampling_mode: Optional[str]) -> str:
    """Validate and default the sampling mode string."""
    if sampling_mode is None:
        return "weighted"
    if sampling_mode not in _VALID_MODES:
        raise ValueError(f"Unknown sampling_mode={sampling_mode!r}; "
                         f"expected one of {sorted(_VALID_MODES)}")
    return sampling_mode


def build_train_loader(raw_dir: Path, traj_ids: Sequence[int],
                       img_size: Tuple[int, int],
                       batch_size: int,
                       num_workers: int,
                       sampling_mode: Optional[str] = None,
                       num_bins: int = 5,
                       bin_edges: Optional[Sequence[float]] = None,
                       alpha: float = 0.5,
                       seed: int = 0,
                       photometric_augment: Optional[PhotometricAugment] = None,
                       raft_cache: bool = False,
                       imu_context_length: Optional[int] = None,
                       ) -> Tuple[DataLoader, FramePairDataset]:
    """Build a training DataLoader. See module docstring for mode options.

    ``raft_cache=True`` (Step K only) loads precomputed RAFT teacher
    predictions from ``data/raw_g/traj_{N}/raft_predictions.npz`` for every
    trajectory in ``traj_ids`` and emits ``raft_{R, t, sigma, valid}`` keys
    on every batch. Strict mode is enforced — a missing or stale cache is
    a fatal error at training start (lesson L9 / R3).
    """
    ds = FramePairDataset(raw_dir=raw_dir, traj_ids=traj_ids,
                          img_size=img_size, mode="train",
                          imu_context_length=imu_context_length,
                          photometric_augment=photometric_augment,
                          raft_cache=raft_cache,
                          raft_cache_strict=True)
    mode = _resolve_sampling_mode(sampling_mode)

    if mode == "weighted":
        sampler = ds.build_weighted_sampler(
            alpha=alpha, num_bins=num_bins, bin_edges=bin_edges, seed=seed)
        shuffle = False
    elif mode == "small_only":
        sampler = ds.build_small_only_sampler(seed=seed)
        shuffle = False
    else:  # "naive"
        sampler = None
        shuffle = True

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        shuffle=shuffle, num_workers=num_workers,
                        pin_memory=True, drop_last=True)
    return loader, ds


def build_eval_loader(raw_dir: Path, traj_ids: Sequence[int],
                      img_size: Tuple[int, int],
                      num_workers: int = 2,
                      imu_context_length: Optional[int] = None,
                      ) -> Tuple[DataLoader, FramePairDataset]:
    """Build a DataLoader for evaluation (sequential, batch_size=1, no augment).

    ``imu_context_length`` (Phase 2 only) controls the length of the
    optional causal context window in ``batch["imu_context"]``. Default
    ``None`` → legacy behavior (imu_context aliases imu, length 10).
    """
    ds = FramePairDataset(raw_dir=raw_dir, traj_ids=traj_ids,
                          img_size=img_size, mode="eval",
                          imu_context_length=imu_context_length)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    return loader, ds
