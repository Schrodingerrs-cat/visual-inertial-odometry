"""Smoke tests for Step J's simple-shape trajectory samplers.

Verifies that for each (sampler, seed) we use in TRAJ_ID_MAP entries 301-307:

  * shape == (SIMPLE_N_SAMPLES, 3)
  * validate_trajectory passes (no resampling failure beyond the inner loop)
  * altitude is exactly SIMPLE_ALTITUDE_M (constant; we don't add z modulation)
  * post-confinement footprint half-extent + 3.5 m fits inside the plane

Runs in <2 s; catches sampler bugs in seconds vs 10-min Blender renders.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from blender.trajectory_library import (  # noqa: E402
    DT, FAMILY_SAMPLERS, SIMPLE_ALTITUDE_M, SIMPLE_N_SAMPLES,
    simple_circle_trajectory, simple_ellipse_trajectory,
    simple_figure8_trajectory, validate_trajectory,
)

PLANE_X = 28.0
PLANE_Y = 18.69
FOOTPRINT_HALF_M = 3.5

# (family_name, sampler, seed) — mirror the entries we add to
# blender/generate_data.py::TRAJ_ID_MAP for trajs 301-307.
CASES = [
    ("simple_figure8", simple_figure8_trajectory, 4001),  # traj 301
    ("simple_circle",  simple_circle_trajectory,  4002),  # traj 302
    ("simple_ellipse", simple_ellipse_trajectory, 4003),  # traj 303
    ("simple_figure8", simple_figure8_trajectory, 4004),  # traj 304
    ("simple_figure8", simple_figure8_trajectory, 4005),  # traj 305
    ("simple_circle",  simple_circle_trajectory,  4006),  # traj 306
    ("simple_ellipse", simple_ellipse_trajectory, 4007),  # traj 307
]


class SimpleSamplersTest(unittest.TestCase):
    def test_all_simple_trajs(self) -> None:
        for name, fn, seed in CASES:
            with self.subTest(family=name, seed=seed):
                traj = fn(seed, PLANE_X, PLANE_Y)
                pos = traj["pos"]
                self.assertEqual(pos.shape, (SIMPLE_N_SAMPLES, 3),
                                 f"{name} seed={seed}: bad pos shape {pos.shape}")
                # Constant altitude (we don't add z modulation in simple-* samplers).
                self.assertTrue(np.allclose(pos[:, 2], SIMPLE_ALTITUDE_M),
                                f"{name} seed={seed}: altitude not constant at "
                                f"{SIMPLE_ALTITUDE_M}")
                # Validity check (peak velocity, accel, omega, footprint).
                ok, reason = validate_trajectory(traj, dt=DT)
                self.assertTrue(ok,
                                f"{name} seed={seed}: validate failed — {reason}")
                # Footprint sanity post-confinement.
                half_x = 0.5 * (pos[:, 0].max() - pos[:, 0].min())
                half_y = 0.5 * (pos[:, 1].max() - pos[:, 1].min())
                self.assertLess(2 * (half_x + FOOTPRINT_HALF_M), PLANE_X + 1e-3,
                                f"{name} seed={seed}: x footprint exceeds plane")
                self.assertLess(2 * (half_y + FOOTPRINT_HALF_M), PLANE_Y + 1e-3,
                                f"{name} seed={seed}: y footprint exceeds plane")
                scale = traj["meta"].get("confinement_scale", 1.0)
                attempts = traj["meta"].get("n_resample_attempts", 0)
                print(f"  {name:<16} seed={seed} OK "
                      f"(scale={scale:.3f}, resample_attempts={attempts}, "
                      f"half_x={half_x:.2f} m, half_y={half_y:.2f} m)")

    def test_family_samplers_dispatch(self) -> None:
        # Ensure both new families are registered for generate_by_family().
        self.assertIn("simple_circle", FAMILY_SAMPLERS)
        self.assertIn("simple_ellipse", FAMILY_SAMPLERS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
