"""Smoke tests for Step K's structural-family + speed-diversity samplers.

Covers (a) the three Step K corner-shaped samplers (diamond / zigzag /
triangle), (b) the new ``simple_sinusoid_30s`` sampler, and (c) the
``target_peak_speed`` parameter on ALL speed-scalable samplers (circle,
figure-8, ellipse, diamond, zigzag, triangle, sinusoid_30s).

Verifies for each (sampler, seed, target_peak_speed):

  * shape == (SIMPLE_N_SAMPLES, 3)
  * validate_trajectory passes (peak |a|, max |roll/pitch|, ω_body, disp)
  * altitude is exactly SIMPLE_ALTITUDE_M
  * post-confinement footprint half-extent + 3.5 m fits inside the plane
  * peak |acc| > MIN_PEAK_ACC (we want visible cornerness, not stationary)
  * peak |acc| < 5.0 m/s² (much less than the 6.7 m/s² 35° tilt cap)
  * empirical peak |v_xy| matches target_peak_speed within 5%

Plus a TRAJ_ID_MAP / TEXTURE_ASSIGNMENT coverage check spanning all 13
new Step K traj_ids (312-324).

Runs in <2 s; catches sampler bugs before SLURM kicks off a render.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from blender.trajectory_library import (  # noqa: E402
    DT, FAMILY_SAMPLERS, MIN_PEAK_ACC, SIMPLE_ALTITUDE_M, SIMPLE_N_SAMPLES,
    SPEED_SCALABLE_FAMILIES, _peak_velocity_xy,
    attitude_from_acceleration, simple_circle_trajectory,
    simple_diamond_trajectory, simple_ellipse_trajectory,
    simple_figure8_trajectory, simple_sinusoid_30s_trajectory,
    simple_triangle_trajectory, simple_zigzag_trajectory,
    validate_trajectory,
)
from blender.generate_data import TEXTURE_ASSIGNMENT, TRAJ_ID_MAP  # noqa: E402

PLANE_X = 28.0
PLANE_Y = 18.69
FOOTPRINT_HALF_M = 3.5
PEAK_ACC_CEILING = 5.0   # m/s² — sanity ceiling well below 35° tilt cap.
SPEED_TARGET_TOL = 0.05  # 5% — tighter than sampler-internal 10%.

# (traj_id, family_name, sampler, seed, target_peak_speed)
# Mirrors TRAJ_ID_MAP[312..324]. Update this list whenever TRAJ_ID_MAP
# changes for Step K — the test_traj_id_map_coverage check confirms
# alignment.
SamplerCase = Tuple[int, str, callable, int, Optional[float]]  # type: ignore[valid-type]
CASES: List[SamplerCase] = [
    (312, "simple_circle",       simple_circle_trajectory,       5001, 0.10),
    (313, "simple_circle",       simple_circle_trajectory,       5002, 0.40),
    (314, "simple_figure8",      simple_figure8_trajectory,      5003, 0.15),
    (315, "simple_figure8",      simple_figure8_trajectory,      5004, 0.50),
    (316, "simple_diamond",      simple_diamond_trajectory,      5005, 0.20),
    (317, "simple_diamond",      simple_diamond_trajectory,      5006, 0.40),
    (318, "simple_zigzag",       simple_zigzag_trajectory,       5007, 0.20),
    (319, "simple_zigzag",       simple_zigzag_trajectory,       5008, 0.50),
    (320, "simple_ellipse",      simple_ellipse_trajectory,      5009, 0.30),
    (321, "simple_ellipse",      simple_ellipse_trajectory,      5010, 0.50),
    (322, "simple_triangle",     simple_triangle_trajectory,     5011, 0.30),
    (323, "simple_triangle",     simple_triangle_trajectory,     5012, 0.50),
    (324, "simple_sinusoid_30s", simple_sinusoid_30s_trajectory, 5013, 0.50),
]


class StepKSamplersTest(unittest.TestCase):

    def test_all_step_k_trajs(self) -> None:
        for traj_id, name, fn, seed, tps in CASES:
            with self.subTest(traj_id=traj_id, family=name, seed=seed,
                              target_peak_speed=tps):
                traj = fn(seed, PLANE_X, PLANE_Y, target_peak_speed=tps)
                pos = traj["pos"]

                self.assertEqual(pos.shape, (SIMPLE_N_SAMPLES, 3),
                                 f"traj {traj_id}: bad pos shape {pos.shape}")
                self.assertTrue(np.allclose(pos[:, 2], SIMPLE_ALTITUDE_M),
                                f"traj {traj_id}: altitude not constant")

                ok, reason = validate_trajectory(traj, dt=DT)
                self.assertTrue(ok,
                                f"traj {traj_id} ({name}, seed {seed}, "
                                f"tps={tps}): validate failed — {reason}")

                # Per-axis bounds inside plane (largest case: ellipse @ 0.5
                # has bbox ~ 5 m, plane 18.69 m → fits comfortably).
                half_x = 0.5 * (pos[:, 0].max() - pos[:, 0].min())
                half_y = 0.5 * (pos[:, 1].max() - pos[:, 1].min())
                self.assertLess(2 * (half_x + FOOTPRINT_HALF_M),
                                PLANE_X + 1e-3,
                                f"traj {traj_id}: x footprint exceeds plane")
                self.assertLess(2 * (half_y + FOOTPRINT_HALF_M),
                                PLANE_Y + 1e-3,
                                f"traj {traj_id}: y footprint exceeds plane")

                # Acceleration sanity.
                att = attitude_from_acceleration(pos, traj["yaw"], DT)
                peak_a = float(
                    np.max(np.linalg.norm(att["acc_world"], axis=-1)))
                self.assertGreater(peak_a, MIN_PEAK_ACC,
                                   f"traj {traj_id}: peak |a|={peak_a:.4f} "
                                   f"too low")
                self.assertLess(peak_a, PEAK_ACC_CEILING,
                                f"traj {traj_id}: peak |a|={peak_a:.4f} "
                                f"too high")

                # Speed-scaling correctness — tight 5% bound.
                if tps is not None:
                    empirical = _peak_velocity_xy(pos, DT)
                    rel_err = abs(empirical - tps) / tps
                    self.assertLess(
                        rel_err, SPEED_TARGET_TOL,
                        f"traj {traj_id}: empirical peak |v|={empirical:.4f} "
                        f"≠ target {tps:.4f} (rel_err={rel_err*100:.2f}% "
                        f"> {SPEED_TARGET_TOL*100:.0f}%). "
                        f"Confinement may have shrunk the trajectory; "
                        f"check confinement_scale = "
                        f"{traj['meta'].get('confinement_scale')}")

                attempts = traj["meta"].get("n_resample_attempts", 0)
                scale = traj["meta"].get("confinement_scale", 1.0)
                speed_scale = traj["meta"].get("speed_scale")
                emp = traj["meta"].get("empirical_peak_velocity")
                tps_str = f"{tps:.2f}" if tps is not None else "None"
                emp_str = f"{emp:.3f}" if emp is not None else "n/a"
                ss_str = f"{speed_scale:.3f}" if speed_scale is not None else "n/a"
                print(f"  traj {traj_id} {name:<20s} seed={seed} "
                      f"tps={tps_str} v_emp={emp_str} ss={ss_str} "
                      f"peak|a|={peak_a:.3f} m/s² confine={scale:.3f} "
                      f"resample={attempts}")

    def test_family_samplers_dispatch(self) -> None:
        # New families must be registered for generate_by_family().
        self.assertIn("simple_diamond",      FAMILY_SAMPLERS)
        self.assertIn("simple_zigzag",       FAMILY_SAMPLERS)
        self.assertIn("simple_triangle",     FAMILY_SAMPLERS)
        self.assertIn("simple_sinusoid_30s", FAMILY_SAMPLERS)
        # Speed scaling must be enabled for all simple_* families.
        for fam in ("simple_circle", "simple_figure8", "simple_ellipse",
                    "simple_diamond", "simple_zigzag", "simple_triangle",
                    "simple_sinusoid_30s"):
            self.assertIn(fam, SPEED_SCALABLE_FAMILIES,
                          f"family {fam} should be in SPEED_SCALABLE_FAMILIES")

    def test_traj_id_map_coverage(self) -> None:
        # Every Step K id must exist in BOTH TRAJ_ID_MAP and
        # TEXTURE_ASSIGNMENT, and the (family, seed, tps) tuple must match
        # CASES. TRAJ_ID_MAP entries for Step K are 4-tuples with
        # target_peak_speed as the 4th element.
        for traj_id, name, _fn, seed, tps in CASES:
            self.assertIn(traj_id, TRAJ_ID_MAP,
                          f"traj_id {traj_id} missing from TRAJ_ID_MAP")
            entry = TRAJ_ID_MAP[traj_id]
            self.assertEqual(len(entry), 4,
                             f"traj_id {traj_id} TRAJ_ID_MAP entry should "
                             f"be a 4-tuple (split, family, seed, "
                             f"target_peak_speed); got len={len(entry)}: "
                             f"{entry!r}")
            split, family, mapped_seed, mapped_tps = entry
            self.assertEqual(family, name,
                             f"traj_id {traj_id}: family mismatch "
                             f"map={family!r} vs case={name!r}")
            self.assertEqual(mapped_seed, seed,
                             f"traj_id {traj_id}: seed mismatch "
                             f"map={mapped_seed} vs case={seed}")
            self.assertEqual(float(mapped_tps), float(tps),
                             f"traj_id {traj_id}: target_peak_speed "
                             f"mismatch map={mapped_tps} vs case={tps}")
            self.assertIn(traj_id, TEXTURE_ASSIGNMENT,
                          f"traj_id {traj_id} missing from TEXTURE_ASSIGNMENT")
            pool, tex = TEXTURE_ASSIGNMENT[traj_id]
            self.assertEqual(tex, "tex_03.webp",
                             f"traj_id {traj_id}: texture should be "
                             f"tex_03.webp")

    def test_back_compat_no_speed_param(self) -> None:
        # Calling samplers without target_peak_speed must keep working
        # (Steps F/G/H/I/J still call the 3-arg form).
        for fn, name, seed in (
            (simple_circle_trajectory,   "circle",   4002),
            (simple_figure8_trajectory,  "figure8",  4001),
            (simple_ellipse_trajectory,  "ellipse",  4003),
            (simple_diamond_trajectory,  "diamond",  5005),
            (simple_zigzag_trajectory,   "zigzag",   5004),
            (simple_triangle_trajectory, "triangle", 5007),
        ):
            with self.subTest(family=name):
                traj = fn(seed, PLANE_X, PLANE_Y)
                self.assertEqual(traj["pos"].shape,
                                 (SIMPLE_N_SAMPLES, 3))
                ok, reason = validate_trajectory(traj, dt=DT)
                self.assertTrue(ok, f"{name} (no speed) failed: {reason}")
                # No speed-scale metadata should be set when tps is None.
                self.assertNotIn("speed_scale", traj["meta"],
                                 f"{name}: speed_scale leaked when tps=None")
                self.assertNotIn("empirical_peak_velocity", traj["meta"],
                                 f"{name}: speed metadata leaked when tps=None")


if __name__ == "__main__":
    unittest.main(verbosity=2)
