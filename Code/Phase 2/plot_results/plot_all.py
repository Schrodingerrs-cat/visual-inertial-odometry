"""Run every Phase 2 plotter end-to-end.

All inputs come from ``Code/Phase 2/outputs/`` (artifacts written by
``scripts/train_*.py``, ``scripts/evaluate.py``, and the study scripts).
All outputs land under ``Results/Phase2/``.

Run from ``Code/Phase 2/``:

    python3 -m plot_results.plot_all
    python3 -m plot_results.plot_all --outputs outputs --results ../../Results/Phase2
    python3 -m plot_results.plot_all --only training,trajectories
    python3 -m plot_results.plot_all --split train     # plot train trajectories too

The script is additive: every sub-plotter warns and skips when its inputs are
missing, so partial pipelines still produce what they can.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

from . import common as C
from .training_plots import plot_all_training
from .trajectory_plots import (
    plot_all_trajectories, plot_train_test_gt_overview,
)
from .distribution_plots import plot_all_distributions
from .comparison_plots import plot_all_comparisons
from .ablation_plots import plot_all_ablations


_STAGES = ("training", "trajectories", "distributions", "comparison", "ablations")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--outputs", type=Path, default=None,
                   help="Phase 2 outputs/ directory. "
                        "Defaults to sibling of the plot_results package.")
    p.add_argument("--results", type=Path, default=None,
                   help="Destination. Defaults to <repo>/Results/Phase2.")
    p.add_argument("--split", choices=("test", "train", "both"), default="both",
                   help="Which trajectory-id set to plot.")
    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated methods; default all.")
    p.add_argument("--only", type=str, default=None,
                   help=f"Comma-separated stages; subset of {_STAGES}.")
    p.add_argument("--skip", type=str, default=None,
                   help="Comma-separated stages to skip.")
    return p.parse_args()


def _traj_ids_for_split(split: str) -> tuple:
    if split == "train":
        return tuple(C.TRAIN_IDS)
    if split == "test":
        return tuple(C.TEST_IDS)
    return tuple(C.TRAIN_IDS) + tuple(C.TEST_IDS)


def _stages_to_run(only: str | None, skip: str | None) -> tuple:
    stages = list(_STAGES)
    if only:
        keep = [s.strip() for s in only.split(",") if s.strip()]
        unknown = [s for s in keep if s not in _STAGES]
        if unknown:
            raise SystemExit(f"unknown --only stages: {unknown}")
        stages = [s for s in stages if s in keep]
    if skip:
        drop = {s.strip() for s in skip.split(",") if s.strip()}
        stages = [s for s in stages if s not in drop]
    return tuple(stages)


def main() -> int:
    args = _parse()

    outputs = Path(args.outputs) if args.outputs else C.OUTPUTS_ROOT
    results = Path(args.results) if args.results else C.RESULTS_ROOT

    methods = (
        tuple(m.strip() for m in args.methods.split(",") if m.strip())
        if args.methods else C.ALL_METHODS
    )
    traj_ids = _traj_ids_for_split(args.split)
    stages = _stages_to_run(args.only, args.skip)

    C.info(f"outputs = {outputs}")
    C.info(f"results = {results}")
    C.info(f"methods = {methods}")
    C.info(f"traj_ids = {traj_ids}")
    C.info(f"stages = {stages}")
    C.ensure_dir(results)

    summary = {
        "outputs": str(outputs),
        "results": str(results),
        "methods": list(methods),
        "traj_ids": list(traj_ids),
        "stages_run": [],
        "timings": {},
    }

    traj_dir = outputs / "trajectories"
    studies_dir = outputs / "studies"
    results_csv = outputs / "results.csv"

    bundles = {}

    if "training" in stages:
        t0 = time.time()
        plot_all_training(outputs, results)
        summary["timings"]["training"] = time.time() - t0
        summary["stages_run"].append("training")

    if "trajectories" in stages:
        t0 = time.time()
        plot_train_test_gt_overview(
            traj_dir, results,
            train_ids=C.TRAIN_IDS, test_ids=C.TEST_IDS,
        )
        bundles = plot_all_trajectories(
            traj_dir, results, methods=methods, traj_ids=traj_ids,
        )
        summary["timings"]["trajectories"] = time.time() - t0
        summary["stages_run"].append("trajectories")

    if "distributions" in stages:
        t0 = time.time()
        if not bundles:
            from .common import discover_pairs, load_bundle
            pairs = discover_pairs(traj_dir, methods, traj_ids)
            for m, t in pairs:
                b = load_bundle(m, t, traj_dir)
                if b is not None:
                    bundles[(m, t)] = b
        plot_all_distributions(bundles, results)
        summary["timings"]["distributions"] = time.time() - t0
        summary["stages_run"].append("distributions")

    if "comparison" in stages:
        t0 = time.time()
        plot_all_comparisons(
            results_csv, results, methods=methods, traj_ids=traj_ids,
        )
        summary["timings"]["comparison"] = time.time() - t0
        summary["stages_run"].append("comparison")

    if "ablations" in stages:
        t0 = time.time()
        plot_all_ablations(studies_dir, results, traj_ids=C.TEST_IDS)
        summary["timings"]["ablations"] = time.time() - t0
        summary["stages_run"].append("ablations")

    # Dump a summary so downstream consumers (report, autograder) know what
    # actually got produced.
    summary_path = results / "plot_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    C.info(f"wrote summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
