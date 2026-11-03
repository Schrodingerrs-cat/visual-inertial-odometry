# VIO archive branch

Files from the local VIO working folder that are gitignored on `main`, archived
before the local folder was deleted. Not part of the main project history.

- `Code/Phase 2/` — full Phase 2 source (models, scripts, slurm, blender, configs, tests, textures)
- `Code/Phase 2/outputs/` — `best.pt` checkpoints + `results.csv` per run
- checkpoints (`*.pt`) from `Code/Phase 2/outputs/` and `Results/Phase2/step_*` (added in later commits)
- `full_results/` — per-step result plots/metrics
- `report.md`, `speech.md`, `Phase2_*.md`, `Phase2_VIO_Deck.pdf`, `Report_ph1.pdf` — notes / report / deck
- `CLAUDE.md`, `architecture/`

Not included: EuRoC `Data/` (public dataset), 6.5 GB of raw renders (`Code/Phase 2/data/raw_g`),
Blender binaries, `Results/Phase1/Phase1_video.webm` (179 MB, over GitHub's 100 MB limit).
