#!/bin/bash
# submit_all.sh — Phase 2 pipeline orchestrator with sbatch dependency chaining.
#
# DAG:
#   prepare_textures
#        └─► gen_shard{0,1,2,3} (parallel x4)
#                 └─► run_sanity
#                        ├─► train_v ┐
#                        └─► train_i ┴─► train_vi ─► evaluate
#                                                          ├─► motion_magnitude
#                                                          └─► ablation
#
# Usage:
#   bash slurm/submit_all.sh                  # full pipeline
#   bash slurm/submit_all.sh --skip-gen       # skip data generation
#   bash slurm/submit_all.sh --skip-sanity    # skip sanity checks
#   bash slurm/submit_all.sh --skip-training  # skip train_v / train_i / train_vi
#   bash slurm/submit_all.sh --only gen|sanity|train|eval|motion|ablation
#   bash slurm/submit_all.sh --dry-run        # print sbatch commands, submit nothing

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_ROOT="$(cd "$HERE/.." && pwd)"
cd "$PHASE2_ROOT"

# --- Flags -------------------------------------------------------------------
SKIP_TEXPREP=0
SKIP_GEN=0
SKIP_SANITY=0
SKIP_TRAINING=0
ONLY=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-texprep)  SKIP_TEXPREP=1;;
        --skip-gen)      SKIP_GEN=1;;
        --skip-sanity)   SKIP_SANITY=1;;
        --skip-training) SKIP_TRAINING=1;;
        --only)          ONLY="$2"; shift;;
        --dry-run)       DRY_RUN=1;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0;;
        *)
            echo "[submit] unknown flag: $1" >&2; exit 2;;
    esac
    shift
done

# --- Idempotency helpers -----------------------------------------------------
# Return the jobid of an already-queued job with the given name, or empty.
existing_jobid () {
    local jobname="$1"
    squeue -u "$USER" -h -o "%i %j" 2>/dev/null \
        | awk -v n="$jobname" '$2 == n {print $1; exit}'
}

# Wrapper around sbatch: honors --dry-run, idempotency, and --parsable parsing.
# Usage: submit <friendly-label> <script> [--dependency=...]
submit () {
    local label="$1"; shift
    local script="$1"; shift
    local jobname
    jobname="$(awk '/^#SBATCH --job-name=/{sub(/^#SBATCH --job-name=/,""); print; exit}' "$script")"

    local existing
    existing="$(existing_jobid "$jobname")"
    if [[ -n "$existing" ]]; then
        echo "[submit] $label ($jobname): already queued as job $existing — reusing." >&2
        echo "$existing"
        return 0
    fi

    local cmd=(sbatch --parsable "$@" "$script")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[submit] DRY: ${cmd[*]}" >&2
        echo "DRY_$label"
        return 0
    fi
    local jid
    jid="$("${cmd[@]}" | awk -F';' '{print $1}')"
    if [[ -z "$jid" ]]; then
        echo "[submit] ERROR: sbatch returned no jobid for $label" >&2
        exit 1
    fi
    echo "[submit] $label ($jobname): job $jid" >&2
    echo "$jid"
}

# --- Stage filter ------------------------------------------------------------
want_stage () {
    local s="$1"
    [[ -z "$ONLY" || "$ONLY" == "$s" ]]
}

# --- Stage 0: texture preparation --------------------------------------------
# Normalizes Data/textures_raw/*.{jpg,jpeg,png,webp} (7 files) into
# Data/textures_processed/tex_00..tex_06.webp and splits into train/test.
TEXPREP_JID=""
if [[ "$SKIP_TEXPREP" == "0" ]] && want_stage texprep; then
    TEXPREP_JID="$(submit prepare_textures slurm/prepare_textures.sh)"
fi

TEXPREP_DEP=""
[[ -n "$TEXPREP_JID" && "$TEXPREP_JID" != DRY_* ]] && \
    TEXPREP_DEP="--dependency=afterok:$TEXPREP_JID"

# --- Stage 1: data generation (4 parallel shards) ----------------------------
# Shard 0 → (0, 100), Shard 1 → (1, 101), Shard 2 → (2, 102), Shard 3 → (3, 200).
# With 100 s trajectories, each shard takes ~160 min; 4 parallel → ~2.7 h total.
GEN_JIDS=()
if [[ "$SKIP_GEN" == "0" ]] && want_stage gen; then
    for sh in 0 1 2 3; do
        jid="$(submit "gen_shard$sh" "slurm/gen_data_shard${sh}.sh" ${TEXPREP_DEP})"
        GEN_JIDS+=("$jid")
    done
fi

# --- Stage 2: sanity checks --------------------------------------------------
SANITY_DEP=""
if [[ ${#GEN_JIDS[@]} -gt 0 ]]; then
    SANITY_DEP="--dependency=afterok:$(IFS=:; echo "${GEN_JIDS[*]}")"
fi
SANITY_JID=""
if [[ "$SKIP_SANITY" == "0" ]] && want_stage sanity; then
    SANITY_JID="$(submit sanity slurm/run_sanity.sh ${SANITY_DEP})"
fi

# --- Stage 3: training V + I (parallel), then VI ----------------------------
TRAIN_DEP=""
[[ -n "$SANITY_JID" && "$SANITY_JID" != DRY_* ]] && TRAIN_DEP="--dependency=afterok:$SANITY_JID"
V_JID=""
I_JID=""
VI_JID=""
if [[ "$SKIP_TRAINING" == "0" ]] && want_stage train; then
    V_JID="$(submit train_v slurm/train_v.sh ${TRAIN_DEP})"
    I_JID="$(submit train_i slurm/train_i.sh ${TRAIN_DEP})"
    VI_DEPS=()
    [[ -n "$V_JID" && "$V_JID" != DRY_* ]] && VI_DEPS+=("$V_JID")
    [[ -n "$I_JID" && "$I_JID" != DRY_* ]] && VI_DEPS+=("$I_JID")
    VI_DEP=""
    [[ ${#VI_DEPS[@]} -gt 0 ]] && VI_DEP="--dependency=afterok:$(IFS=:; echo "${VI_DEPS[*]}")"
    VI_JID="$(submit train_vi slurm/train_vi.sh ${VI_DEP})"
fi

# --- Stage 4: evaluation -----------------------------------------------------
EVAL_JID=""
if want_stage eval; then
    EVAL_DEP=""
    [[ -n "$VI_JID" && "$VI_JID" != DRY_* ]] && EVAL_DEP="--dependency=afterok:$VI_JID"
    EVAL_JID="$(submit evaluate slurm/evaluate.sh ${EVAL_DEP})"
fi

# --- Stage 5: studies (parallel after eval) ----------------------------------
STUDY_DEP=""
[[ -n "$EVAL_JID" && "$EVAL_JID" != DRY_* ]] && STUDY_DEP="--dependency=afterok:$EVAL_JID"
if want_stage motion;   then submit motion_magnitude slurm/motion_magnitude.sh ${STUDY_DEP} >/dev/null; fi
if want_stage ablation; then submit ablation         slurm/ablation.sh         ${STUDY_DEP} >/dev/null; fi

echo "[submit] done."
