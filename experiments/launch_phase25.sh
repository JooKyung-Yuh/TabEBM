#!/bin/bash
# ==============================================================================
# Phase 2.5 launcher (parallel)
#
# Phase A: per-EBM augmentation with default SGLD hyperparameters
#          class 0 + class 1 (distance + subsample, 4 members each) → 16 runs
# Phase B: hyperparameter sweep on a single representative EBM
#          5 params × 4 values = 20 runs
# Total: 36 SGLD runs.
#
# GPU parallelism:
#   - N_GPUS controls concurrency (default 4).
#   - Each run pins one GPU; a simple round-robin + wait-every-N_GPUS sem.
#   - stdout per run goes to experiments/results/phase25/log_*.txt (tailable).
# ==============================================================================

cd "$(dirname "$0")/.."

OUTPUT_DIR="experiments/results/phase25"
mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/_manifest.txt"

WANDB_PROJECT="tabebm-ensemble-phase25"
N_GPUS=${N_GPUS:-4}               # concurrent GPU count
NUM_SAMPLES=${NUM_SAMPLES:-500}
N_SPLITS=${N_SPLITS:-10}
CLASSIFIERS="knn lr rf xgboost mlp"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MANIFEST"; }

# Launches one run in the background, pinned to $GPU, output to $LOGFILE.
# Extra args (e.g. --sgld_step_size X, --wandb_group Y) go after the 5 fixed args.
run_one_bg() {
    local EBM_DIR=$1 IDX=$2 CLS=$3 GPU=$4 LOGFILE=$5
    shift 5
    {
        conda run -n TabEBM python experiments/phase25_run.py \
            --ebm_dir "$EBM_DIR" --ebm_idx "$IDX" --target_class "$CLS" \
            --num_samples "$NUM_SAMPLES" --n_splits "$N_SPLITS" \
            --classifiers $CLASSIFIERS \
            --gpu "$GPU" --output_dir "$OUTPUT_DIR" \
            --wandb_project "$WANDB_PROJECT" \
            "$@"
    } > "$LOGFILE" 2>&1 &
}

# Global job counter for GPU rotation; waits every N_GPUS.
JOB_IDX=0
schedule() {
    # schedule <ebm_dir> <ebm_idx> <target_class> <logfile> <extra-args...>
    local EBM_DIR=$1 IDX=$2 CLS=$3 LOGFILE=$4
    shift 4
    local GPU=$((JOB_IDX % N_GPUS))
    run_one_bg "$EBM_DIR" "$IDX" "$CLS" "$GPU" "$LOGFILE" "$@"
    JOB_IDX=$((JOB_IDX + 1))
    if [ $((JOB_IDX % N_GPUS)) -eq 0 ]; then
        wait
    fi
}

log "=============================================="
log "  Phase 2.5 launch (parallel)"
log "  N_GPUS=$N_GPUS  num_samples=$NUM_SAMPLES  n_splits=$N_SPLITS"
log "  Output: $OUTPUT_DIR"
log "=============================================="

# =====================================================================
# Phase A1 — class 0 per-EBM (default hyperparams)  : 8 runs
# =====================================================================
log ""
log "===== Phase A1: class 0 (per-EBM default) ====="
JOB_IDX=0
for METHOD in distance subsample; do
    EBM_DIR="experiments/ebms/stock_${METHOD}"
    for IDX in 0 1 2 3; do
        LOGFILE="$OUTPUT_DIR/log_A1_${METHOD}_ebm${IDX}.txt"
        log ">>> A1 ${METHOD} ebm ${IDX} class 0 -> GPU $((JOB_IDX % N_GPUS))"
        schedule "$EBM_DIR" "$IDX" 0 "$LOGFILE" \
            --wandb_group "phase-A1-class0" \
            --wandb_tags phase-A class-0 "method-${METHOD}"
    done
done
wait
log "Phase A1 done."

# =====================================================================
# Phase A2 — class 1 per-EBM (default hyperparams)  : 8 runs
# =====================================================================
log ""
log "===== Phase A2: class 1 (per-EBM default) ====="
JOB_IDX=0
for METHOD in distance subsample; do
    EBM_DIR="experiments/ebms/stock_${METHOD}_c1"
    for IDX in 0 1 2 3; do
        LOGFILE="$OUTPUT_DIR/log_A2_${METHOD}_ebm${IDX}.txt"
        log ">>> A2 ${METHOD} ebm ${IDX} class 1 -> GPU $((JOB_IDX % N_GPUS))"
        schedule "$EBM_DIR" "$IDX" 1 "$LOGFILE" \
            --wandb_group "phase-A2-class1" \
            --wandb_tags phase-A class-1 "method-${METHOD}"
    done
done
wait
log "Phase A2 done."

# =====================================================================
# Phase B — hyperparameter sweep on distance ebm_1, class 0  : 20 runs
# =====================================================================
REPR_EBM_DIR="experiments/ebms/stock_distance"
REPR_IDX=1
REPR_CLS=0

log ""
log "===== Phase B: sweep on ${REPR_EBM_DIR} ebm_${REPR_IDX} class ${REPR_CLS} ====="
JOB_IDX=0

# B.1  sgld_step_size  (default 0.1)
log "--- B.1 sgld_step_size (0.05/0.1/0.2/0.5) ---"
for SS in 0.05 0.1 0.2 0.5; do
    LOGFILE="$OUTPUT_DIR/log_B1_ss${SS}.txt"
    log ">>> B1 sgld_step_size=${SS} -> GPU $((JOB_IDX % N_GPUS))"
    schedule "$REPR_EBM_DIR" "$REPR_IDX" "$REPR_CLS" "$LOGFILE" \
        --sgld_step_size "$SS" \
        --run_tag "B1-ss${SS}" \
        --wandb_group "phase-B1-step_size" \
        --wandb_tags phase-B sweep-step_size "step_size-${SS}"
done

# B.2  sgld_noise_std  (default 0.01)
log "--- B.2 sgld_noise_std (0.0/0.01/0.05/0.1) ---"
for NS in 0.0 0.01 0.05 0.1; do
    LOGFILE="$OUTPUT_DIR/log_B2_ns${NS}.txt"
    log ">>> B2 sgld_noise_std=${NS} -> GPU $((JOB_IDX % N_GPUS))"
    schedule "$REPR_EBM_DIR" "$REPR_IDX" "$REPR_CLS" "$LOGFILE" \
        --sgld_noise_std "$NS" \
        --run_tag "B2-ns${NS}" \
        --wandb_group "phase-B2-noise_std" \
        --wandb_tags phase-B sweep-noise_std "noise_std-${NS}"
done

# B.3  sgld_steps  (default 200)
log "--- B.3 sgld_steps (50/100/200/500) ---"
for T in 50 100 200 500; do
    LOGFILE="$OUTPUT_DIR/log_B3_T${T}.txt"
    log ">>> B3 sgld_steps=${T} -> GPU $((JOB_IDX % N_GPUS))"
    schedule "$REPR_EBM_DIR" "$REPR_IDX" "$REPR_CLS" "$LOGFILE" \
        --sgld_steps "$T" \
        --run_tag "B3-T${T}" \
        --wandb_group "phase-B3-steps" \
        --wandb_tags phase-B sweep-steps "steps-${T}"
done

# B.4  starting_point_noise_std  (default 0.01)
log "--- B.4 starting_point_noise_std (0.0/0.01/0.1/0.5) ---"
for SPN in 0.0 0.01 0.1 0.5; do
    LOGFILE="$OUTPUT_DIR/log_B4_sp${SPN}.txt"
    log ">>> B4 starting_point_noise_std=${SPN} -> GPU $((JOB_IDX % N_GPUS))"
    schedule "$REPR_EBM_DIR" "$REPR_IDX" "$REPR_CLS" "$LOGFILE" \
        --starting_point_noise_std "$SPN" \
        --run_tag "B4-sp${SPN}" \
        --wandb_group "phase-B4-start_noise" \
        --wandb_tags phase-B sweep-start_noise "start_noise-${SPN}"
done

# B.5  num_samples  (default 500)  — augmentation budget
# --num_samples 는 run_one_bg 가 이미 $NUM_SAMPLES 로 넣지만 argparse 는 후자 값을 씀.
log "--- B.5 num_samples (100/200/500/1000) ---"
for NSAMP in 100 200 500 1000; do
    LOGFILE="$OUTPUT_DIR/log_B5_num${NSAMP}.txt"
    log ">>> B5 num_samples=${NSAMP} -> GPU $((JOB_IDX % N_GPUS))"
    schedule "$REPR_EBM_DIR" "$REPR_IDX" "$REPR_CLS" "$LOGFILE" \
        --num_samples "$NSAMP" \
        --run_tag "B5-num${NSAMP}" \
        --wandb_group "phase-B5-num_samples" \
        --wandb_tags phase-B sweep-num_samples "num_samples-${NSAMP}"
done
wait
log "Phase B done."

log ""
log "=============================================="
log "  Phase 2.5 complete."
log "  Result CSVs: $(ls "$OUTPUT_DIR"/*.csv 2>/dev/null | wc -l)"
log "  Per-run logs: $(ls "$OUTPUT_DIR"/log_*.txt 2>/dev/null | wc -l)"
log "  wandb project: $WANDB_PROJECT"
log "=============================================="
