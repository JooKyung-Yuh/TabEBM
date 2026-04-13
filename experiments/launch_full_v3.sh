#!/bin/bash
# ==============================================================================
# Full v3 Reproduction: Bottleneck-free launcher
#
# Strategy:
#   Stage A (light, binary datasets)  : biodeg, stock, steel
#     → GPU-per-dataset, 4 datasets parallel on 4 GPUs, cpu_jobs=6
#     → Only 3 datasets, so 1 GPU stays idle (OK)
#
#   Stage B (heavy, multi-class)      : fourier, protein, texture, collins, energy
#     → 4-GPU class-parallel, datasets run sequentially, cpu_jobs=16
#     → Uses augment_tabebm's multi-GPU path (len(gpus)>1)
#
#   Stage C (UCI leakage-free)        : clinical, support2, mushroom, auction, abalone, statlog
#     → GPU-per-dataset like Stage A, 4 at a time
#
# This design avoids:
#   - GPU oversubscription (no 2 datasets on same GPU)
#   - CPU oversubscription (cpu_jobs budgeted for concurrent jobs)
#   - Unused class-parallel path (heavy datasets use --gpus 0 1 2 3)
# ==============================================================================

cd "$(dirname "$0")/.."

OUTPUT_DIR="experiments/results/full_v3"
mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/_manifest.txt"

SEED=42
N_SYN=500
N_SPLITS=10
N_REAL_LIST="20 50 100 200 500"

W1_METHODS="baseline smote tabebm"
W2_METHODS="tvae ctgan"
W3_METHODS="baseline smote tabebm tvae ctgan"
ALL_METHODS_OPENML="baseline smote tabebm tvae ctgan"
CLASSIFIERS="lr knn mlp rf xgboost tabpfn"

# Light binary datasets (2 classes) — fast per generation
LIGHT_DATASETS="biodeg stock steel"
# Heavy multi-class datasets (8-26 classes, high feature count)
HEAVY_DATASETS="fourier protein texture collins energy"
# UCI leakage-free (all N_real=100)
UCI_DATASETS="clinical support2 mushroom auction abalone statlog"

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$MANIFEST"
}

log "=============================================="
log "  Full v3 Reproduction"
log "  Output: $OUTPUT_DIR"
log "  Start: $(date)"
log "=============================================="

# Skip helper: only skip if CSV has >= MIN_ROWS valid (status=ok) rows.
# Protects against partial/corrupted CSVs from interrupted runs.
should_skip() {
    local CSV="$1"
    local MIN_ROWS="${2:-10}"
    [ ! -f "$CSV" ] && return 1  # no file -> don't skip
    local OK_ROWS
    OK_ROWS=$(grep -c ",ok$" "$CSV" 2>/dev/null || echo 0)
    [ "$OK_ROWS" -ge "$MIN_ROWS" ] && return 0  # enough ok rows -> skip
    return 1  # not enough -> rerun
}

# ------- Stage A: light binary datasets in parallel -------
# biodeg → GPU0, stock → GPU1, steel → GPU2 (GPU3 idle for light)
# Run all N_real × methods sequentially within each dataset.
light_worker() {
    local DS=$1 GPU=$2
    local LOG="$OUTPUT_DIR/${DS}_log.txt"
    {
        for NR in $N_REAL_LIST; do
            should_skip "${OUTPUT_DIR}/${DS}_n${NR}.csv" 10 && echo "  skip $DS n=$NR (complete)" && continue
            echo "=== $DS n=$NR on GPU $GPU ==="
            conda run -n TabEBM python -u experiments/run_experiment.py \
                --dataset "$DS" --n_real "$NR" --n_splits $N_SPLITS \
                --n_syn $N_SYN --seed $SEED \
                --methods $ALL_METHODS_OPENML --classifiers $CLASSIFIERS \
                --gpus "$GPU" --cpu_jobs 6 \
                --output_dir "$OUTPUT_DIR"
        done
        echo "[DONE] $DS"
    } > "$LOG" 2>&1
}

log ""
log "===== STAGE A: Light binary datasets (parallel, GPU-per-dataset) ====="
log "  biodeg -> GPU 0"
log "  stock  -> GPU 1"
log "  steel  -> GPU 2"

light_worker biodeg 0 &
P_A1=$!
light_worker stock 1 &
P_A2=$!
light_worker steel 2 &
P_A3=$!

log "  Stage A PIDs: $P_A1 $P_A2 $P_A3"
wait $P_A1 $P_A2 $P_A3
log "===== STAGE A complete ====="

# ------- Stage B: heavy multi-class datasets sequentially with 4-GPU class-parallel -------
log ""
log "===== STAGE B: Heavy multi-class (sequential, 4-GPU class-parallel) ====="

for DS in $HEAVY_DATASETS; do
    LOG="$OUTPUT_DIR/${DS}_log.txt"
    {
        for NR in $N_REAL_LIST; do
            should_skip "${OUTPUT_DIR}/${DS}_n${NR}.csv" 10 && echo "  skip $DS n=$NR (complete)" && continue
            echo "=== $DS n=$NR on GPUs 0 1 2 3 (class-parallel) ==="
            conda run -n TabEBM python -u experiments/run_experiment.py \
                --dataset "$DS" --n_real "$NR" --n_splits $N_SPLITS \
                --n_syn $N_SYN --seed $SEED \
                --methods $ALL_METHODS_OPENML --classifiers $CLASSIFIERS \
                --gpus 0 1 2 3 --cpu_jobs 16 \
                --output_dir "$OUTPUT_DIR"
        done
        echo "[DONE] $DS"
    } > "$LOG" 2>&1
    log "  $DS done"
done
log "===== STAGE B complete ====="

# ------- Stage C: UCI datasets (parallel, GPU-per-dataset) -------
log ""
log "===== STAGE C: UCI leakage-free (parallel, GPU-per-dataset) ====="

uci_worker() {
    local DS=$1 GPU=$2
    local LOG="$OUTPUT_DIR/${DS}_log.txt"
    {
        should_skip "${OUTPUT_DIR}/${DS}_n100.csv" 10 && echo "  skip $DS (complete)" && return
        echo "=== $DS n=100 on GPU $GPU ==="
        conda run -n TabEBM python -u experiments/run_experiment.py \
            --dataset "$DS" --n_real 100 --n_splits $N_SPLITS \
            --n_syn $N_SYN --seed $SEED \
            --methods $W3_METHODS --classifiers $CLASSIFIERS \
            --gpus "$GPU" --cpu_jobs 6 \
            --output_dir "$OUTPUT_DIR"
        echo "[DONE] $DS"
    } > "$LOG" 2>&1
}

# Round 1: 4 datasets in parallel
uci_worker clinical 0 & P_C1=$!
uci_worker support2 1 & P_C2=$!
uci_worker mushroom 2 & P_C3=$!
uci_worker auction  3 & P_C4=$!
wait $P_C1 $P_C2 $P_C3 $P_C4

# Round 2: remaining 2
uci_worker abalone 0 & P_C5=$!
uci_worker statlog 1 & P_C6=$!
wait $P_C5 $P_C6
log "===== STAGE C complete ====="

# ------- Final summary -------
log ""
log "===== Final summary ====="
N_CSV=$(ls "$OUTPUT_DIR"/*.csv 2>/dev/null | grep -v summary | grep -v ranks | grep -v improvement | wc -l)
N_ERR=$(ls "$OUTPUT_DIR"/*.err 2>/dev/null | wc -l)
log "  Result CSVs: $N_CSV"
log "  Error files: $N_ERR"

conda run -n TabEBM python -u experiments/analyze.py summary --results_dir "$OUTPUT_DIR" 2>&1 | tail -60
conda run -n TabEBM python -u experiments/analyze.py plots --results_dir "$OUTPUT_DIR" 2>&1 | tail -10

log ""
log "=============================================="
log "  COMPLETE: $(date)"
log "  Results: $OUTPUT_DIR/"
log "=============================================="
