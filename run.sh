#!/bin/bash
# Run FedRAPT on a specific dataset.
#
# Usage:
#   bash run.sh <dataset> [options]
#
# Datasets:
#   wisdm          WISDM v1.1, natural user-based split (36 clients)
#   ucihar         UCI-HAR, natural subject-based split (30 clients)
#   ucihar_01      UCI-HAR, Dirichlet alpha=0.1 (extreme non-IID, 30 clients)
#   ucihar_05      UCI-HAR, Dirichlet alpha=0.5 (moderate non-IID, 30 clients)
#   motionsense    MotionSense, natural subject-based split (24 clients)
#
# Options:
#   --runs N             Number of independent runs (default: 1)
#   --global_rounds N    Communication rounds (default: 100)
#   --local_epochs N     Local training epochs per round (default: 3)
#   --data_dir PATH      Override default data directory
#   --seed N             Random seed (default: unset = random per run)
#
# Examples:
#   bash run.sh wisdm
#   bash run.sh ucihar_01 --runs 5
#   bash run.sh ucihar_01 --runs 5 --global_rounds 100 --local_epochs 3
#   bash run.sh ucihar --data_dir /my/data/ucihar_npz --seed 42

set -e

# ── Defaults ──────────────────────────────────
GLOBAL_ROUNDS=100
LOCAL_EPOCHS=3
BATCH=50
RUNS=1
SEED=""
CUSTOM_DATA_DIR=""
CLIENTS_PER_ROUND=""   # set per dataset below

# ── Parse dataset argument ─────────────────────
if [ $# -eq 0 ]; then
    echo "Usage: bash run.sh <dataset> [--runs N] [--global_rounds N] [--local_epochs N] [--seed N] [--data_dir PATH]"
    echo ""
    echo "Datasets:"
    echo "  wisdm        WISDM v1.1 natural split (36 clients)"
    echo "  ucihar       UCI-HAR natural split (30 clients)"
    echo "  ucihar_01    UCI-HAR Dirichlet alpha=0.1 — extreme non-IID (30 clients)"
    echo "  ucihar_05    UCI-HAR Dirichlet alpha=0.5 — moderate non-IID (30 clients)"
    echo "  motionsense  MotionSense natural split (24 clients)"
    exit 1
fi

DATASET_KEY=$1
shift

# ── Parse optional flags ───────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)          RUNS=$2;           shift 2 ;;
        --global_rounds) GLOBAL_ROUNDS=$2;  shift 2 ;;
        --local_epochs)  LOCAL_EPOCHS=$2;   shift 2 ;;
        --batch)         BATCH=$2;          shift 2 ;;
        --data_dir)      CUSTOM_DATA_DIR=$2; shift 2 ;;
        --seed)          SEED=$2;           shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Dataset config ─────────────────────────────
case $DATASET_KEY in
    wisdm)
        DATASET="wisdm"
        DEFAULT_DIR="./data/wisdm_npz"
        OUT_TAG="wisdm"
        CLIENTS_PER_ROUND=12
        ;;
    ucihar)
        DATASET="ucihar"
        DEFAULT_DIR="./data/ucihar_npz"
        OUT_TAG="ucihar"
        CLIENTS_PER_ROUND=10
        ;;
    ucihar_01)
        DATASET="ucihar"
        DEFAULT_DIR="./data/ucihar_npz_alpha01"
        OUT_TAG="ucihar_alpha01"
        CLIENTS_PER_ROUND=10
        ;;
    ucihar_05)
        DATASET="ucihar"
        DEFAULT_DIR="./data/ucihar_npz_alpha05"
        OUT_TAG="ucihar_alpha05"
        CLIENTS_PER_ROUND=10
        ;;
    motionsense)
        DATASET="motionsense"
        DEFAULT_DIR="./data/motionsense_npz"
        OUT_TAG="motionsense"
        CLIENTS_PER_ROUND=8
        ;;
    *)
        echo "Unknown dataset: $DATASET_KEY"
        echo "Choose from: wisdm | ucihar | ucihar_01 | ucihar_05 | motionsense"
        exit 1
        ;;
esac

DATA_DIR=${CUSTOM_DATA_DIR:-$DEFAULT_DIR}
OUT_DIR="result/${OUT_TAG}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  FedRAPT"
echo "  Dataset       : $DATASET_KEY  ->  $DATA_DIR"
echo "  Global rounds : $GLOBAL_ROUNDS"
echo "  Local epochs  : $LOCAL_EPOCHS  (per client per round)"
echo "  Batch size    : $BATCH"
echo "  Clients/round : $CLIENTS_PER_ROUND"
echo "  Runs          : $RUNS  (independent repetitions)"
echo "  Seed          : ${SEED:-random}"
echo "  Output        : $OUT_DIR/"
echo "============================================================"

for RUN in $(seq 1 $RUNS); do
    echo ""
    echo "--- Run $RUN / $RUNS ---"

    # Each independent run gets a different seed (base + run - 1)
    # so results are statistically independent while still reproducible.
    SEED_ARG=""
    if [ -n "$SEED" ]; then
        RUN_SEED=$((SEED + RUN - 1))
        SEED_ARG="--seed $RUN_SEED"
    fi

    python3 main.py \
        --dataset           "$DATASET" \
        --data_dir          "$DATA_DIR" \
        --r                 "$GLOBAL_ROUNDS" \
        --E                 "$LOCAL_EPOCHS" \
        --B                 "$BATCH" \
        --clients_per_round "$CLIENTS_PER_ROUND" \
        --metrics_csv       "${OUT_DIR}/fedrapt_metrics_iter${RUN}.csv" \
        --run_id            "$RUN" \
        --checkpoint_tag    "$OUT_TAG" \
        $SEED_ARG

    echo "Saved: ${OUT_DIR}/fedrapt_metrics_iter${RUN}.csv"
done

echo ""
echo "Done. To print summary table:"
echo "  python scripts/summarize_results.py"
