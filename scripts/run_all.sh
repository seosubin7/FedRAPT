#!/bin/bash
# Reproduce main results from Table in the paper.
# Runs 5 independent seeds for each of the 5 dataset settings.
# Results are saved to result/{dataset}/fedrapt_metrics_iter{n}.csv

set -e

ROUNDS=100
EPOCHS=3
BATCH=50

# Set these to your preprocessed .npz directories
DATA_WISDM=${DATA_WISDM:-"./data/wisdm_npz"}
DATA_UCIHAR=${DATA_UCIHAR:-"./data/ucihar_npz"}
DATA_UCIHAR_01=${DATA_UCIHAR_01:-"./data/ucihar_npz_alpha01"}
DATA_UCIHAR_05=${DATA_UCIHAR_05:-"./data/ucihar_npz_alpha05"}
DATA_MOTIONSENSE=${DATA_MOTIONSENSE:-"./data/motionsense_npz"}

run_experiment() {
    local DATASET=$1
    local DATA_DIR=$2
    local TAG=$3
    local CPR=$4        # clients_per_round
    local OUT_DIR="result/${TAG}"

    mkdir -p "$OUT_DIR"
    echo "============================================================"
    echo " Dataset: $DATASET ($TAG)  clients/round=$CPR"
    echo "============================================================"

    for ITER in 1 2 3 4 5; do
        echo "  --- Iter $ITER/5 ---"
        python3 main.py \
            --dataset      "$DATASET" \
            --data_dir     "$DATA_DIR" \
            --r            "$ROUNDS" \
            --E            "$EPOCHS" \
            --B            "$BATCH" \
            --clients_per_round "$CPR" \
            --metrics_csv  "${OUT_DIR}/fedrapt_metrics_iter${ITER}.csv"
        echo "  Saved: ${OUT_DIR}/fedrapt_metrics_iter${ITER}.csv"
    done
}

run_experiment wisdm       "$DATA_WISDM"       wisdm          12
run_experiment ucihar      "$DATA_UCIHAR"      ucihar         10
run_experiment ucihar      "$DATA_UCIHAR_01"   ucihar_alpha01 10
run_experiment ucihar      "$DATA_UCIHAR_05"   ucihar_alpha05 10
run_experiment motionsense "$DATA_MOTIONSENSE"  motionsense     8

echo ""
echo "All experiments done. Summarizing results..."
python3 scripts/summarize_results.py
