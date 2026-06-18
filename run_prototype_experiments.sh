#!/bin/bash
# Prototype update strategy comparison
# 3 strategies x 5 iterations, no fixed seed
# EMA (FedRAPT default) uses existing results from result/ucihar0.1/

set -e

DATASET="ucihar"
DATA_DIR="${DATA_UCIHAR_01:-./data/ucihar_npz_alpha01}"
ROUNDS=100
LOG_DIR="result/proto_experiments"
mkdir -p "$LOG_DIR"

for STRATEGY in direct simple_avg cumulative; do
    for ITER in 1 2 3 4 5; do
        echo "============================================================"
        echo " Strategy: $STRATEGY  |  Iter: $ITER/5"
        echo "============================================================"
        python3 main.py \
            --dataset "$DATASET" \
            --data_dir "$DATA_DIR" \
            --r "$ROUNDS" \
            --proto_update_strategy "$STRATEGY" \
            --prototype_beta 0.9 \
            --run_id "$ITER" \
            --checkpoint_tag "proto_${STRATEGY}" \
            --metrics_csv "$LOG_DIR/strategy_${STRATEGY}_iter${ITER}.csv" \
            > "$LOG_DIR/strategy_${STRATEGY}_iter${ITER}.log" 2>&1
        echo " Done: strategy_${STRATEGY}_iter${ITER}"
    done
done

echo ""
echo "============================================================"
echo " All experiments done. Results in $LOG_DIR/"
echo "============================================================"
