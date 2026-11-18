#!/bin/bash
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train.out
#BSUB -e hpc/logs/train.err

# Run from the project root: bsub < hpc/train.sh
set -euo pipefail
cd "${LS_SUBCWD:-$(pwd)}"
source hpc/env.sh

# ===== Edit training settings here =====
MODEL_NAME='MLP'
NUM_EPOCHS=100
CONTEXT_SIZE=3
EMBED_DIM=32
HIDDEN_DIM=128
MAX_LENGTH=128
LEARNING_RATE=0.0001
PATIENCE=10
# =======================================

uv run python train.py \
    --model-name "$MODEL_NAME" \
    --num-epochs "$NUM_EPOCHS" \
    --context-size "$CONTEXT_SIZE" \
    --embed-dim "$EMBED_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --learning-rate "$LEARNING_RATE" \
    --patience "$PATIENCE" \