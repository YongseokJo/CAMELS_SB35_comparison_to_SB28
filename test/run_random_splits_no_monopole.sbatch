#!/bin/bash
#SBATCH --job-name=SB_rand_no_monopole
#SBATCH --mail-type=ALL
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --time=120:00:00
#SBATCH -C a100-80gb
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=0-19
#SBATCH -o log/slurm_rand_no_monopole_%A_%a.out
#SBATCH -e log/slurm_rand_no_monopole_%A_%a.err

set -euo pipefail

pwd; hostname; date

module add python
module add cuda
module add cudnn
module add openmpi

source /mnt/home/yjo10/pyenv/torch/bin/activate
export PATH="$VIRTUAL_ENV/bin:$PATH"

cd "$(pwd)"

# =============================================================================
# Random-split runs for SB35_cutout15 and SB28 using no_monopole runner
#
# Array layout (20 tasks total):
#   - 2 boxes: SB35_cutout15, SB28
#   - 2 heads: om (label-index 0), sig (label-index 1)
#   - 5 seeds: 0, 42, 123, 456, 789
#
# task_id = box_idx * (n_heads * n_seeds) + head_idx * n_seeds + seed_idx
# =============================================================================

# Run definitions
RUN_NAME=(
  "SB35_cutout15"
  "SB28"
)
RUN_BOX=(
  "SB35"
  "SB28"
)
RUN_SB35_MODE=(
  "cutout15"
  ""
)

# Random seeds to sweep
SEEDS=(0 42 123 456 789)

N_BOXES=${#RUN_NAME[@]}
N_HEADS=2
N_SEEDS=${#SEEDS[@]}

# Common training args
EPOCHS=600
SCHEDULER="onecycle"
LR="1e-3"

# Decode task ID
task_id=$SLURM_ARRAY_TASK_ID

box_idx=$((task_id / (N_HEADS * N_SEEDS)))
remainder=$((task_id % (N_HEADS * N_SEEDS)))
head_idx=$((remainder / N_SEEDS))
seed_idx=$((remainder % N_SEEDS))

SEED="${SEEDS[$seed_idx]}"

if [[ $head_idx -eq 0 ]]; then
  LABEL_INDEX=0
  TAG="om"
else
  LABEL_INDEX=1
  TAG="sig"
fi

NAME="${RUN_NAME[$box_idx]}"
BOX="${RUN_BOX[$box_idx]}"
SB35_MODE="${RUN_SB35_MODE[$box_idx]}"

# Include seed and no_monopole in save prefix for reproducibility
SAVE_PREFIX="${NAME}_${TAG}_no_monopole_seed${SEED}"

echo "========================================"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "box_idx=$box_idx  head_idx=$head_idx  seed_idx=$seed_idx"
echo "NAME=$NAME"
echo "BOX=$BOX"
echo "SB35_MODE=$SB35_MODE"
echo "LABEL_INDEX=$LABEL_INDEX"
echo "SEED=$SEED"
echo "SAVE_PREFIX=$SAVE_PREFIX"
echo "========================================"

cmd=(python train_sb.py
  --box "$BOX"
  --label-indices "$LABEL_INDEX"
  --save-prefix "$SAVE_PREFIX"
  --epochs "$EPOCHS"
  --scheduler "$SCHEDULER"
  --lr "$LR"
  --split-mode random
  --split-seed "$SEED"
  --individual
)

if [[ "$BOX" == "SB35" ]]; then
  cmd+=(--sb35-mode "$SB35_MODE")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

date
