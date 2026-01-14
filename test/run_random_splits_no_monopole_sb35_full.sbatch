#!/bin/bash
#SBATCH --job-name=SB35full_rand_no_monopole
#SBATCH --mail-type=ALL
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --time=120:00:00
#SBATCH -C a100-80gb
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=0-9
#SBATCH -o log/slurm_rand_no_monopole_sb35full_%A_%a.out
#SBATCH -e log/slurm_rand_no_monopole_sb35full_%A_%a.err

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
# Random-split runs for SB35 full30 using the no_monopole runner
#
# Array layout (10 tasks total):
#   - 2 heads: om (label-index 0), sig (label-index 1)
#   - 5 seeds: 0, 42, 123, 456, 789
#
# task_id = head_idx * n_seeds + seed_idx
# =============================================================================

SEEDS=(0 42 123 456 789)
N_HEADS=2
N_SEEDS=${#SEEDS[@]}

# Common training args
EPOCHS=600
SCHEDULER="onecycle"
LR="1e-3"

# Decode task ID
TASK_ID=$SLURM_ARRAY_TASK_ID
head_idx=$((TASK_ID / N_SEEDS))
seed_idx=$((TASK_ID % N_SEEDS))

SEED="${SEEDS[$seed_idx]}"

if [[ $head_idx -eq 0 ]]; then
  LABEL_INDEX=0
  TAG="om"
else
  LABEL_INDEX=1
  TAG="sig"
fi

NAME="SB35_full30"
BOX="SB35"
SB35_MODE="full30"

SAVE_PREFIX="${NAME}_${TAG}_no_monopole_seed${SEED}"

echo "========================================"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "head_idx=$head_idx  seed_idx=$seed_idx"
echo "NAME=$NAME"
echo "BOX=$BOX"
echo "SB35_MODE=$SB35_MODE"
echo "LABEL_INDEX=$LABEL_INDEX"
echo "SEED=$SEED"
echo "SAVE_PREFIX=$SAVE_PREFIX"
echo "========================================"

cmd=(python train_sb.py \
  --box "$BOX" \
  --label-indices "$LABEL_INDEX" \
  --save-prefix "$SAVE_PREFIX" \
  --epochs "$EPOCHS" \
  --scheduler "$SCHEDULER" \
  --lr "$LR" \
  --split-mode random \
  --split-seed "$SEED" \
  --individual \
  --sb35-mode "$SB35_MODE")

echo "Running: ${cmd[*]}"
"${cmd[@]}"

date
