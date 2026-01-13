#!/bin/bash
#SBATCH --job-name=SB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --time=120:00:00
#SBATCH -C a100-80gb
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=0-9
#SBATCH -o log/slurm_%A_%a.out
#SBATCH -e log/slurm_%A_%a.err

set -euo pipefail

pwd; hostname; date

module add python
module add cuda
module add cudnn
module add openmpi

source /mnt/home/yjo10/pyenv/torch/bin/activate
export PATH="$VIRTUAL_ENV/bin:$PATH"

cd "$(pwd)"

# Each "run" below is expanded into 2 array tasks:
#  - even task id  => label-index 0 (Omega_m) => save-prefix ends with _om
#  - odd task id   => label-index 1 (sigma_8) => save-prefix ends with _sig

# Base run definitions (edit here to match your intended sweep)
RUN_NAME=(
  "SB35_full30"
  "SB28_full"
  "SB28"
  "SB35_cutout15"
  "SB35_half15"
)
RUN_BOX=(
  "SB35"
  "SB28_full"
  "SB28"
  "SB35"
  "SB35"
)
RUN_SB35_MODE=(
  "full30"
  ""
  ""
  "cutout15"
  "half15"
)

# Common training args
EPOCHS=600
SCHEDULER="onecycle"
LR="1e-3"

run_idx=$((SLURM_ARRAY_TASK_ID / 2))
head=$((SLURM_ARRAY_TASK_ID % 2))

if [[ $head -eq 0 ]]; then
  LABEL_INDEX=0
  TAG="om"
else
  LABEL_INDEX=1
  TAG="sig"
fi

NAME="${RUN_NAME[$run_idx]}"
BOX="${RUN_BOX[$run_idx]}"
SB35_MODE="${RUN_SB35_MODE[$run_idx]}"

SAVE_PREFIX="${NAME}_${TAG}"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "run_idx=$run_idx"
echo "BOX=$BOX"
echo "SB35_MODE=$SB35_MODE"
echo "LABEL_INDEX=$LABEL_INDEX"
echo "SAVE_PREFIX=$SAVE_PREFIX"

cmd=(python train_sb.py
  --box "$BOX"
  --label-indices "$LABEL_INDEX"
  --save-prefix "$SAVE_PREFIX"
  --epochs "$EPOCHS"
  --scheduler "$SCHEDULER"
  --lr "$LR"
)

if [[ "$BOX" == "SB35" ]]; then
  cmd+=(--sb35-mode "$SB35_MODE")
fi

"${cmd[@]}"

date
