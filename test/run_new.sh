#!/bin/bash
#SBATCH --job-name=SB
#SBATCH --mail-type=ALL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail
#SBATCH --time=120:00:00               # Time limit hrs:min:sec #SBATCH -p gpu --gpus=1 -c 16
#SBATCH -C a100-80gb
#SBATCH -p gpu 
#SBATCH --ntasks-per-node=1         # 4 GPUs, 4 tasks
#SBATCH --gpus-per-node=1            # 4 GPUs
#SBATCH --cpus-per-task=12

pwd; hostname; date

#module add cuda/11.8.0
#module add modules/2.0-20220630
#module add modules/2.1.1-20230405
module add python
module add cuda
module add cudnn
module add openmpi


source /mnt/home/yjo10/pyenv/torch/bin/activate

export PATH="$VIRTUAL_ENV/bin:$PATH"

cd $(pwd)


#python train_sb.py --box SB35 --sb35-mode full30 --label-slice 0:2 --save-prefix SB35_full30 --epochs 600 --scheduler onecycle --lr 1e-3
#python train_sb.py --box SB28_full --label-slice 0:2 --save-prefix SB28_full --epochs 600 --scheduler onecycle --lr 1e-3
#python train_sb.py --box SB28 --label-slice 0:2 --save-prefix SB28 --epochs 600 --scheduler onecycle --lr 1e-3
python train_sb.py --box SB35 --sb35-mode cutout15 --label-slice 0:2 --save-prefix SB35_cutout15 --epochs 600 --scheduler onecycle --lr 1e-3
#python train_sb.py --box SB35 --sb35-mode half15 --label-slice 0:2 --save-prefix SB35_half15 --epochs 600 --scheduler onecycle --lr 1e-3

date
