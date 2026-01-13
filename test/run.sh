#!/bin/bash
###SBATCH --job-name=SB28_ob
###SBATCH --job-name=SB35
#SBATCH --job-name=SB35_prop
###SBATCH --job-name=SB28_CoaT
###SBATCH --job-name=SIMBA
###SBATCH --job-name=TNG+SIMBA
###SBATCH --job-name=SB35
###SBATCH --job-name=SB35_cutout
#SBATCH --mail-type=ALL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail
#SBATCH --time=120:00:00               # Time limit hrs:min:sec #SBATCH -p gpu --gpus=1 -c 16
###SBATCH -p cca -c 1


###SBATCH -p cmbas -c 10
###SBATCH --partition=cmbas
###SBATCH -C skylake,opa
###SBATCH --nodes=1
###SBATCH --ntasks-per-node=
###SBATCH -p cmbas -C rome,ib
###SBATCH -p gen
###SBATCH -p gpu --gpus=<gpu_type>:<N> -c <M>

#SBATCH -C a100-80gb
###SBATCH -C a100
#SBATCH -p gpu 
###--gpus=4 -c 64
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


#deepspeed --num_gpus=2 ./SB28.py > log/stdout_SB28 2> log/stderr_SB28
#python ./SB28.py > log/stdout_SB28 2> log/stderr_SB28
#python ./SB28_twice.py > log/stdout_SB28 2> log/stderr_SB28
#python ./SB35.py > log/stdout_SB35 2> log/stderr_SB35
python ./SB35_proposal.py > log/stdout_SB35 2> log/stderr_SB35
#python ./SB35_half.py > log/stdout_SB35_half 2> log/stderr_SB35_half
#python ./SB35_cutout.py > log/stdout_SB35_cutout 2> log/stderr_SB35_cutout
#deepspeed --num_gpus=2 ./SB35.py > log/stdout_SB35 2> log/stderr_SB35
#deepspeed --num_gpus=2 ./SB28_CoaT.py > log/stdout_SB28_CoaT 2> log/stderr_SB28_CoaT
#deepspeed --num_gpus=2 ./AREPO-SIMBA.py > log/stdout_TNG 2> log/stderr_TNG
#deepspeed --num_gpus=2 ./AREPO-SIMBA.py > log/stdout_SIMBA 2> log/stderr_SIMBA
#python ./AREPO-SIMBA.py > log/stdout_TNG 2> log/stderr_TNG
#python ./AREPO-SIMBA+TNG.py > log/stdout_TS 2> log/stderr_TS
#python ./AREPO-SIMBA.py > log/stdout_SIMBA 2> log/stderr_SIMBA
#python -m torch.distributed.launch  --nproc_per_node=2 ./SB28_CoaT.py > log/stdout_SB28_CoaT 2> log/stderr_SB28_CoaT
#deepspeed --num_gpus=2 ./SB28_advanced.py > log/stdout_SB28 2> log/stderr_SB28
#deepspeed --num_gpus=2 ./SB35_advanced.py > log/stdout_SB35 2> log/stderr_SB35

date
