#!/bin/bash

# execute in the general partition
##SBATCH --partition=general

# execute with 40 processes/tasks
#SBATCH --ntasks=16

# execute on 4 nodes
#SBATCH --nodes=1

# execute 4 threads per task
#SBATCH --cpus-per-task=4

# maximum time is 30 minutes
#SBATCH --time=00:30:00

# job name is my_job
#SBATCH --job-name=lmd

#only use if gpu access required, 2GPUs requested
#SBATCH --gres=gpu:1

# load environment
source /opt/ohpc/admin/lmod/8.2.10/init/bash
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
module load cuda/11.3
module load anaconda/4.9.2
conda activate lmmct
cd /home/tnguyen/coding/LMMCT/lmd/

#Debug
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

#Output
#SBATCH -o slurm_jobs/slurm_v1/stdout.txt
#SBATCH -e slurm_jobs/slurm_v1/stderr.txt

# application execution
srun python3 main.py --devices 1 --batch_size 1 --cls_loss_coef 7 --bbox_loss_coef 0 --giou_loss_coef 0 --check_val_every_n_epoch 1\
 --no_shared_voca --focal_gamma 2 --focal_alpha 0.25 --log_every_n_steps 50 --accumulate_batches 0_1 --cost_feats_weight 2_2_2_2_1\
  --lr 1.1e-04  --max_epochs 5