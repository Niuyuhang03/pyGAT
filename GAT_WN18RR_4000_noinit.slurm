#!/bin/bash
#SBATCH -o ./GAT_WN18RR_4000_noinit/GAT_WN18RR_4000_result_%j.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J w4ni             # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
#SBATCH -p sugon
time=`date +%Y%m%d-%H%M%S`
mkdir ./GAT_WN18RR_4000_noinit/${time}
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=0 python train.py --model_name GAT --dataset WN18RR_4000 --hidden 10 --nb_heads 10 --experiment GAT_WN18RR_4000_noinit/${time}
mv ./GAT_WN18RR_4000_noinit/GAT_WN18RR_4000_result_$SLURM_JOB_ID.log ./GAT_WN18RR_4000_noinit/${time}/GAT_WN18RR_4000_result.log
