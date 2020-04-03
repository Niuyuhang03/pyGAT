#!/bin/bash
#SBATCH -o ./GAT_cora/GAT_cora_result_%j.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J co             # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
time=`date +%Y%m%d-%H%M%S`
mkdir ./GAT_cora/${time}
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cora --hidden 10 --nb_heads 10 --experiment GAT_cora/${time}
mv ./GAT_cora/GAT_cora_result_$SLURM_JOB_ID.log ./GAT_cora/${time}/GAT_cora_result.log