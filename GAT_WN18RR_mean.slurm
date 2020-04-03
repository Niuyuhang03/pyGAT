#!/bin/bash
#SBATCH -o ./GAT_WN18RR_mean/GAT_WN18RR_result_%j.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J WNm             # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
#SBATCH -p sugon             # 曙光没有时间限制
time=`date +%Y%m%d-%H%M%S`
mkdir ./GAT_WN18RR_mean/${time}
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=1 python train.py --dataset WN18RR --hidden 10 --nb_heads 10 --experiment GAT_WN18RR_mean/${time} --mean
mv ./GAT_WN18RR_mean/GAT_WN18RR_result_$SLURM_JOB_ID.log ./GAT_WN18RR_mean/${time}/GAT_WN18RR_result.log
