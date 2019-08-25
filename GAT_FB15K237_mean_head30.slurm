#!/bin/bash
#SBATCH -o ./GAT_FB15K237_mean_head30/GAT_FB15K237_result.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J FBmh3             # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
#SBATCH -p sugon             # 曙光没有时间限制
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=1 python train.py --dataset FB15K237 --hidden 10 --nb_heads 30 --experiment GAT_FB15K237_mean_head30 --mean
