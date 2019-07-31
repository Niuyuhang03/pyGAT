#!/bin/bash
#SBATCH -o ./GAT_WN18RR_sub30000_rel_epochs100/GAT_WN18RR_result.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J WNs30r100             # 任务名称
#SBATCH --gres=gpu:V100:1
#SBATCH -c 5
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=1 python train.py --dataset WN18RR_sub30000 --hidden 100 --nb_heads 10 --rel --experiment GAT_WN18RR_sub30000_rel_epochs100 --epochs 100
