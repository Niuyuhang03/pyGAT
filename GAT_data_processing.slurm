#!/bin/bash
#SBATCH -o ./GAT_data_processing/GAT_data_processing_result.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J data             # 任务名称
#SBATCH --gres=gpu:V100:1
#SBATCH -c 5
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
python data_processing.py --dataset WN18RR --hidden 100 --nb_heads 10 --experiment GAT_data_processing --rel

