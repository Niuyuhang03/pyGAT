#!/bin/bash
#SBATCH -o ./GAT_FB15K237_4000_all/GAT_FB15K237_4000_result_%j.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J allf4             # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
#SBATCH -p sugon
time=`date +%Y%m%d-%H%M%S`
mkdir ./GAT_FB15K237_4000_all/${time}
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=0 python train.py --model_name GAT_all --dataset FB15K237_4000 --experiment GAT_FB15K237_4000_all/${time}
mv ./GAT_FB15K237_4000_all/GAT_FB15K237_4000_result_$SLURM_JOB_ID.log ./GAT_FB15K237_4000_all/${time}/GAT_FB15K237_4000_result.log
