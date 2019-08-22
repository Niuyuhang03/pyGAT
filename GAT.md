# GPU集群提交任务格式

## 环境配置

+ 用户`niuyh`下有已经配置好的conda虚拟环境`cuda9.1`。

+ 若配置新环境，推荐使用conda虚拟环境配置。`source /home/LAB/anaconda3/etc/profile.d/conda.sh`激活conda3，`conda create -n my_env_name python=3.6`新建虚拟环境，`conda activate my_env_name`进入虚拟环境，通过

  ```
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  conda config --set show_channel_urls yes
  ```

  进行conda换源。**进入虚拟环境后**，通过`conda install xxx`或`python3 -m pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --user`安装包，`conda deactivate`退出虚拟环境。注意cuda已经由管理员安装。

## 提交作业

+ 编写提交脚本，后缀一般为`.slurm`或`.sh`：

```bash
#!/bin/bash
#SBATCH -o GAT_WN18RR_result.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J WN_100            			 # 任务名称
#SBATCH --gres=gpu:V100:1                # 申请一张GPUV100
#SBATCH -c 5             			     # 申请CPU核数
#SBATCH -p sugon         			     # 若指定使用曙光gpu，则没有时间限制。否则限时48h。
source /home/LAB/anaconda3/etc/profile.d/conda.sh		# 激活conda3
conda activate cuda9.1									# 激活conda虚拟环境
CUDA_VISIBLE_DEVICES=0 python train.py --dataset WN18RR --hidden 10 --nb_heads 10 --epochs 100 --experiment GAT_WN18RR_epochs100			# 运行任务
```

+ 运行方法：在同级路径下`sbatch xxx.slurm`提交任务，使用`squeue -u niuyh`查看运行状态，使用`tail -f xxx.log`动态查看输出日志（任务较大时输出可能不及时），使用`scancel jobid`取消任务（jobid详见squeue第一列）。**如果输出日志有文件夹，需要先手动新建文件夹再提交**。

+ 注意事项：

  + sugon gpu较少，提交可能会出现`Priority`即排队状态。若排队超过1min建议删除此行，使用dell gpu运行耗时较短任务。
  + 申请1个gpu时，`CUDA_VISIBLE_DEVICES`应设置为0，可用于运行GAT-cora、ConvE、DistMult、ComplEx。
  + `CUDA_VISIBLE_DEVICES`若设置为1，则会使`torch.cuda.is_available()=False`，没有使用cuda。可用于运行GAT-FB15K237、GAT-WN18RR、GAT-WN18RR_sub30000。这些数据集如果使用`CUDA_VISIBLE_DEVICES=0`会出现内存不足。

# 数据集处理

## 实体和关系embeddings

+ 代码为[OpenKE](https://github.com/Niuyuhang03/OpenKE)的`GAT_data_process`分支的`TransE`模型。运行方式：提交`train_FB15K237.slurm`和`train_WN18RR.slurm` 。

## label标注

+ label标注代码同上。其中原始数据由OpenKE和[DKRL](https://github.com/xrb92/DKRL)得到，具体数据来源见运行文件注释。直接运行`./FB15K237_result/FB15K237_process.py`、`./WN18RR/WN18RR_process.py`、`./WN18RR/WN18RR_sub30000_process.py`，会生成新数据文件`.content`、`.rel`、`.cites`，输出统计信息。

+ 数据详情：

  + FB15K237共14414个实体，237种关系，25种label，297846个三元组。实体和关系的embeddings都为100维。**每种labels**的分布如下：

    ![FB15K237](https://i.loli.net/2019/08/22/t5m8TNrvK1jFxpw.png)

  + WN18RR共40943个实体，11种关系，4种label，93003个三元组。实体和关系的embeddings都为100维。**实体和labels**的分布如下：

    ![WN18RR](https://i.loli.net/2019/08/22/uTKV6FnxfwBdc2b.png)

  + WN18RR_sub30000共30943个实体，11种关系，4种label，37219个三元组。实体和关系的embeddings都为100维。**相比WN18RR，删除了10000个标签为n的实体**。**实体和labels**的分布如下：

    ![WN18RR_sub30000](https://i.loli.net/2019/08/22/uaFSCfk76R8OpVB.png)

+ 注意事项：

    + 直接重新运行代码可能会在git提示输出文件内容有修改，实际为输出内容的label顺序更换，但内容未变化。可以直接通过`git checkout filename`撤销对文件的变化。
    + **每次更新数据集，需要手动将数据复制到rgcn、RDF2VEC、rgcn、pyGAT**项目中。

## 生成ConvE、DistMult、ComplEx数据



# pyGAT

+ 模型代码为[pyGAT](https://github.com/Niuyuhang03/pyGAT)的`similar_impl_tensorflow_with_comment`分支。运行FB、WN时间一般在18-72小时之间。
+ 注意事项：
  + 在**关系**gat中，nhidden参数无效，nhidden永远等于nfeat，除全连接层外没有修改列维度的操作。
  + GAT的.slurm文件中，`--experiment`参数为文件夹名称，必须和输出文件的文件夹名称相同。

## GAT数据集

|  数据集   |  实体/关系  |dataset.content|dataset.cites|dataset.rel|classes|数据处理时间|运行1个epoch时间|总时间|
| :-------: | :--------: | :-----------: | :---------: | :------: | :-----: | :----: | :----: | :------: |
|   cora    |    实体    |  2708\*1433   |     5429    |    -     |    7    |   6s   |  0.05s |   30s    |
| FB15K-237 |    实体    |  14414\*100   |    297846   |    -     |   25    |   11s  | 2min30s|   34h    |
| FB15K-237 |    关系    |  14414\*100   |    297846   | 237\*100 |   25    |   11s  | 3min30s|   18h    |
|   WN18RR  |    实体    |  40943\*100   |    93003    |    -     |    4    |   43s  |  26min |45h(100 epochs)|
|   WN18RR  |    关系    |  40943\*100   |    93003    |  11\*100 |    4    |   43s  |17min10s|   44h    |
| WN18RR_sub30000 | 实体 |  30943\*100   |    37219    |    -     |    4    |   25s  |  |          |
| WN18RR_sub30000 | 关系 |  30943\*100   |    37219    |  11\*100 |    4    |   26s  |10min30s|   39h    |

+ pyGAT的output文件将作为ConvE、DistMult、ComplEx的输入，需要手动复制进去。

## 目前实验结果

### 关系GAT第一层nhead对比实验

| nheads|实体FB15K-237|关系FB15K-237拼接|关系FB15K-237平均|
| :---: | :---------: | :------------: | :------------: |
|   30  |             |                |                |
|   50  |             |                |                |

### 关系GAT第二层采用拼接/平均对比实验

+ FB15K-237中，nheads为50。
+ WN18RR和WN18RR_sub30000中，nheads为10。

|        |FB15K-237| WN18RR |WN18RR_sub30000|
| :----: | :-----: | :----: | :-----------: |
|  拼接  |         |        |     0.8467    |
|  平均  |         |        |     0.8231    |

### GAT结果和baseline结果

|              |     cora   |  FB15K-237 |   WN18RR   |WN18RR_sub30000|
| :----------: | :--------: | :--------: | :---------:| :---------:|
|    实体GAT   | **0.8200** |   0.2952   |0.8609(100 epochs)|      |
|    关系GAT   |      -     |   0.4434   |   0.8592   |   0.8666   |
| RDF2VEC(nb)  |   0.4948   |   0.0185   |   0.6919   |   0.6840   |
| RDF2VEC(svm) |   0.3021   |   0.1604   |   0.7817   |   0.6092   |
|     R-GCN    |   0.7374   | **0.5382** | **0.9759** | **0.9410** |

# RDF2VEC

+ 代码为[RDF2VEC](https://github.com/Niuyuhang03/RDF2VEC_MultiLabel)，直接提交`RDF2Vec.slurm`运行。
+ 采用朴素贝叶斯和svm两种模型。

# R-GCN

# ConvE、DistMult、ComplEx

+ 代码为[ConvE](https://github.com/Niuyuhang03/ConvE)的`master_with_comment`分支。
+ 运行方式：
  + 第一次运行需要执行`sh preprocess.sh`，处理.txt为.json。
  + 此后为提交每个`model_dataset.slurm`文件。
+ 注意事项
  + 参数`process`无效，代码中一定会process。