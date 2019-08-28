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

# TransE

+ 代码为[OpenKE](https://github.com/Niuyuhang03/OpenKE)的`GAT_data_process`分支的`TransE`模型。提交`train_FB15K237.slurm`和`train_WN18RR.slurm`运行代码 。得到的结果为实体和关系的embeddings。

# TransE->pyGAT数据处理

+ 对实体的label进行标注，代码同样为[OpenKE](https://github.com/Niuyuhang03/OpenKE)的`GAT_data_process`分支。其中原始数据由OpenKE和[DKRL](https://github.com/xrb92/DKRL)得到，具体数据来源见运行文件注释。直接运行`./FB15K237_result/FB15K237_process.py`、`./WN18RR/WN18RR_process.py`、`./WN18RR/WN18RR_sub30000_process.py`。得到结果为新数据文件`.content`、`.rel`、`.cites`，同时输出统计信息。**处理结果需要手动将复制到rgcn、RDF2VEC、rgcn、pyGAT项目中。**

+ 数据详情：

  + FB15K237共14414个实体，237种关系，25种label，297846个三元组。实体和关系的embeddings都为100维。**每种labels**的分布如下：

    ![FB15K237](https://i.loli.net/2019/08/22/t5m8TNrvK1jFxpw.png)

  + WN18RR共40943个实体，11种关系，4种label，93003个三元组。实体和关系的embeddings都为100维。**实体和labels**的分布如下：

    ![WN18RR](https://i.loli.net/2019/08/22/uTKV6FnxfwBdc2b.png)

  + WN18RR_sub30000共30943个实体，11种关系，4种label，52201个三元组。实体和关系的embeddings都为100维。**相比WN18RR，删除了10000个标签为n的实体**。**实体和labels**的分布如下：

    ![WN18RR_sub30000](https://i.loli.net/2019/08/24/VQM6JxOFrwDGIso.png)

+ 注意事项：

    + 无任何修改，直接重新运行代码时，可能会在git提示输出文件内容有修改，实际为输出内容的label顺序更换，但内容未变化。可通过git命令直接撤销对输出文件的变化。

# pyGAT

+ 模型代码为[pyGAT](https://github.com/Niuyuhang03/pyGAT)的`similar_impl_tensorflow_with_comment`分支。运行FB、WN时间一般在18-72小时之间。提交`GAT_dataset.slurm`运行。输出文件为新的`nEntity*100`维的实体embeddings结果。
+ 注意事项：
  + 在**关系**gat中，nhidden参数无效，nhidden永远等于nfeat，除全连接层外没有修改列维度的操作。
  + GAT的.slurm文件中，`--experiment`参数含义为输出文件文件夹名称，必须和`#SBATCH -o`的输出文件的文件夹名称相同。
  + cora数据集可以使用`CUDA_VISIBLE_DEVICES=0`来使用GPU运行。其他几乎所有数据集都要设置`CUDA_VISIBLE_DEVICES=1`，以保证`torch.cuda.is_available()=False`，才能不会出现out of memory。
  + 通常会每跑完一次，手动复制`GAT_dataset_result.log`为`GAT_dataset_result_final.log`，防止下一次提交覆盖log。
  + 通常会将跑完后可用于ConvE输入的结果`GAT_dataset_output.txt`保存到`./GAT_result/dataset/`中，以便ConvE使用。

## GAT数据集

|  数据集   |  实体/关系  |dataset.content|dataset.cites|dataset.rel|classes|数据处理时间|运行1个epoch时间|总时间|
| :-------: | :--------: | :-----------: | :---------: | :------: | :-----: | :----: | :----: | :------: |
|   cora    |    实体    |  2708\*1433   |     5429    |    -     |    7    |   6s   |  0.05s |   30s    |
| FB15K-237 |    实体    |  14414\*100   |    297846   |    -     |   25    |   11s  | 2min30s|   34h    |
| FB15K-237 |    关系    |  14414\*100   |    297846   | 237\*100 |   25    |   11s  | 3min30s|   18h    |
|   WN18RR  |    实体    |  40943\*100   |    93003    |    -     |    4    |   43s  |  26min |45h(100 epochs)|
|   WN18RR  |    关系    |  40943\*100   |    93003    |  11\*100 |    4    |   43s  |17min10s|   44h    |
| WN18RR_sub30000 | 实体 |  30943\*100   |    52201    |    -     |    4    |   25s  |  15min |   50h    |
| WN18RR_sub30000 | 关系 |  30943\*100   |    52201    |  11\*100 |    4    |   26s  |  5min  |   21h    |

## 模型结构和目前实验结果

### 模型结构

输入实体n\*nfeat，输入关系n\*nrel：

+ 实体GAT拼接：第一层结果n\*nhid，将nhead个n\*nhid拼接为n*(nhid\*nhead)，第二层结果n\*(nhid\*nhead)，全连接层结果n\*nclass。

+ 实体GAT平均：第一层结果n\*nhid，将nhead个n\*nhid平均为n\*nhid，第二层结果n\*nfeat，全连接层结果n\*nclass。

+ 关系GAT拼接：第一层结果n\*nfeat，将nhead个n\*nfeat拼接为n*(nfeat\*nhead)，第二层结果n\*(nfeat\*nhead)，全连接层结果n\*nclass。

+ 关系GAT平均：第一层结果n\*nfeat，将nhead个n\*nfeat平均为n\*nfeat，第二层结果n\*nfeat，全连接层结果n\*nclass。

### FB15K237对比实验

|        |实体FB15K-237拼接|实体FB15K-237平均|关系FB15K-237拼接|关系FB15K-237平均|
| :----: | :------------: | :------------: | :-------------: | :------------: |
|nhead 50|      0.3133    |                |      0.4985     |                |
|nhead 30|      0.3432    |                |      0.4837     |                |

+ nhidden实体FB15K-237为10，关系FB15K-237为100。

### WN18RR对比实验

|        | 实体WN18RR拼接  | 实体WN18RR平均 |  关系WN18RR拼接  |  关系WN18RR平均 |
| :----: | :------------: | :------------: | :-------------: | :------------: |
|nhead 10|                |                |     0.8573      |                |

+ nhidden实体WN18RR为10，关系WN18RR为100。

### WN18RR_sub30000对比实验

|        |实体WN18RR_sub30000拼接|实体WN18RR_sub30000平均|关系WN18RR_sub30000拼接|关系WN18RR_sub30000平均|
| :----: | :------------: | :------------: | :-------------: | :------------: |
|nhead 10|     0.8583     |     0.8583     |     0.8554      |     0.8576     |

+ nhidden实体WN18RR_sub30000为10，关系WN18RR_sub30000为100。

### GAT结果和baseline结果

|              |     cora   |  FB15K-237 |   WN18RR   |WN18RR_sub30000|
| :----------: | :--------: | :--------: | :---------:| :---------:|
|    实体GAT   | **0.8200** |   0.2952   |0.8609(100 epochs)|(0.8683)|
|    关系GAT   |      -     |   0.4434   |  0.8592  |   (0.8168)   |
| RDF2VEC(nb)  |   0.4948   |   0.0185   |  0.6919  |   0.6943   |
| RDF2VEC(svm) |   0.3021   |   0.1604   |  0.7817  |   0.7815   |
|     R-GCN    |   0.7374   | **0.5382** |**0.9759**| **0.9478** |

# pyGAT->ConvE数据处理

+ 代码为[ConvE](https://github.com/Niuyuhang03/ConvE)的`master_with_comment`分支。先运行`sh preprocess.sh`将`train.txt` `valid.txt` `test.txt`变为`.json`文件。再将pyGAT中输出的实体embeddings结果`/pyGAT/GAT_dataset/GAT_dataset_output.txt`复制到`/ConvE/data/dataset/`的`dataset.content`中。将pyGAT的关系embeddings`/pyGAT/data/dataset/dataset.rel`复制到`/ConvE/data/dataset/`的`dataset.rel`中。

# ConvE

+ 代码为[ConvE](https://github.com/Niuyuhang03/ConvE)的`master_with_comment`分支。每个数据集和模型提交一个`model_dataset.slurm`文件运行。
+ 注意事项
  + 参数`process`无效，代码中一定会执行process函数。
  + 都可以使用`CUDA_VISIBLE_DEVICES=0`来使用GPU运行。
  + 每次修改过数据，都需要执行“pyGAT->ConvE数据处理”部分。
  + 不能同时运行一个数据集的多个实验（process函数会将json转换为存在硬盘中的结果，运行多个时会删除其他作业的数据）

# Baseline

## RDF2VEC

+ 代码为[RDF2VEC](https://github.com/Niuyuhang03/RDF2VEC_MultiLabel)的`master`分支，直接提交`RDF2Vec.slurm`运行全部三个数据集。
+ 采用朴素贝叶斯和svm两种模型。

## R-GCN

+ 代码为[R-GCN](https://github.com/KarCute/rgcn_pytorch_implementation)的`master`分支。对每个数据集提交一个`rgcn_dataset.slurm`运行代码。
+ FB15K237可能会出现内存爆炸，需要设置epoch最大为230。WN数据集无此情况。