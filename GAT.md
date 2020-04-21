# GPU集群提交任务格式

# TransE

+ 代码为[OpenKE](https://github.com/Niuyuhang03/OpenKE) 的`GAT_data_process`分支的`TransE`模型。提交`train_FB15K237.slurm`和`train_WN18RR.slurm`运行代码 。得到的结果为实体和关系的embeddings。

# TransE->pyGAT数据处理

+ 对实体的label进行标注，代码同样为[OpenKE](https://github.com/Niuyuhang03/OpenKE) 的`GAT_data_process`分支。其中原始数据由OpenKE和[DKRL](https://github.com/xrb92/DKRL) 得到，具体数据来源见运行文件注释。直接运行`./FB15K237_result/FB15K237_process.py`。得到结果为新数据文件`.content`、`.rel`、`.cites`，同时输出统计信息。**处理结果需要手动将复制到pyGAT项目中。**

+ 无任何修改，直接重新运行代码时，可能会在git提示输出文件内容有修改，实际为输出内容的label顺序更换，但内容未变化。可通过git命令直接撤销对输出文件的变化。

+ nltk需要下载语料，`import nltk; nltk.download('wordnet')`

+ 数据详情：

  + FB15K237共14541个实体，237种关系，25种label，310116个三元组。实体和关系的embeddings都为100维。**每种labels**的分布如下：

    ![FB15K237](https://i.loli.net/2020/04/21/AlDC1eYysnk24Uv.png)

  + ==fb15k237中部分结果难以分类别，设置label为file，共127个实体，存储在.dele中。==

  + WN18RR共40943个实体，11种关系，4种label，93003个三元组。实体和关系的embeddings都为100维。**实体和labels**的分布如下：
  
    ![WN18RR](https://i.loli.net/2020/04/21/sRAFfEKx5IkYZjB.png)

# pyGAT

+ 模型代码为[pyGAT](https://github.com/Niuyuhang03/pyGAT)的`similar_impl_tensorflow_with_comment`分支。运行FB、WN时间一般在18-72小时之间。提交`GAT_dataset.slurm`运行。输出文件`GAT_dataset_output.txt`为新的实体embeddings结果。**需要择优将output复制到`./pyGAT/GAT_result/`对应文件夹，以便ConvE使用。**
+ 注意事项：
  + 在**关系**gat中，nhidden参数无效，nhidden永远等于nfeat，除全连接层外没有修改列维度的操作。
  + GAT的.slurm文件中，`--experiment`参数含义为输出文件文件夹名称，必须和`#SBATCH -o`的输出文件的文件夹名称相同。
  + cora数据集可以使用`CUDA_VISIBLE_DEVICES=0`来使用GPU运行。其他几乎所有数据集都要设置`CUDA_VISIBLE_DEVICES=1`，以保证`torch.cuda.is_available()=False`，才能不会出现out of memory。

## GAT数据集

|  数据集   |  实体/关系  |dataset.content|dataset.cites|dataset.rel|classes|
| :-------: | :--------: | :-----------: | :---------: | :------: | :-----: |
|   cora    |    实体    |  2708\*1433   |     5429    |    -     |    7    |
| FB15K-237 |    关系    | 14541\*100 | 310116   | 237\*100 |   25    |
|   WN18RR  |    关系    |  40943\*100   |    93003    |  11\*100 |    4    |

## 模型结构

输入实体n\*nfeat，输入关系n\*nrel：

+ 实体GAT拼接：第一层结果n\*nhid，将nhead个n\*nhid拼接为n*(nhid\*nhead)，第二层结果n\*(nhid\*nhead)，全连接层结果n\*nclass。

+ 实体GAT平均：第一层结果n\*nhid，将nhead个n\*nhid平均为n\*nhid，第二层结果n\*nfeat，全连接层结果n\*nclass。

+ 关系GAT拼接：第一层结果n\*nfeat，将nhead个n\*nfeat拼接为n*(nfeat\*nhead)，第二层结果n\*(nfeat\*nhead)，全连接层结果n\*nclass。

+ 关系GAT平均：第一层结果n\*nfeat，将nhead个n\*nfeat平均为n\*nfeat，第二层结果n\*nfeat，全连接层结果n\*nclass。

## FB15K237对比实验

|        |实体FB15K-237拼接|实体FB15K-237平均|关系FB15K-237拼接|关系FB15K-237平均|
| :----: | :------------: | :------------: | :-------------: | :------------: |
|nhead 50|      0.3133    |                |      0.4985     |                |
|nhead 30|      0.3432    |                |      0.4837     |                |

+ nhidden实体FB15K-237为10，关系FB15K-237为100。

## WN18RR对比实验

|        | 实体WN18RR拼接  | 实体WN18RR平均 |  关系WN18RR拼接  |  关系WN18RR平均 |
| :----: | :------------: | :------------: | :-------------: | :------------: |
|nhead 10|                |                |     0.8573      |                |

+ nhidden实体WN18RR为10，关系WN18RR为100。

# pyGAT->ConvE数据处理

+ 代码为[ConvE](https://github.com/Niuyuhang03/ConvE)的`master_with_comment`分支，运行`sh preprocess.sh`，会将`train.txt` `valid.txt` `test.txt`变为`.json`文件，并从`pyGAT/GAT_result/`中复制结果到`ConvE/data/`。

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

## baseline结果

|              |     cora   |  FB15K-237 |   WN18RR   |
| :----------: | :--------: | :--------: | :---------:|
| RDF2VEC(nb)  |   0.4973   |   0.0191   |  0.6920  |
| RDF2VEC(svm) |   0.3021   |   0.1605   |  0.7817  |
|     R-GCN    | **0.7374** | **0.5382** |**0.9759**|
