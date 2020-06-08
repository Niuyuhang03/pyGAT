# 代码

## 仓库说明

+ [DKRL](https://github.com/Niuyuhang03/DKRL)：提供FB15k-237的实体描述信息和初步分类结果
+ [OpenKE](https://github.com/Niuyuhang03/OpenKE)：提供FB15k-237数据集和WN18RR数据集，TransE模型
+ [pyGAT](https://github.com/Niuyuhang03/pyGAT)：提供cora数据集，GAT模型
+ [ConvE](https://github.com/Niuyuhang03/ConvE)：提供ConvE、DistMult、ComplEx模型
+ [RDF2VEC_MultiLabel](https://github.com/Niuyuhang03/RDF2VEC_MultiLabel)：提供RDF2VEC模型
+ [ADaptive-Structural-Fingerprint](https://github.com/Niuyuhang03/ADaptive-Structural-Fingerprint)：提供ADSF模型（已融合到pyGAT中，本仓库已废弃）
+ [rgcn_pytorch_implementation](https://github.com/KarCute/rgcn_pytorch_implementation)：提供R-GCN模型
+ [gcn](https://github.com/tkipf/gcn)：提供citeseer数据

## 注意事项

+ 提交前注意分支，模型用到的分支均已设置为默认分支，但不是master分支
+ RDF2VEC模型没有embeddings，只用到了nb和svm核对已经embeddings好的数据进行评测
+ 数据集FB15k-237和WN18RR出现内存问题，最终使用的数据集是我们创建的FB15k-237_4000和WN18RR_4000
+ 提交文件使用GPU集群的slurm系统，在管理节点上以`sbatch xxx.slurm`方式提交，不能直接运行

## [OpenKE](https://github.com/Niuyuhang03/OpenKE)

+ 安装依赖：`import nltk; nltk.download('wordnet')`
+ [train_FB15K237.py](https://github.com/Niuyuhang03/OpenKE/train_FB15K237.py)：在FB15k-237上训练TransE，输出文件在[FB15K237_result文件夹的TransE.json](https://github.com/Niuyuhang03/OpenKE/tree/GAT_data_process/FB15K237_result/TransE.json)
+ [train_WN18RR.py](https://github.com/Niuyuhang03/OpenKE/blob/GAT_data_process/train_WN18RR.py)：在WN18RR上训练TransE，输出文件在[WN18RR_result文件夹的TransE.json](https://github.com/Niuyuhang03/OpenKE/tree/GAT_data_process/WN18RR_result/TransE.json)
+ [FB15K237_process.py](https://github.com/Niuyuhang03/OpenKE/blob/GAT_data_process/FB15K237_result/FB15K237_process.py)：给FB15k-237实体标注，输出文件在[FB15K237_result文件夹](https://github.com/Niuyuhang03/OpenKE/tree/GAT_data_process/FB15K237_result)
+ [WN18RR_process.py](https://github.com/Niuyuhang03/OpenKE/blob/GAT_data_process/WN18RR_result/WN18RR_process.py)：给WN18RR实体标注，输出文件在[WN18RR_result文件夹](https://github.com/Niuyuhang03/OpenKE/tree/GAT_data_process/WN18RR_result)
+ [FB15K237_4000_process.py](https://github.com/Niuyuhang03/OpenKE/blob/GAT_data_process/FB15K237_4000_result/FB15K237_4000_process.py)：将FB15k-237划分为FB15k-237_4000，输出文件在[FB15K237_4000_result文件夹](https://github.com/Niuyuhang03/OpenKE/tree/GAT_data_process/FB15K237_4000_result)
+ [WN18RR_4000_process.py](https://github.com/Niuyuhang03/OpenKE/blob/GAT_data_process/WN18RR_4000_result/WN18RR_4000_process.py)：将WN18RR划分为WN18RR_4000，输出文件在[WN18RR_4000_result文件夹](https://github.com/Niuyuhang03/OpenKE/tree/GAT_data_process/WN18RR_4000_result)

+ 输出实体和关系向量需要手动复制到[pyGAT](https://github.com/Niuyuhang03/pyGAT)仓库

## [pyGAT](https://github.com/Niuyuhang03/pyGAT)

+ [train.py](https://github.com/Niuyuhang03/pyGAT/blob/similar_impl_tensorflow_with_comment/train.py)：模型入口
+ [utils.py](https://github.com/Niuyuhang03/pyGAT/blob/similar_impl_tensorflow_with_comment/utils.py)：数据处理
+ [models.py](https://github.com/Niuyuhang03/pyGAT/blob/similar_impl_tensorflow_with_comment/models.py)：模型定义
+ [layers.py](https://github.com/Niuyuhang03/pyGAT/blob/similar_impl_tensorflow_with_comment/layers.py)：层定义

+ 输出实体向量需要执行`sh process.sh`复制到[ConvE](https://github.com/Niuyuhang03/ConvE)仓库，复制前需要修改[preprocess.sh](https://github.com/Niuyuhang03/ConvE/blob/master_with_comment/preprocess.sh)文件中的路径，路径以job在slurm的提交时间命名

## [ConvE](https://github.com/Niuyuhang03/ConvE)

+ 安装依赖：`pip install -r requirements.txt;python -m spacy download en;sh preprocess.sh`
+ [create_FB15K237_4000.py](https://github.com/Niuyuhang03/ConvE/blob/master_with_comment/create_FB15K237_4000.py)：生成FB15K237_4000数据格式
+ [create_WN18RR_4000.py](https://github.com/Niuyuhang03/ConvE/blob/master_with_comment/create_WN18RR_4000.py)：生成WN18RR_4000数据格式
+ [main.py](https://github.com/Niuyuhang03/ConvE/blob/master_with_comment/main.py)：模型入口
+ [model.py](https://github.com/Niuyuhang03/ConvE/blob/master_with_comment/model.py)：模型和层定义
+ [evaluation.py](https://github.com/Niuyuhang03/ConvE/blob/master_with_comment/evaluation.py)：链接预测评估

## 数据集

| 数据集       | Cora | Citeseer | FB15k-237_4000 | WN18RR_4000 |
| ------------ | ---- | -------- | -------------- | ----------- |
| 实体数       | 2708 | 3327     | 4457           | 3846        |
| 关系数       | -    | -        | 110            | 10          |
| 三元组数     | 5429 | 4732     | 27232          | 6439        |
| 特征数       | 1433 | 3703     | 100            | 100         |
| 实体类别数   | 7    | 6        | 25             | 4           |
| 训练集实体数 | 140  | 120      | 3565           | 3076        |
| 验证集实体数 | 300  | 500      | 446            | 385         |
| 测试集实体数 | 1000 | 1000     | 446            | 385         |

## 实验结果

+ 实体分类

|              | Cora       | Citeseer   | FB15k-237_4000 | WN18RR_4000 |
| ------------ | ---------- | ---------- | -------------- | ----------- |
| RDF2VEC(NB)  | 0.5026     | 0.6116     | 0.0121         | 0.7779      |
| RDF2VEC(SVM) | 0.3020     | 0.2107     | 0.1711         | 0.7548      |
| R-GCN        | -          | -          | 0.4308         | 0.9105      |
| GAT          | 0.8211     | 0.6730     | 0.4642         | **0.9130**  |
| GAT_rel      | -          | -          | 0.5117         | 0.8977      |
| GAT_adsf     | **0.8460** | **0.7050** | **0.5217**     | 0.9003      |
| GAT_all      | -          | -          | 0.5085         | 0.8977      |

+ 链接预测
    + ConvE在FB15k-237_4000上

    |          | MR        | MRR        | Hits@1     | Hits@10    |
    | -------- | --------- | ---------- | ---------- | ---------- |
    | GAT      | 961.3     | 0.1537     | 0.1263     | 0.1891     |
    | GAT_rel  | 806.5     | 0.1662     | 0.1266     | 0.2194     |
    | GAT_adsf | **681.0** | **0.2150** | **0.1484** | **0.3277** |
    | GAT_all  | 898.5     | 0.1568     | 0.1263     | 0.2008     |

    + ComplEx在FB15k-237_4000上

    |          | MR        | MRR        | Hits@1     | Hits@10    |
    | -------- | --------- | ---------- | ---------- | ---------- |
    | GAT      | 1543.1    | 0.0658     | 0.0478     | 0.1181     |
    | GAT_rel  | 2264.8    | 0.0645     | 0.0214     | 0.1165     |
    | GAT_adsf | **908.1** | **0.0845** | 0.0091     | **0.1917** |
    | GAT_all  | 1799.6    | 0.1014     | **0.0768** | 0.1250     |

    + DistMult在FB15k-237_4000上

    |          | MR         | MRR        | Hits@1     | Hits@10    |
    | -------- | ---------- | ---------- | ---------- | ---------- |
    | GAT      | 1603.6     | 0.0649     | 0.0423     | 0.1425     |
    | GAT_rel  | 2361.3     | 0.1075     | **0.1914** | 0.1165     |
    | GAT_adsf | **1038.9** | 0.0774     | 0.0276     | **0.1764** |
    | GAT_all  | 2051.7     | **0.1117** | 0.1894     | 0.1223     |

+ 随机初始化

    + FB15k-237_4000实体分类

    |          | 随机初始化 | TransE初始化 |
    | -------- | ---------- | ------------ |
    | GAT      | 0.3850     | **0.4642**   |
    | GAT_adsf | 0.5158     | **0.5217**   |

