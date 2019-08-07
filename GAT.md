## 数据集和数据处理

|  数据集   |实体/关系|dataset.content|dataset.cites|dataset.rel| 数据处理时间 | 运行1个epoch时间 |
| :-------: | :---: | :--------: | :----: | :------: | :---: | :----: |
|   cora    |  实体 | 2708\*1433 | 5429   |    -     |   9s  |  0.05s |
| FB15K-237 |  实体 | 14435\*100 | 298927 |    -     |   13s | 2min40s|
| FB15K-237 |  关系 | 14435\*100 | 298927 | 237\*100 |   15s | 3min20s|
|   WN18RR  |  实体 | 40943\*100 | 93003  |    -     |   44s |        |
|   WN18RR  |  关系 | 40943\*100 | 93003  |  11\*100 |   46s |15min30s|

## 实验结果

### GAT

+ epochs：all：运行至收敛。100：运行100个epochs后提前结束。

|   job   |   数据集  |实体/关系|  nhid | nheads |  epochs  | 运行时间 |  准确率 |
| :-----: | :-------: | :----: | :---: | :----: | :------: | :------: | :----: |
|   co    |    cora   |  实体  |   10  |   10   | all(265) |   30s    | 0.8200 |
| FB_100  | FB15K-237 |  实体  |   10  |   10   |   100    | 4h20min  | 0.2774 |
|   FB    | FB15K-237 |  实体  |   10  |   10   | all(642) | 27h20min | 0.2876 |
| FB_r100 | FB15K-237 |  关系  |  100  |   10   |   100    | 5h40min  | 0.4566 |
|  FB_r   | FB15K-237 |  关系  |  100  |   10   | all(223) | 12h40min | 0.4574 |
| WN_100  |   WN18RR  |  实体  |   10  |   10   |   100    | out of memory |        |
|   WN    |   WN18RR  |  实体  |   10  |   10   |   all    | out of memory |        |
| WN_r100 |   WN18RR  |  关系  |  100  |   10   |   100    | 26h20min | 0.8597 |
|  WN_r   |   WN18RR  |  关系  |  100  |   10   | all(168) | 43h40min | 0.8592 |

### baseline

| baseline模型 |  数据集   |         acc          |
| :----------: | :-------: | :------------------: |
| RDF2VEC(nb)  |   cora    |  0.4877546229932895  |
| RDF2VEC(multilabel-nb)  | FB15K-237 | 0.016072001044300215 |
| RDF2VEC(multilabel-nb)  |  WN18RR   |  0.6919378726901931  |
| RDF2VEC(svm) |   cora    | 0.30207475441104414  |
| RDF2VEC(multilabel-svm) | FB15K-237 | 0.15961101736725006  |
| RDF2VEC(multilabel-svm) |  WN18RR   |  0.7817455187704332  |
|      WL      |   cora    |                      |
|      WL      | FB15K-237 |                      |
|      WL      |  WN18RR   |                      |
|     FEAT     |   cora    |                      |
|     FEAT     | FB15K-237 |                      |
|     FEAT     |  WN18RR   |                      |
|    R-GCN     |   cora    |                      |
|    R-GCN     | FB15K-237 |                      |
|    R-GCN     |  WN18RR   |                      |

## 任务进度(生产实习手册用)

1. 配置环境pytorch+cuda，申请gpu集群账号；clone pyGAT，尝试cpu运行，时间较长；熟悉知识图谱背景
2. 熟悉pygat代码结构，熟悉项目目标
3. 熟悉gpu提交作业方式，运行cora数据集，符合论文结果
4. 下载fb15k237和wn18rr，数据标注，通过dkrl和wn进行
5. 运行transE，进行fb和wn的entities和relations的embeddings。通过OpenKE进行
6. 修改数据处理和accuracy，适应多标签问题
7. 修改loss为sigmoid+bceloss，适应多标签问题，运行cora验证，准确率下降为0.35，preds全部相同，不正常
8. 研究问题来源。对比bceloss和nllloss原理，发现问题为loss前的激活函数sigmoid限制到0.7，研究改进方法
9. 手动搭建适应多标签问题的nllloss，激活函数仍然采用log_softmax
10. 在cora上验证正确后，尝试在fb和wn上运行，出现内存问题
11. 问题为显存过大，申请4个gpu节点无效果，发现问题为超过单个节点显存限制，研究发现由于邻接矩阵的存在，数据集无法分割batch
12. 根据gat issue，存在一个内存更小的分支similar，更加贴合代码，但准确率不top，决定切换分支，迁移之前的修改
13. 仍然出现问题，研究后认为问题可能为torch和tf在gpu上出现冲突，尝试cuda_visable_devices=1，解决。出现数据集重复问题
14. 两分支对邻接矩阵adj构造方式不同，新分支导致数据集的重复被发现，修改数据集cites部分
15. 出现问题：preds有全0，把根据阈值变为01张量改为根据下标；预测准确率特别大：acc函数有问题，且01多标签问题应是正确的1在预测的1中的比例为准确率
16. cora正常。fb运行时间过长，砍为100epoches。两个分支增加全连接层，在cora和fb上验证
17. 在master分支未修改时运行fb报错内存爆炸，根据issue更改代码为@，得到解决。对比修改后的fb准确率，master为0.3，similar为0.5，决定使用similar。
18. 使用RDF2Vec作为baseline，对比fb和wn数据。需要修改数据处理部分，修改RDF2Vec的model为多标签，并修改数据集中空格为\t。得到rdf2vec准确率
19. 为修改实体gat为关系gat，熟悉pyGAT两分支实现layers的方式，修改数据cite和rel，数据处理部分增加对rel的处理。
20. 为修改实体gat为关系gat，处理对于relation的读入和layer中设置权重logits，修改rel_dict为单向关系。
21. 修改输出文件，同时运行多个程序。记录实验数据。修改sigmoid为relu。
22. 使用feat和Weisfeiler-Lehman kernels作为baseline。
23. 在rgcn上运行fb、wn
24. 解决内存问题