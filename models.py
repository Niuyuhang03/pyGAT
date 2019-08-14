import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphAttentionLayer, GraphAttentionLayer_rel
import numpy as np


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, dataset, experiment):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.dataset = dataset
        self.experiment = experiment
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False)
        self.linear_att = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, adj, print_flag=False):
        # 学习K个不同的attention，对应参数aij^k，W^k，然后在生成节点i的新特征时拼接起来：
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        # 在整个图神经网络的最后一层，使用平均替代拼接，得到节点最终的embedding
        x = self.out_att(x, adj)
        if print_flag:
            with open("./{}/{}_output.txt".format(self.experiment, self.dataset), "w") as output_f:
                with open("./data/{}/{}.content".format(self.dataset, self.dataset), "r") as input_f:
                    input_content = input_f.readlines()
                    x_array = np.array(x.detach())
                    for idx in range(len(input_content)):
                        line = input_content[idx].split('\t')
                        output_f.write(str(line[0]) + '\t')
                        for i in x_array[idx]:
                            output_f.write(str(i) + '\t')
                        output_f.write(str(line[-1]))
        # 增加一个全连接层
        x = self.linear_att(x)
        return F.log_softmax(x, dim=1)


class GAT_rel(nn.Module):
    def __init__(self, nrel, nhid, nclass, dropout, alpha, nheads, dataset, experiment):
        super(GAT_rel, self).__init__()
        self.dropout = dropout
        self.dataset = dataset
        self.experiment = experiment
        self.attentions = [GraphAttentionLayer_rel(nrel, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer_rel(nrel, nhid * nheads, dropout=dropout, alpha=alpha, concat=False)
        self.linear_att = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, rel, rel_dict, adj, print_flag=False):
        # 学习K个不同的attention，对应参数aij^k，W^k，然后在生成节点i的新特征时拼接起来
        x = torch.cat([att(x, rel, rel_dict, adj) for att in self.attentions], dim=1)

        # 在整个图神经网络的最后一层，使用平均替代拼接，得到节点最终的embedding
        x = self.out_att(x, rel, rel_dict, adj)
        if print_flag:
            with open("./{}/GAT_{}_output.txt".format(self.experiment, self.dataset), "w") as output_f:
                with open("./data/{}/{}.content".format(self.dataset, self.dataset), "r") as input_f:
                    input_content = input_f.readlines()
                    x_array = np.array(x.detach())
                    for idx in range(len(input_content)):
                        line = input_content[idx].split('\t')
                        output_f.write(str(line[0]) + '\t')
                        for i in x_array[idx]:
                            output_f.write(str(i) + '\t')
                        output_f.write(str(line[-1]))
        # 增加一个全连接层
        x = self.linear_att(x)
        return F.log_softmax(x, dim=1)
