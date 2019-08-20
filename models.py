import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphAttentionLayer, GraphAttentionLayer_rel
import numpy as np


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, dataset, experiment, use_cuda):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.dataset = dataset
        self.experiment = experiment
        self.use_cuda = use_cuda
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, use_cuda=use_cuda) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False, use_cuda=use_cuda)
        self.linear_att = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, adj, names = None, print_flag=False):
        # 学习K个不同的attention，对应参数aij^k，W^k，然后在生成节点i的新特征时拼接起来：
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        # 在整个图神经网络的最后一层，使用平均替代拼接，得到节点最终的embedding
        x = self.out_att(x, adj)
        if print_flag:
            with open("./{}/GAT_{}_output.txt".format(self.experiment, self.dataset), "w") as output_f:
                x_array = np.array(x.detach())
                for idx in range(len(x_array)):
                    line = names[idx].split('\t')
                    output_f.write(str(line[0]))
                    for i in x_array[idx]:
                        output_f.write('\t' + str(i))
                    output_f.write('\n')
        # 增加一个全连接层
        x = self.linear_att(x)
        return F.log_softmax(x, dim=1)


class GAT_rel(nn.Module):
    def __init__(self, nrel, nfeat, nclass, dropout, alpha, nheads, dataset, experiment, use_cuda, use_mean):
        super(GAT_rel, self).__init__()
        self.dropout = dropout
        self.dataset = dataset
        self.experiment = experiment
        self.use_cuda = use_cuda
        self.use_mean = use_mean
        self.attentions = [GraphAttentionLayer_rel(nrel, nfeat, dropout=dropout, alpha=alpha, concat=True, use_cuda=use_cuda) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        if not use_mean:
            self.out_att = GraphAttentionLayer_rel(nrel, nfeat * nheads, dropout=dropout, alpha=alpha, concat=False, use_cuda=use_cuda)
            self.linear_att = nn.Linear(nfeat * nheads, nclass)
        else:
            self.out_att = GraphAttentionLayer_rel(nrel, nfeat, dropout=dropout, alpha=alpha, concat=False, use_cuda=use_cuda)
            self.linear_att = nn.Linear(nfeat, nclass)

    def forward(self, x, rel, rel_dict, adj, names = None, print_flag=False):
        # 学习K个不同的attention，对应参数aij^k，W^k，然后在生成节点i的新特征时拼接起来
        if not self.use_mean:
            x = torch.cat([att(x, rel, rel_dict, adj) for att in self.attentions], dim=1)
        else:
            x = torch.mean([att(x, rel, rel_dict, adj) for att in self.attentions], dim=1, keepdim=True)

        # 在整个图神经网络的最后一层，使用平均替代拼接，得到节点最终的embedding
        x = self.out_att(x, rel, rel_dict, adj)
        if print_flag:
            with open("./{}/GAT_{}_output.txt".format(self.experiment, self.dataset), "w") as output_f:
                x_array = np.array(x.detach())
                for idx in range(len(x_array)):
                    line = names[idx].split('\t')
                    output_f.write(str(line[0]))
                    for i in x_array[idx]:
                        output_f.write('\t' + str(i))
                    output_f.write('\n')
        # 增加一个全连接层
        x = self.linear_att(x)
        return F.log_softmax(x, dim=1)
