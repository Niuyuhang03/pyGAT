import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False)
        self.linear_att = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, adj):
        # 学习K个不同的attention，对应参数aij^k，W^k，然后在生成节点i的新特征时拼接起来：
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        # 在整个图神经网络的最后一层，使用平均替代拼接，得到节点最终的embedding
        x = self.out_att(x, adj)
        # 增加一个全连接层
        x = self.linear_att(x)
        return F.log_softmax(x, dim=1)
        # sigmoid_fn = nn.Sigmoid()
        # return sigmoid_fn(x)
