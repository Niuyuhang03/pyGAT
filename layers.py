import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


np.set_printoptions(threshold=np.inf)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, use_cuda=True, residual=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.use_cuda = use_cuda
        self.residual = residual

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.seq_dropout = nn.Dropout(dropout)
        self.coefs_dropout = nn.Dropout(dropout)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # fb中input.shape = [14435, 100], adj.shape = [14435, 14435]

        seq = torch.transpose(input, 0, 1).unsqueeze(0)  # fb中seq.shape = [1, 100, 14435]
        seq_fts = self.seq_transformation(seq)  # Wh, fb中seq_fts.shape = [1, 10, 14435]

        f_1 = self.f_1(seq_fts)  # a1Wh1, fb中f_1.shape = [1, 1, 14435]
        f_2 = self.f_2(seq_fts)  # a2Wh2, fb中f_2.shape = [1, 1, 14435]
        logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)  # a(Wh1||Wh2), fb中logits.shape = [14435, 14435]
        coefs = F.softmax(self.leakyrelu(logits) + adj, dim=1)  # softmax(leakyrelu(a(Wh1||Wh2))), fb中coefs.shape = [14435, 14435]
        seq_fts = seq_fts.squeeze(0)
        seq_fts = torch.transpose(seq_fts, 0, 1)
        seq_fts = self.seq_dropout(seq_fts)  # fb中seq_fts.shape = [14435, 10]
        coefs = self.coefs_dropout(coefs)  # fb中coefs.shape = [14435, 14435]

        ret = torch.mm(coefs, seq_fts) + self.bias  # alphaWh, fb中ret.shape = [14435, 10]

        if self.residual:
            if seq.size()[-1] != ret.size()[-1]:
                ret += torch.transpose(self.proj_residual(seq).squeeze(0), 0, 1)
            else:
                ret += input

        if self.concat:
            return F.elu(ret)  # fb中F.elu(ret).shape = [14435, 10]
        else:
            return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer_rel(nn.Module):
    """
    GAT with relations. out_features has to be in_features to nfeat in GAT_rel
    """

    def __init__(self, in_rels, out_features, dropout, alpha, concat=True, use_cuda=True):
        super(GraphAttentionLayer_rel, self).__init__()
        self.in_rels = in_rels
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.use_cuda = use_cuda

        self.seq_transformation_rel = nn.Conv1d(in_rels, 1, kernel_size=1, stride=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.coefs_dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, input, rel, rel_dict, adj):
        # fb中input.shape = [14435, 100], rel.shape = [237, 100], adj.shape = [14435, 14435]

        seq_rel = torch.transpose(rel, 0, 1).unsqueeze(0)  # fb中seq_rel.shape = [1, 100, 237]
        seq_fts_rel = self.seq_transformation_rel(seq_rel)  # fb中seq_fts_rel.shape = [1, 1, 237]

        # 根据rel构造权重coefs
        logits = torch.zeros_like(adj)
        for e1e2, r in rel_dict.items():
            e1, e2 = e1e2.split('+')
            e1, e2 = int(e1), int(e2)
            logits[e2][e1] = logits[e1][e2] = float(seq_fts_rel[0, 0, list(r)].max())  # 取所有e1和e2之间的r的Conv1d后的最大值
        # logits = torch.FloatTensor(logits)
        if self.use_cuda:
            logits = logits.cuda()
        coefs = F.softmax(self.relu(logits) + adj, dim=1)
        coefs = self.coefs_dropout(coefs)  # fb中coefs.shape = [14435, 14435]

        ret = torch.mm(coefs, input) + self.bias  # fb中ret.shape = [14435, 100]

        if self.concat:
            return F.elu(ret)  # fb中F.elu(ret).shape = [14435, 100]
        else:
            return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
