import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, residual=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.residual = residual

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Too harsh to use the same dropout. TODO add another dropout
        # input = F.dropout(input, self.dropout, training=self.training)

        seq = torch.transpose(input, 0, 1).unsqueeze(0)
        seq_fts = self.seq_transformation(seq) # Wh

        f_1 = self.f_1(seq_fts) # a1Wh1
        f_2 = self.f_2(seq_fts) # a2Wh2
        logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0) # a(Wh1||Wh2)
        coefs = F.softmax(self.leakyrelu(logits) + adj, dim=1) # softmax(leakyrelu(a(Wh1||Wh2)))

        seq_fts = F.dropout(torch.transpose(seq_fts.squeeze(0), 0, 1), self.dropout, training=self.training)
        coefs = F.dropout(coefs, self.dropout, training=self.training)

        ret = torch.mm(coefs, seq_fts) + self.bias # alphaWh

        if self.residual:
            if seq.size()[-1] != ret.size()[-1]:
                ret += torch.transpose(self.proj_residual(seq).squeeze(0), 0, 1)
            else:
                ret += input

        if self.concat:
            return F.elu(ret)
        else:
            return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttentionLayer_rel(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, residual=False):
        super(GraphAttentionLayer_rel, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.residual = residual

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_rel = nn.Conv1d(in_features, 1, kernel_size=1, stride=1, bias=False)
        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, rel, rel_dict, adj):
        # Too harsh to use the same dropout. TODO add another dropout
        # input = F.dropout(input, self.dropout, training=self.training)

        seq = torch.transpose(input, 0, 1).unsqueeze(0)
        seq_fts = self.seq_transformation(seq) # Wh

        seq_rel = torch.transpose(rel, 0, 1).unsqueeze(0)
        seq_fts_rel = self.seq_transformation_rel(seq_rel) # rel m*1
        print("seq_fts_rel:", seq_fts_rel.shape)

        logits = torch.zeros_like(adj).float()
        for key, value_index in rel_dict.items():
            e1, e2 = key.split('+')
            mean_value = seq_fts_rel[0, 0, value_index].mean()
            logits[int(e1)][int(e2)] = mean_value
            logits[int(e2)][int(e1)] = mean_value
        print("logits:", logits)
        coefs = F.softmax(self.sigmoid(logits) + adj, dim=1)

        seq_fts = F.dropout(torch.transpose(seq_fts.squeeze(0), 0, 1), self.dropout, training=self.training)
        coefs = F.dropout(coefs, self.dropout, training=self.training)

        ret = torch.mm(coefs, seq_fts) + self.bias

        if self.residual:
            if seq.size()[-1] != ret.size()[-1]:
                ret += torch.transpose(self.proj_residual(seq).squeeze(0), 0, 1)
            else:
                ret += input

        if self.concat:
            return F.elu(ret)
        else:
            return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
