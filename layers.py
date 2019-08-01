import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


np.set_printoptions(threshold=np.inf)


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
        # fb中input.shape = [14435, 100], adj.shape = [14435, 14435]

        seq = torch.transpose(input, 0, 1).unsqueeze(0)  # fb中seq.shape = [1, 100, 14435]
        seq_fts = self.seq_transformation(seq)  # Wh, fb中seq_fts.shape = [1, 10, 14435]

        f_1 = self.f_1(seq_fts)  # a1Wh1, fb中f_1.shape = [1, 1, 14435]
        f_2 = self.f_2(seq_fts)  # a2Wh2, fb中f_2.shape = [1, 1, 14435]
        logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)  # a(Wh1||Wh2), fb中logits.shape = [14435, 14435]
        coefs = F.softmax(self.leakyrelu(logits) + adj, dim=1)  # softmax(leakyrelu(a(Wh1||Wh2))), fb中coefs.shape = [14435, 14435]

        seq_fts = F.dropout(torch.transpose(seq_fts.squeeze(0), 0, 1), self.dropout, training=self.training)  # fb中seq_fts.shape = [14435, 10]
        coefs = F.dropout(coefs, self.dropout, training=self.training)  # fb中coefs.shape = [14435, 14435]

        ret = torch.mm(coefs, seq_fts) + self.bias # alphaWh, fb中ret.shape = [14435, 10]

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

    def __init__(self, in_rels, out_rels, in_out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer_rel, self).__init__()
        self.dropout = dropout
        self.in_rels = in_rels
        self.out_rels = out_rels
        self.out_features = in_out_features
        self.alpha = alpha
        self.concat = concat

        self.seq_transformation_rel = nn.Conv1d(in_rels, out_rels, kernel_size=1, stride=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(in_out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.ReLU = nn.ReLU()

    def forward(self, input, rel, rel_dict, adj):
        # fb中input.shape = [14435, 100], rel.shape = [237, 100], adj.shape = [14435, 14435]

        seq_rel = torch.transpose(rel, 0, 1).unsqueeze(0)  # fb中seq_rel.shape = [1, 100, 237]
        seq_fts_rel = self.seq_transformation_rel(seq_rel)  # fb中seq_fts_rel.shape = [1, 1, 237]

        logits = torch.zeros_like(adj).float()
        for e1, e2r in rel_dict.items():
            for e2, r in e2r.items():
                mean_value = seq_fts_rel[0, 0, r].max()
                logits[int(e1)][int(e2)] = mean_value
        coefs = F.softmax(self.ReLU(logits) + adj, dim=1)
        coefs = F.dropout(coefs, self.dropout, training=self.training)  # fb中coefs.shape = [14435, 14435]

        ret = torch.mm(coefs, input) + self.bias  # fb中ret.shape = [14435, 100]

        if self.concat:
            print("F.elu(ret)")
            return F.elu(ret)  # fb中F.elu(ret).shape = [14435, 100]
        else:
            print("ret")
            return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
