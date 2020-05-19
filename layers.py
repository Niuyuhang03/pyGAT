import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


np.set_printoptions(threshold=np.inf)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

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
        # input: N * in_feature, adj: N * N
        seq = torch.transpose(input, 0, 1).unsqueeze(0)  # 1 * in_feature * N
        seq_fts = self.seq_transformation(seq)  # 1 * out_feature * N

        f_1 = self.f_1(seq_fts)  # 1 * 1 * N
        f_2 = self.f_2(seq_fts)  # 1 * 1 * N
        logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)  # N * N
        coefs = F.softmax(self.leakyrelu(logits) + adj, dim=1)  # N * N
        coefs = coefs.cuda()
        coefs = self.coefs_dropout(coefs)

        seq_fts = seq_fts.squeeze(0)  # out_feature * N
        seq_fts = torch.transpose(seq_fts, 0, 1)  # N * out_feature
        seq_fts = self.seq_dropout(seq_fts)

        h_prime = torch.mm(coefs, seq_fts) + self.bias  # N * out_feature

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer_rel(nn.Module):
    """
    GAT with relations. out_features has to be in_features to nfeat in GAT_rel
    """

    def __init__(self, in_features, out_features, nrel, dropout, alpha, concat=True):
        super(GraphAttentionLayer_rel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nrel = nrel
        self.alpha = alpha
        self.concat = concat

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_rel = nn.Conv1d(nrel, 1, kernel_size=1, stride=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.seq_dropout = nn.Dropout(dropout)
        self.coefs_dropout = nn.Dropout(dropout)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, rel, rel_dict, adj):
        # input: N * in_feature, adj: N * N
        seq = torch.transpose(input, 0, 1).unsqueeze(0)  # 1 * in_feature * N
        seq_fts = self.seq_transformation(seq)  # Wh, 1 * out_feature * N

        seq_rel = torch.transpose(rel, 0, 1).unsqueeze(0)  # 1 * in_feature * M
        seq_fts_rel = self.seq_transformation_rel(seq_rel)  # 1 * 1 * M

        # 根据rel构造权重coefs
        logits = torch.zeros_like(adj)  # N * N
        for e1e2, r in rel_dict.items():
            e1, e2 = e1e2.split('+')
            e1, e2 = int(e1), int(e2)
            logits[e2][e1] = logits[e1][e2] = float(seq_fts_rel[0, 0, list(r)].max())  # 取所有e1和e2之间的r的Conv1d后的最大值
            logits = logits.cuda()
        coefs = F.softmax(self.leakyrelu(logits) + adj, dim=1)
        coefs = coefs.cuda()
        coefs = self.coefs_dropout(coefs)  # N * N

        seq_fts = seq_fts.squeeze(0)  # out_feature * N
        seq_fts = torch.transpose(seq_fts, 0, 1)  # N * out_feature
        seq_fts = self.seq_dropout(seq_fts)

        h_prime = torch.mm(coefs, seq_fts) + self.bias  # N * out_feature

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RWRLayer(nn.Module):
    """
    Random Walker Rstart layer
    """

    def __init__(self, in_features, out_features, dropout, alpha, dataset_str, concat=True):
        super(RWRLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dataset_str = dataset_str

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.seq_dropout = nn.Dropout(dropout)
        self.coefs_dropout = nn.Dropout(dropout)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, adj_ad):
        # input: N * in_feature, adj: N * N
        seq = torch.transpose(input, 0, 1).unsqueeze(0)  # 1 * in_feature * N
        seq_fts = self.seq_transformation(seq)  # 1 * out_feature * N

        f_1 = self.f_1(seq_fts)  # 1 * 1 * N
        f_2 = self.f_2(seq_fts)  # 1 * 1 * N
        logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)  # N * N
        coefs = F.softmax(self.leakyrelu(logits) + adj, dim=1)  # N * N
        coefs = coefs.cuda()
        coefs = self.coefs_dropout(coefs)

        s = adj_ad  # N * N

        ri_all = []
        ri_index = []
        # You may replace adj.shape[0] with the size of dataset
        for i in range(adj.shape[0]):
            # You may replace 1,4 with the .n-hop neighbors you want
            index_i = torch.nonzero((s[i] < 4) & (s[i] > 1), as_tuple=True)  # replace torch.where(condition)
            I = torch.eye((len(index_i[0]) + 1)).cuda()
            ei = torch.FloatTensor([[0] for _ in range(len(index_i[0]) + 1)]).cuda()
            ei[0] = torch.FloatTensor([1])

            W = torch.FloatTensor([[0 for i in range((len(index_i[0])) + 1)] for j in range((len(index_i[0])) + 1)]).cuda()
            W[0, 1:] = 1
            W[1:, 0] = 1

            # the choice of the c parameter in RWR
            c = 0.5
            rw_left = (I - c * W)
            try:
                rw_left = torch.inverse(rw_left)  # 求逆
            except:
                rw_left = rw_left

            ri = torch.mm(rw_left, ei)
            ri = torch.transpose(ri, 1, 0)
            ri = abs(ri[0]).cpu().numpy().tolist()
            ri_index.append(index_i[0].cpu().numpy())
            ri_all.append(ri)

        fw = open('data/{}/ri_index_c_0.5_{}_highorder_1_x_abs.pkl'.format(self.dataset_str, self.dataset_str), 'wb')
        pickle.dump(ri_index, fw)
        fw.close()

        fw = open('data/{}/ri_all_c_0.5_{}_highorder_1_x_abs.pkl'.format(self.dataset_str, self.dataset_str), 'wb')
        pickle.dump(ri_all, fw)
        fw.close()

        seq_fts = seq_fts.squeeze(0)  # out_feature * N
        seq_fts = torch.transpose(seq_fts, 0, 1)  # N * out_feature
        seq_fts = self.seq_dropout(seq_fts)

        h_prime = torch.mm(coefs, seq_fts) + self.bias  # N * out_feature

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class StructuralFingerprintLayer(nn.Module):
    """
    adaptive structural fingerprint layer
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(StructuralFingerprintLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.seq_dropout = nn.Dropout(dropout)
        self.coefs_dropout = nn.Dropout(dropout)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))  # 1 * 1
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))  # 1 * 1
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)

    def forward(self, input, adj, adj_ad):
        # input: N * in_feature, adj: N * N
        seq = torch.transpose(input, 0, 1).unsqueeze(0)  # 1 * in_feature * N
        seq_fts = self.seq_transformation(seq)  # 1 * out_feature * N

        f_1 = self.f_1(seq_fts)  # 1 * 1 * N
        f_2 = self.f_2(seq_fts)  # 1 * 1 * N
        logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)  # N * N
        e = F.softmax(self.leakyrelu(logits) + adj, dim=1)  # N * N
        e = e.cuda()

        s = adj_ad  # N * N

        coefs = F.softmax(abs(self.W_ei) * e + abs(self.W_si) * s, dim=1)  # N * N
        coefs = coefs.cuda()
        coefs = self.coefs_dropout(coefs)

        seq_fts = seq_fts.squeeze(0)  # out_feature * N
        seq_fts = torch.transpose(seq_fts, 0, 1)  # N * out_feature
        seq_fts = self.seq_dropout(seq_fts)

        h_prime = torch.mm(coefs, seq_fts) + self.bias  # N * out_feature

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer_all(nn.Module):
    def __init__(self, in_features, out_features, nrel, dropout, alpha, concat=True):
        super(GraphAttentionLayer_all, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.nrel = nrel
        self.alpha = alpha
        self.concat = concat

        self.seq_transformation_rel = nn.Conv1d(nrel, 1, kernel_size=1, stride=1, bias=False)
        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.seq_dropout = nn.Dropout(dropout)
        self.coefs_dropout = nn.Dropout(dropout)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))  # 1 * 1
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))  # 1 * 1
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)
        self.W_ri = nn.Parameter(torch.zeros(size=(1, 1)))  # 1 * 1
        nn.init.xavier_uniform_(self.W_ri.data, gain=1.414)

    def forward(self, input, rel, rel_dict, adj, adj_ad):
        # input: N * in_feature, adj: N * N
        seq = torch.transpose(input, 0, 1).unsqueeze(0)  # 1 * in_feature * N
        seq_fts = self.seq_transformation(seq)  # Wh, 1 * out_feature * N

        seq_rel = torch.transpose(rel, 0, 1).unsqueeze(0)  # 1 * in_feature * M
        seq_fts_rel = self.seq_transformation_rel(seq_rel)  # 1 * 1 * M

        # 根据rel构造权重coefs
        logits_r = torch.zeros_like(adj)  # N * N
        for e1e2, r in rel_dict.items():
            e1, e2 = e1e2.split('+')
            e1, e2 = int(e1), int(e2)
            logits_r[e2][e1] = logits_r[e1][e2] = float(seq_fts_rel[0, 0, list(r)].max())  # 取所有e1和e2之间的r的Conv1d后的最大值
            logits_r = logits_r.cuda()
        r = F.softmax(self.leakyrelu(logits_r) + adj, dim=1)  # N * N

        f_1 = self.f_1(seq_fts)  # 1 * 1 * N
        f_2 = self.f_2(seq_fts)  # 1 * 1 * N
        logits_e = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)  # N * N
        e = F.softmax(self.leakyrelu(logits_e) + adj, dim=1)  # N * N

        s = adj_ad.cuda()  # N * N

        coefs = F.softmax(abs(self.W_ei) * e + abs(self.W_ri) * r + abs(self.W_si) * s, dim=1)  # N * N
        coefs = coefs.cuda()
        coefs = self.coefs_dropout(coefs)

        seq_fts = seq_fts.squeeze(0)  # out_feature * N
        seq_fts = torch.transpose(seq_fts, 0, 1)  # N * out_feature
        seq_fts = self.seq_dropout(seq_fts)

        h_prime = torch.mm(coefs, seq_fts) + self.bias  # N * out_feature

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
