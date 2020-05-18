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

    def __init__(self, nrel, inout_features, dropout, alpha, concat=True, use_cuda=True):
        super(GraphAttentionLayer_rel, self).__init__()
        self.inout_features = inout_features
        self.alpha = alpha
        self.concat = concat
        self.use_cuda = use_cuda

        self.seq_transformation_rel = nn.Conv1d(nrel, 1, kernel_size=1, stride=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(inout_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

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
        return self.__class__.__name__ + ' (' + str(self.inout_features) + ' -> ' + str(self.inout_features) + ')'


class StructuralFingerprintLayer(nn.Module):
    """
    adaptive structural fingerprint layer
    """

    def __init__(self, in_features, out_features, dropout, alpha, adj_ad, adj, concat=True):
        super(StructuralFingerprintLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj_ad = adj_ad
        self.adj = adj
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 均匀分布
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)

    def forward(self, input):
        h = torch.mm(input, self.W)  # h * w
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # leakyrelu(h * w * a)，即leakyrelu的eij
        s = self.adj_ad  # sij
        adj = self.adj
        e = e.cuda()
        s = s.cuda()
        adj = adj.cuda()  # adj为图连通性

        # combine sij and eij
        e = abs(self.W_ei) * e + abs(self.W_si) * s  # aij=前半部分+后半部分（均未softmax）

        zero_vec = -9e15 * torch.ones_like(e)
        # k_vec = -9e15 * torch.ones_like(e)

        np.set_printoptions(threshold=np.inf)
        attention = torch.where(adj > 0, e, zero_vec)  # 第一个参数是条件，第二个参数是满足时的值，第三个参数时不满足时的值
        attention = F.softmax(attention, dim=1)  # alpha
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # h=alpha * W * h
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

    def __init__(self, in_features, out_features, dropout, alpha, adj_ad, adj, dataset_str, concat=True):
        super(RWRLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj_ad = adj_ad
        self.adj = adj
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dataset_str = dataset_str

    def forward(self, input):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        s = self.adj_ad
        adj = self.adj
        e = e.cuda()
        s = s.cuda()
        adj = adj.cuda()

        # Dijkstra = s.numpy()
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

        zero_vec = -9e15 * torch.ones_like(e)
        # k_vec = -9e15*torch.ones_like(e)
        # adj = adj.cuda()
        # np.set_printoptions(threshold=np.inf)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
