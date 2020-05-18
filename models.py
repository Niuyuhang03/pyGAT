import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphAttentionLayer, GraphAttentionLayer_rel, StructuralFingerprintLayer, RWRLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, dataset, experiment, use_cuda):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.dataset = dataset
        self.experiment = experiment
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, use_cuda=use_cuda) for _ in range(nheads)]  # nfeat -> nhid
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nfeat, dropout=dropout, alpha=alpha, concat=False, use_cuda=use_cuda)  # nhid * nheads -> nfeat
        self.linear_att = nn.Linear(nfeat, nclass)

    def forward(self, x, adj, names=None, print_flag=False):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        if print_flag:
            with open("./{}/GAT_{}_output.txt".format(self.experiment, self.dataset), "w") as output_f:
                x_array = x.cpu().detach().numpy()
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
    def __init__(self, nfeat, nclass, dropout, alpha, nheads, dataset, experiment, use_cuda):
        super(GAT_rel, self).__init__()
        self.dropout = dropout
        self.dataset = dataset
        self.experiment = experiment
        self.attentions = [GraphAttentionLayer_rel(nfeat, nfeat, dropout=dropout, alpha=alpha, concat=True, use_cuda=use_cuda) for _ in range(nheads)]  # nfeat -> nfeat
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer_rel(nfeat, nfeat * nheads, dropout=dropout, alpha=alpha, concat=False, use_cuda=use_cuda)  # nfeat * nheads -> nfeat * nheads
        self.linear_att1 = nn.Linear(nfeat * nheads, nfeat)
        self.linear_att2 = nn.Linear(nfeat, nclass)

    def forward(self, x, rel, rel_dict, adj, names=None, print_flag=False):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, rel, rel_dict, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, rel, rel_dict, adj))
        x = self.linear_att1(x)
        if print_flag:
            with open("./{}/GAT_{}_output.txt".format(self.experiment, self.dataset), "w") as output_f:
                x_array = x.cpu().detach().numpy()
                for idx in range(len(x_array)):
                    line = names[idx].split('\t')
                    output_f.write(str(line[0]))
                    for i in x_array[idx]:
                        output_f.write('\t' + str(i))
                    output_f.write('\n')
        # 增加一个全连接层
        x = self.linear_att2(x)
        return F.log_softmax(x, dim=1)


class ADSF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj_ad, adj):
        """version of ADSF."""
        super(ADSF, self).__init__()
        self.dropout = dropout
        self.attentions = [StructuralFingerprintLayer(nfeat, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad, adj=adj, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 按attention_i名使用layer，似乎未用到
        self.out_att = StructuralFingerprintLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, adj_ad=adj_ad, adj=adj, concat=False)

    def forward(self, x, names=None, print_flag=False):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        return F.log_softmax(x, dim=1)


class RWR_process(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj_ad, adj, dataset_str):
        """version of RWR_process."""
        super(RWR_process, self).__init__()
        self.dropout = dropout
        self.attentions = [RWRLayer(nfeat, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad, adj=adj, dataset_str=dataset_str, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = RWRLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, adj_ad=adj_ad, adj=adj, dataset_str=dataset_str, concat=False)

    def forward(self, x, names=None, print_flag=False):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        return F.log_softmax(x, dim=1)
