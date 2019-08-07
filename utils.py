import numpy as np
import scipy.sparse as sp
import torch
from collections import Counter


np.set_printoptions(threshold=np.inf)


def encode_onehot(labels):
    classes = set()
    for label in labels:
        classes |= set(label)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array([np.sum([classes_dict.get(l) for l in label], axis=0) for label in labels], dtype=np.int32)
    return labels_onehot, len(classes)


def load_data(path, dataset, process_rel):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = list(map(lambda x: x.split(','), idx_features_labels[:, -1]))
    labels, nclass = encode_onehot(labels)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 将每组关系中的entity用索引表示
    edges = np.array(list(map(idx_map.get, edges_unordered[:, :2].flatten())), dtype=np.int32).reshape(edges_unordered[:, :2].shape)
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    # 将非对称邻接矩阵转变为对称邻接矩阵，有向图变为无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    # Implementation from paper
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj = torch.FloatTensor(np.array(adj.todense()))

    # Tricky implementation of official GAT
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj[adj==0] = -9e15
    adj[adj>=1] = 0
    adj = torch.FloatTensor(np.array(adj))

    # 生成relation embeddings的结果rel和entities之间rel的对应字典rel_dict，该字典中ent1和ent2的所有关系都存入ent1+ent2中
    rel_dict = {}
    if process_rel:
        idx_rel = np.genfromtxt("{}{}.rel".format(path, dataset), dtype=np.dtype(str))
        rel = torch.FloatTensor(np.array(sp.csr_matrix(idx_rel[:, 1:], dtype=np.float32).todense()))
        for index in range(len(edges_unordered)):
            e1, e2 = edges[index][:2]
            r = edges_unordered[index][2]
            if rel_dict.get(e1, None) is not None:
                rel_dict[str(e1) + '+' + str(e2)] = rel_dict[str(e1) + '+' + str(e2)].add(r)
            elif rel_dict.get(e2, None) is not None:
                rel_dict[str(e2) + '+' + str(e1)] = rel_dict[str(e2) + '+' + str(e1)].add(r)
            else:
                rel_dict[str(e1)+'+'+str(e2)] = {r}
    else:
        rel = torch.FloatTensor()

    if dataset == 'cora':  # cora采用固定划分法，train中每个class取20个，valid大小300，test大小1000
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    else:  # 其他数据集采用train:val:test = 6:2:2划分
        idx_train = range(len(idx_map) // 10 * 8)
        idx_val = range(len(idx_map) // 10 * 8, len(idx_map) // 10 * 9)
        idx_test = range(len(idx_map) // 10 * 9, len(idx_map))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print('Loading {} dataset finishes...'.format(dataset))
    return adj, features, rel, rel_dict, labels, idx_train, idx_val, idx_test, nclass


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels, is_cuda):
    output = np.array(output.detach())
    cnt = len(np.where(labels)[1])
    counter = Counter(np.where(labels)[0])
    preds = np.zeros_like(labels)
    for idx in range(labels.shape[0]):
        labels_1_length = counter[idx]
        predict_1_index = np.argsort(-output[idx])[:labels_1_length]
        preds[idx][predict_1_index] = 1
    preds = torch.FloatTensor(preds)
    if is_cuda:
        preds = preds.cuda()
    correct = preds.type_as(labels).mul(labels).sum()
    return correct.item() / cnt, preds


def multi_labels_nll_loss(output, labels):
    # labels和output按位点乘，结果相加，除以labels中1的总数，作为适用于多标签的nll_loss。
    loss = -labels.type_as(output).mul(output).sum()
    cnt = len(np.where(labels)[1])
    return loss / cnt
