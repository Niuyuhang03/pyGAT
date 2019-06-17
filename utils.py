import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set()
    for label in labels:
        classes |= set(label)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array([np.sum([classes_dict.get(l) for l in label], axis=0) for label in labels], dtype=np.int32)
    return labels_onehot, len(classes)


def load_data(path, dataset):
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
    # 将样本之间的引用关系用样本索引之间的关系表示
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix, 将非对称邻接矩阵转变为对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    # Implementation from paper
    #adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    #adj = torch.FloatTensor(np.array(adj.todense()))

    # Tricky implementation of official GAT
    adj = (adj + sp.eye(adj.shape[0])).todense()
    for x in range(0, adj.shape[0]):
        for y in range(0, adj.shape[1]):
            if adj[x,y] == 0:
                adj[x,y] = -9e15
            elif adj[x,y] == 1:
                adj[x,y] = 0
            else:
                print(adj[x,y], 'error')
    adj = torch.FloatTensor(np.array(adj))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print('Loading {} dataset finishes...'.format(dataset))
    return adj, features, labels, idx_train, idx_val, idx_test, nclass


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
    # print("labels:", labels)
    # print("output:", output)
    preds = torch.zeros(labels.shape[0], labels.shape[1])
    all_labels_1_length = 0
    correct = 0
    for idx in range(len(labels)):
        labels_1_index = np.where(labels[idx])[0]
        labels_1_length = len(labels_1_index)
        predict_1_index = output[idx].sort()[1][-labels_1_length:]
        all_labels_1_length += labels_1_length
        for labels_1 in labels_1_index:
            if labels_1 in predict_1_index:
                correct += 1
        # 生成预测结果preds，仅查看用
        output_01 = np.zeros(labels.shape[1], dtype=np.float32)
        for i in predict_1_index:
            output_01[i] = 1.0
        preds[idx] = torch.from_numpy(output_01)
    if is_cuda:
        preds = preds.cuda()
    preds = preds.type_as(labels)
    # print("preds:", preds)
    return correct / all_labels_1_length, preds

