import numpy as np
import scipy.sparse as sp
import torch


# 将标签转换为one-hot编码形式
def encode_onehot(labels):
    classes = set(labels)
    # np.identity()函数创建方针，返回主对角线元素为1，其余元素为0的数组
    # enumerate()函数用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列
    # 同时列出数据和数据下标，一般用在for循环中
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # map()函数根据提供的函数对指定序列做映射
    # map(function, iterable)
    # 第一个参数function以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path, dataset):
    """Load citation network dataset (cora only for now)"""
    # str.format()函数用于格式化字符串
    print('Loading {} dataset...'.format(dataset))

    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype, delimiter, usecols, skip_header)
    # frame：文件名
    # dtype：数据类型
    # delimiter：分隔符
    # usecols：选择读哪几列，通常将属性集读为一个数组，将标签读为一个数组
    # skip_header：是否跳过表头
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 提取样本的标签，并将其转换为one-hot编码形式
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 样本的id数组
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 有样本id到样本索引的映射字典
    idx_map = {j: i for i, j in enumerate(idx)}
    # 样本之间的引用关系数组
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 将样本之间的引用关系用样本索引之间的关系表示
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    # 将非对称邻接矩阵转变为对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 在邻接矩阵中加入自连接
    features = normalize_features(features)
    # 对加入自连接的邻接矩阵进行对称归一化处理
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # 数据集划分
    # 训练集索引列表
    idx_train = range(140)
    # 验证集索引列表
    idx_val = range(200, 500)
    # 测试集索引列表
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels_one_hot = torch.FloatTensor(labels)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, labels_one_hot, idx_train, idx_val, idx_test


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


def accuracy(output, labels_one_hot, labels):
    print('label_one_hot.shape:', labels_one_hot.shape)
    print('output.shape', output.shape)
    print('output.max(1)', output.max(1))
    print('output.max(1)[1]', output.max(1)[1])
    print('output.max(1)[1].shape', output.max(1)[1].shape)
    print('preds.shape', output.max(1)[1].type_as(labels_one_hot).shape)
    print('preds', output.max(1)[1].type_as(labels_one_hot))
    
    preds = output.max(1)[1].type_as(labels_one_hot)
    correct = preds.eq(labels_one_hot).double()
    correct = correct.sum()
    return
    return correct / len(labels)

