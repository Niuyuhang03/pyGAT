from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT

# Training settings
# 处理参数
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
# 迭代次数
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
# 头数，即K
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
# dropout概率
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
# LeakyReLU在x<0的斜率
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# 提前停止参数
parser.add_argument('--patience', type=int, default=100, help='Patience')
# 数据集
parser.add_argument('--dataset', type=str, default='cora', help='DataSet of model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 生成随机数种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# 加载数据
adj, features, labels, idx_train, idx_val, idx_test, nclass = load_data(path='./data/'+ args.dataset + '/', dataset=args.dataset)
# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=nclass,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=nclass,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)
loss_fn = nn.BCEWithLogitsLoss(reduce=True, size_average=True)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)

    # loss_fn = nn.BCELoss(reduce=True, size_average=True)
    # sigmoid_fn = nn.Sigmoid()
    # sigmoid_output_idx_train = sigmoid_fn(output[idx_train])
    # loss_train = loss_fn(sigmoid_output_idx_train, labels[idx_train].type_as(output))

    loss_train = loss_fn(output[idx_train], labels[idx_train].type_as(output))

    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train], args.cuda)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    # loss_fn = nn.BCELoss(reduce=True, size_average=True)
    # sigmoid_fn = nn.Sigmoid()
    # sigmoid_output_idx_val = sigmoid_fn(output[idx_val])
    # loss_val = loss_fn(sigmoid_output_idx_val, labels[idx_val].type_as(output))

    loss_val = loss_fn(output[idx_val], labels[idx_val].type_as(output))

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    
    acc_val = accuracy(output[idx_val], labels[idx_val], args.cuda)
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data[0]),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.data[0]),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data[0]


def compute_test():
    # 使model进入测试模式
    model.eval()
    output = model(features, adj)

    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])

    # loss_fn = nn.BCELoss(reduce=True, size_average=True)
    # sigmoid_fn = nn.Sigmoid()
    # sigmoid_output_idx_test = sigmoid_fn(output[idx_test])
    # loss_test = loss_fn(sigmoid_output_idx_test, labels[idx_test].type_as(output))

    loss_test = loss_fn(output[idx_test], labels[idx_test].type_as(output))

    acc_test = accuracy(output[idx_test], labels[idx_test], args.cuda)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test))

# Train model
# 训练模型
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    # 损失连续100次迭代没有优化时，则提取停止
    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
# 验证
compute_test()
