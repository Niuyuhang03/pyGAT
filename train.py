from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
from torch.autograd import Variable

from utils import load_data, accuracy, multi_labels_nll_loss
from models import GAT, GAT_rel

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--mean', action='store_true', default=False, help='In Rel_GAT use mean or concat')
# LeakyReLU在x<0的斜率
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
# 数据集
parser.add_argument('--dataset', type=str, default='cora', help='DataSet of model')
# 是否考虑relation类型
parser.add_argument('--rel', action='store_true', default=False, help='Process relation')
# 实验名称，用于生成.pkl文件夹
parser.add_argument('--experiment', type=str, default='GAT', help='Name of current experiment.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print("use cuda: {}".format(args.cuda))
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, rel, rel_dict, labels, idx_train, idx_val, idx_test, nclass, names = load_data(path='./data/'+ args.dataset + '/', dataset=args.dataset, process_rel=args.rel)

# Model and optimizer
if args.rel:
    model = GAT_rel(nrel=rel.shape[1], nfeat=features.shape[1], nclass=nclass, dropout=args.dropout, nheads=args.nb_heads, alpha=args.alpha, dataset=args.dataset, experiment=args.experiment, use_cuda=args.cuda, use_mean=args.mean)
else:
    model = GAT(nfeat=features.shape[1], nhid=args.hidden, nclass=nclass, dropout=args.dropout, nheads=args.nb_heads, alpha=args.alpha, dataset=args.dataset, experiment=args.experiment, use_cuda=args.cuda)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    if args.rel:
        rel = rel.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)
if args.rel:
    rel = Variable(rel)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if args.rel:
        output = model(features, rel, rel_dict, adj)
    else:
        output = model(features, adj)
    loss_train = multi_labels_nll_loss(output[idx_train], labels[idx_train])
    acc_train, preds = accuracy(output[idx_train], labels[idx_train], args.cuda)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        if args.rel:
            output = model(features, rel, rel_dict, adj)
        else:
            output = model(features, adj)

    loss_val = multi_labels_nll_loss(output[idx_val], labels[idx_val])
    acc_val, preds = accuracy(output[idx_val], labels[idx_val], args.cuda)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data[0]),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.data[0]),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data[0]


def compute_test():
    model.eval()
    if args.rel:
        output = model(features, rel, rel_dict, adj, names, True)
    else:
        output = model(features, adj, names, True)
    loss_test = multi_labels_nll_loss(output[idx_test], labels[idx_test])
    acc_test, preds = accuracy(output[idx_test], labels[idx_test], args.cuda)
    print("pres:", preds)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test))


# Train model
files = glob.glob('./{}/*.pkl'.format(args.experiment))
for file in files:
    os.remove(file)

t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
if not os.path.exists(args.experiment):
    os.mkdir('{}'.format(args.experiment))

for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), './{}/{}.pkl'.format(args.experiment, epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    # 损失连续100次迭代没有优化时，则提取停止
    if bad_counter == args.patience:
        break

    files = glob.glob('./{}/*.pkl'.format(args.experiment))
    for file in files:
        epoch_nb = int(file.split('/')[-1].split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('./{}/*.pkl'.format(args.experiment))
for file in files:
    epoch_nb = int(file.split('/')[-1].split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('./{}/{}.pkl'.format(args.experiment, best_epoch)))

# Testing
compute_test()
