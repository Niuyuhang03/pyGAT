import torch
import torch.nn as nn
import numpy as np

output = torch.randn(5, 7)
labels = torch.randn(5, 7)
length = 2
for idx in range(len(labels)):
    predict_1_index = labels[idx].sort()[1][-length:]
    temp = np.zeros(7, dtype=np.float32)
    for i in predict_1_index:
        temp[i] = 1.0
    labels[idx] = torch.from_numpy(temp)

loss_fn = nn.BCEWithLogitsLoss()
loss_train = loss_fn(output, labels)
print("loss_train", loss_train)

print("labels", labels)
print("output:", output)

def accuracy(output, labels):
    preds = torch.zeros(labels.shape[0], labels.shape[1])
    for idx in range(len(labels)):
        length = len(np.where(labels[idx])[0])
        # print('labels_one_hot[idx]', labels_one_hot[idx])
        # print("length", length)
        # print('output[idx]:', output[idx])
        predict_1_index = output[idx].sort()[1][-length:]
        # print("output[idx].sort()", output[idx].sort()[0])
        # print("output[idx].sort()[-length]", output[idx].sort()[0][-length])
        output_01 = np.zeros(7, dtype=np.float32)
        for i in predict_1_index:
            output_01[i] = 1.0
        preds[idx] = torch.from_numpy(output_01)
        # print("preds[idx]:", preds[idx])
    preds = preds.type_as(labels)
    print("preds:", preds)
    correct = preds.eq(labels).double()
    print("correct:", correct)
    correct = correct.sum()
    print("correct.sum:", correct)
    return correct / labels.shape[0] / labels.shape[1]
print("accuracy:", accuracy(output, labels))