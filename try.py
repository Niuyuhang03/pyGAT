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

def accuracy(output, labels):
    print("labels", labels)
    print("output:", output)
    preds = torch.zeros(labels.shape[0], labels.shape[1])
    all_labels_1_length = 0
    correct = 0
    for idx in range(len(labels)):
        labels_1_index = np.where(labels[idx])[0]
        labels_1_length = len(labels_1_index)
        print('labels_1_index:', labels_1_index)
        predict_1_index = output[idx].sort()[1][-labels_1_length:]
        print('predict_1_index:', predict_1_index)
        all_labels_1_length += labels_1_length
        for labels_1 in labels_1_index:
            if labels_1 in predict_1_index:
                correct += 1
        # 生成预测结果preds，仅查看用
        output_01 = np.zeros(7, dtype=np.float32)
        for i in predict_1_index:
            output_01[i] = 1.0
        preds[idx] = torch.from_numpy(output_01)
    preds = preds.type_as(labels)
    print("preds:", preds)
    return correct / all_labels_1_length
print("accuracy:", accuracy(output, labels))