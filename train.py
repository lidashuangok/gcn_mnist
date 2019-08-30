# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
NHID = 16
weight_decay = 5e-4
learning_rate = 0.001
Dropout=0.5

import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import GCN
import utils_graph

adj, features, labels, idx_train, idx_val, idx_test = utils_graph.load_data()

#print(features.shape)

model = GCN(features.shape[1],NHID,10,Dropout)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
#print(model.parameters())
model = model.cuda()
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    loss_train = F.nll_loss(output[idx_train],labels[idx_train])
    acc_val = utils_graph.accuracy(output[idx_val],labels[idx_val])
    loss_train.backward()
    optimizer.step()
    if epoch%100==0:
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train.item()),'acc_val: {:.4f}'.format(acc_val.item()))

def test():

    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = utils_graph.accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
t_total = time.time()
for epoch in range(20500):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
test()
