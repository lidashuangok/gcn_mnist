# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
from sklearn.neighbors import kneighbors_graph
import utils
import time
import numpy
from scipy import sparse

t_star = time.time()
data,label,testdata,test_label = utils.load()
t_load = time.time()
print("load time elapsed: {:.4f}s".format(t_load- t_star))
X = [[0,2], [3,1], [1,4],[4,2],[7,4]]
adj = kneighbors_graph(testdata, 10,mode='connectivity', include_self=True)
#print(adj.toarray())
#print(adj.nnz)
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#print(adj.toarray())
print("total time elapsed: {:.4f}s".format(time.time() - t_star))
numpy.save("adj.npy",adj.toarray())
sparse.save_npz('adj_sp.npz', adj)
print(adj.nnz/2)
#print(adj.toarray())