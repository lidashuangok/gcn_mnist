## Pipeline

先在mnist 上利用k邻近构造了一图，然后在图上做gcn半监督。

## Experiments 

考虑到计算性能，我只在mnist中的测试数据集实验，也就是一万个点。对这一万个点做knn邻近，这里k设置的为10，也就是10个最近的邻近的点。构造了一个1万个点，72800条边的图。在GCN我完全按照kpif & Maxwelling的实验，10%的标签率（1000个点做训练），1000个做验证，1000做测试。 最终结果92.7%左右