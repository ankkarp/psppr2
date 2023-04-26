import numpy as np
from sklearn.datasets import make_blobs
from matplotlib.pyplot import plt


# def random_graph(n):
#     heads, tails, weights = [], [], []
#     for i in range(n-1):
#         for j in range(i+1, n):
#             heads.append(i)
#             tails.append(j)
#     weights = np.random.rand(len(heads))
#     return list(zip(heads, tails, weights))

def eucl(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2, axis=1))


def random_graph(n):
    x, _ = make_blobs(n_samples=10, centers=1, n_features=2)
