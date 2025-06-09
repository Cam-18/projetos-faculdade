# Análise de agrupamento fuzzy no dataset de câncer de mama e visualização dos resultados usando PCA

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import skfuzzy as fuzz

dados = load_breast_cancer()
X = dados.data
y = dados.target

fpcs = []
for n_clusters in range(2, 11):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)
    fpcs.append(fpc)

best_n_clusters = np.argmax(fpcs) + 2

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, best_n_clusters, 2, error=0.005, maxiter=1000, init=None)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
cluster_membership = np.argmax(u, axis=0)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_membership)
plt.show()
