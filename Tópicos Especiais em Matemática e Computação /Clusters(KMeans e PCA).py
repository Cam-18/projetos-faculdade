# Encontra clusters na base de dados Wine

from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dados = load_wine()

X = dados.data
label = dados.target
target_names = dados.target_names

# Agrupa os dados com KMeans
kmeans = KMeans(n_clusters = 3).fit(X)
y_kmeans = kmeans.fit_predict(X)
kmeans.cluster_centers_

# Reduz a dimensionalidade com PCA
pca = PCA(n_components = 3)
X_r = pca.fit_transform(X)

# Cria rótulos para o plot
col = ['r', 'b', 'g', 'm', 'y', 'c', 'k']
clabel = [int(x) for x in y_kmeans]

for i in range(len(clabel)):
      k = clabel[i]
      if k == -1:
            clabel[i] = 'k'
      else:
            clabel[i] = col[k]

mar = ['o', '^', '*', 'x', '+', 's', '8']
mlabel = [int(x) for x in y_kmeans]

for i in range(len(mlabel)):
    l = mlabel[i]
    if l == -1:
            mlabel[i] = 'H'
    else:
            mlabel[i] = mar[l]

x1 = X_r[:,0]; y1 = X_r[:,1]

for i in range(len(x1)):
    plt.scatter(x1[i], y1[i], marker = mlabel[i], s = 25, c = clabel[i])
    plt.text(x1[i]+0.03, y1[i]+0.00, '%s' % 'C'+str(label[i]), ha = 'left', va = 'center', fontsize = 8)

plt.xlabel('Componente 1 expressa ' + str("{:.2f}".format(100*pca.explained_variance_ratio_[0])) + '% de variabilidade')
plt.ylabel('Componente 2 expressa ' + str("{:.2f}".format(100*pca.explained_variance_ratio_[1])) + '% de variabilidade')
plt.title('KMeans Wine')
plt.show()
