# Aplica o PCA (Análise de Componentes Principais) técnica de redução de dimensionalidade e extração de características

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns

# 1: Carregue a base de dados.
wine = load_wine()

X = wine.data
y = wine.target

# 2: Selecione as três características mais relevantes de acordo com o método Select K Best.
selector = SelectKBest(f_classif, k = 3)
X_new = selector.fit_transform(X, y)

# 3: Encontre as duas primeiras componentes principais e plot os dados considerando-as. Mostre a variabilidade
# que cada componente está expressando.
pca = decomposition.PCA(n_components = 2) # Duas primeiras componentes principais.
pca.fit(X)
X = pca.transform(X)
var = pca.explained_variance_ratio_ # Variabilidade de cada componente.

# Plota gráfico do PCA
plt.scatter(X[:, 0], X[:, 1], c = y, alpha = .8, lw = 2)
plt.title('PCA da base de dados WINE')
plt.xlabel(f'Componente 1 expressa {var[0]*100:.2f}% de variabilidade')
plt.ylabel(f'Componente 2 expressa {var[1]*100:.2f}% de variabilidade')
plt.show()

# 4: Plote a matriz de correlação das características utilizando o seaborn.
X_new = pd.DataFrame(X_new)
corr = X_new.corr()

plt.figure(figsize = (8, 6))
color = sns.color_palette("Blues", as_cmap = True)
sns.heatmap(corr, annot = True, cmap = color, fmt = ".2f", linewidths = 0.5)
plt.title('Matriz de Correlação')
plt.show()
