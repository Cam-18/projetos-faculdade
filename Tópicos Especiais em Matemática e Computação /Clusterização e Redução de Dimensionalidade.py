# Análise completa do dataset de diabetes, incluindo pré-processamento, seleção de características e clustering com K-Means

from sklearn.datasets import load_diabetes # Regressão
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import ParameterGrid
import numpy as np
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)

np.random.seed(1)

dados = load_diabetes(scaled = False)
df = pd.DataFrame(dados.data, columns = dados.feature_names) # Transformando em DataFrame

# Definindo X e y
X = dados.data
y = dados.target

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(f'X:\n{X}\ny:\n{y}\n')

# Descrição da base de dados - início
# Boxplot para checar outliers

plt.figure(figsize = (10, 6))  # Tamanho da figura
plt.boxplot(df.values, patch_artist = True)  # Cria o boxplot
plt.xticks(range(1, len(dados.feature_names) + 1), dados.feature_names, rotation = 45)  # Rótulos do eixo x
plt.title('Boxplot de Diabetes')  # Título do gráfico
plt.xlabel('Características')  # Nome do eixo x
plt.ylabel('Valores')  # Nome do eixo y
plt.tight_layout()  # Ajusta o layout
plt.show()

# Checando se há duplicatas
linhasDuplicadas = df[df.duplicated()]
if not linhasDuplicadas.empty:
    print(f'Linhas duplicadas: {linhasDuplicadas}')
else:
    print(f'Não há linhas duplicadas.')

# Checando se há valores nulos
valores_nulos = df.isnull().any()
if valores_nulos.any():
    print('As seguintes colunas contêm valores nulos:')
    print(valores_nulos[valores_nulos == True])
else:
    print('Não há valores nulos no DataFrame.')
#
# # Checando se há valores faltantes
valoresFaltantes = df.isnull().sum()
if valoresFaltantes.any() > 0:
    print(f'Há {valoresFaltantes.sum()} valores faltantes no DataFrame.')
else:
    print('Não há valores faltantes no DataFrame.\n')
# Descrição da base de dados - fim

# Análise dos dados - início
# Média de cada atributo
media = df.mean()

# Desvio padrão de cada atributo
desvioPadrao = df.std()

# Variância de cada atributo
variancia = df.var()

# Matriz de correlação entre os atributos
matCorrelacao = df.corr()

# Valor mínimo de cada atributo
minimo = df.min()

# Valor máximo de cada atributo
maximo = df.max()

print(f'\nMédia de cada atributo:\n{media}\n\nDesvio padrão de cada atributo:\n{desvioPadrao}\n\nVariância de cada '
      f'atributo:\n{variancia}\n\nValor mínimo de cada '
      f'atributo:\n{minimo}\n\nValor máximo de cada atributo:\n{maximo}\n')
print(f'Matriz de correlação entre os atributos:\n{matCorrelacao}\n\n')
# Análise dos dados - fim

# Pré-Processamento - início
# Padronização dos dados
Sscaler = StandardScaler()
X_scaled = Sscaler.fit_transform(X)

# Aplicando PCA
pca = PCA()
pca.fit(X_scaled)

# Obtendo as componentes principais
pca_components = pca.components_

# Criar um DataFrame para visualização
df_pca = pd.DataFrame(pca_components[:, :3], columns = [f'PC{i+1}' for i in range(3)], index = dados.feature_names)

# Exibir as características mais influentes nas três primeiras componentes principais
print(f'PCA:\n{df_pca}\n')
print(f'Características mais influentes em cada componente principal:\n{df_pca.idxmax()}\n')
# Pré-Processamento - fim


# K-Means e Davies-Bouldin antes da seleção de características
param_grid = {
    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'n_init': [10, 20]
}

best_params = None
best_davies_bouldin_score = np.inf # Varia de 0 ao infinito positivo

for params in ParameterGrid(param_grid):
    kmeans = KMeans(**params)
    y_pred = kmeans.fit_predict(X)
    db = davies_bouldin_score(X, y_pred) # Quanto mais próximo de 0 melhor

    if db < best_davies_bouldin_score:
        best_davies_bouldin_score = db
        best_params = params

kmeans = KMeans(**best_params)
y_pred_antes = kmeans.fit_predict(X)

print(f'Davies-Bouldin antes da seleção de características: {best_davies_bouldin_score}\n')

print(f'Melhores parâmetros antes da seleção de características: {best_params}\n')

print(f'Clusterlabels antes da seleção de características\n{cluster_labels}')

print(f'y_pred antes da seleção de características:\n{y_pred_antes}\n\n')

# Labels pra cor e marker
col = ['r', 'b', 'g', 'm', 'y', 'c', 'k', 'orange', 'purple', 'brown']
mar = ['o', '^', '*', 'x', '+', 's', 'D', 'v', '>', '<']
clabel = [col[int(x)] if x != -1 else 'k' for x in y_pred_antes]
mlabel = [mar[int(x)] if x != -1 else 'H' for x in y_pred_antes]

# PCA antes da seleção de características
pca = PCA(n_components = 2)
cluster_reduced_data = pca.fit_transform(X)

fig = plt.figure(figsize = (10, 6))
x1 = cluster_reduced_data[:, 0]
y1 = cluster_reduced_data[:, 1]
x1 = (x1 - x1.min()) / (x1.max() - x1.min())
y1 = (y1 - y1.min()) / (y1.max() - y1.min())

for i in range(len(x1)):
    plt.scatter(x1[i], y1[i], marker = mlabel[i], s = 100, c = clabel[i], edgecolor = 'k', linewidth = 0.5)
plt.xlim(-0.1, +1.4), plt.ylim(-0.1, +1.1)

plt.xlabel(f'Componente 1 (explica {100*pca.explained_variance_ratio_[0]:.2f}% de variabilidade)', size = 'large')
plt.ylabel(f'Componente 2 (explica {100*pca.explained_variance_ratio_[1]:.2f}% de variabilidade)', size = 'large')

plt.title('KMeans antes da seleção de características')
plt.show()


# Seleção de características - inicio
seletor = SelectKBest(score_func = f_regression, k = 5)
X_new = seletor.fit_transform(X, y)
indicesSelecionados = seletor.get_support(indices = True)
featuresNomes = df.columns[indicesSelecionados]

print('Características selecionadas:', ", ".join(featuresNomes))
# Seleção de características - fim


# K-Means e Davies-Bouldin depois da seleção de características
best_params = None
best_davies_bouldin_score = np.inf # Varia de 0 ao infinito positivo

for params in ParameterGrid(param_grid):
    kmeans = KMeans(**params)
    y_pred = kmeans.fit_predict(X_new)
    db = davies_bouldin_score(X_new, y_pred) # Quanto mais próximo de 0 melhor

    if db < best_davies_bouldin_score:
        best_davies_bouldin_score = db
        best_params = params

kmeans = KMeans(**best_params)
y_pred_depois = kmeans.fit_predict(X_new)

print(f'Davies-Bouldin depois da seleção de características: {best_davies_bouldin_score}\n')

print(f'Melhores parâmetros depois da seleção de características: {best_params}\n')

print(f'Clusterlabels depois da seleção de características\n{cluster_labels}\n')

print(f'y_pred depois da seleção de características\n{y_pred_depois}\n\n')

clabel = [col[int(x)] if x != -1 else 'k' for x in y_pred_depois]
mlabel = [mar[int(x)] if x != -1 else 'H' for x in y_pred_depois]

# PCA depois da seleção de características
pca = PCA(n_components = 2)
cluster_reduced_data = pca.fit_transform(X_new)

fig = plt.figure(figsize = (10, 6))
x1 = cluster_reduced_data[:, 0]
y1 = cluster_reduced_data[:, 1]
x1 = (x1 - x1.min()) / (x1.max() - x1.min())
y1 = (y1 - y1.min()) / (y1.max() - y1.min())

for i in range(len(x1)):
    plt.scatter(x1[i], y1[i], marker = mlabel[i], s = 100, c = clabel[i], edgecolor = 'k', linewidth = 0.5)
plt.xlim(-0.1, +1.4), plt.ylim(-0.1, +1.1)

plt.xlabel(f'Componente 1 (explica {100*pca.explained_variance_ratio_[0]:.2f}% de variabilidade)', size = 'large')
plt.ylabel(f'Componente 2 (explica {100*pca.explained_variance_ratio_[1]:.2f}% de variabilidade)', size = 'large')

plt.title('KMeans depois da seleção de características')
plt.show()
