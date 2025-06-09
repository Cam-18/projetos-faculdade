# Teste do Agglomerative Clustering e avaliação da qualidade dos clusters usando o índice Davies-Bouldin

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

dados = load_breast_cancer()
X = dados.data
y = dados.target

ac = AgglomerativeClustering()
y_pred = ac.fit_predict(X)
davies_bouldin = davies_bouldin_score(X, y_pred)

# Testa diferentes números de clusters e mostra o Davies-Bouldin para cada caso
for i in range(2, 11):
    ac = AgglomerativeClustering(n_clusters = i)
    y_pred = ac.fit_predict(X)
    db = davies_bouldin_score(X, y_pred)
    print(f'N° de cluster {i}\nDavies-Bouldin {db}') # Davies-Bouldin melhor resultado é o de menor número
