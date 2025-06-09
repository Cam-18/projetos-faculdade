# Implementação do classificador KNN para prever tumores malignos ou benignos no dataset de câncer de mama

from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dados = load_breast_cancer()
X = dados.data
y = dados.target

# Divide os dados em treino e teste (80% treino e 20% teste)
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.20)

# Treina um modelo com 3 vizinhos mais próximos
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x_treino, y_treino)

# Previsões nos dados de teste
y_pred = classifier.predict(x_teste)

# Calcula a taxa de acerto
cont = 0
for i in range(len(y_pred)):
	if(y_teste[i] == y_pred[i]):
		cont += 1

# Mostra o percentual de acerto
acertos_Bcancer_knn = cont/len(y_pred)
print(f'Percentual de acerto: {acertos_Bcancer_knn * 100:.2f}%.')
