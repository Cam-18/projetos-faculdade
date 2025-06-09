# Implementação do modelo de regressão KNN no dataset de diabetes e avalia seu desempenho usando validação cruzada KFold
# e erro absoluto médio (MAE).

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

dados = load_diabetes()
X = dados.data
y = dados.target

kf = KFold(n_splits = 5)

for train_index, test_index in kf.split(X):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]

regressor = KNeighborsRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'mean_absolute_error: {mae}')
