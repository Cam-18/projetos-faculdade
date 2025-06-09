# Compara o desempenho de um ensemble (Bagging + MLPRegressor) contra os métodos SVC e SVR em dois datasets

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# 1

dados = load_breast_cancer()
X = dados.data
y = dados.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)

# Ensemble
ens = BaggingRegressor(estimator = MLPRegressor(hidden_layer_sizes = (200,200,200)), n_estimators = 10, random_state = 9)
ens.fit(X_train, y_train)

y_predens = ens.predict(X_test)

mseens = mean_squared_error(y_test, y_predens)

# Método fraco
svc = SVC()
svc.fit(X_train, y_train)

y_predsvc = svc.predict(X_test)

msesvc = mean_squared_error(y_test, y_predsvc)

print(f'MSE Ensemble BC: {mseens}\nMSE SVC BC: {msesvc}\n')

# 2

dados = pd.read_csv('BostonHousing.csv')
X = dados.drop(columns = ['medv'])
y = dados['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)

# Ensemble
ens = BaggingRegressor(estimator = MLPRegressor(hidden_layer_sizes = (200,200,200)), n_estimators = 10, random_state = 9)
ens.fit(X_train, y_train)

y_predens = ens.predict(X_test)

mseens = mean_squared_error(y_test, y_predens)

# Método fraco
svr = SVR()
svr.fit(X_train, y_train)

y_predsvr = svr.predict(X_test)

msesvr = mean_squared_error(y_test, y_predsvr)

print(f'MSE Ensemble BH: {mseens}\nMSE SVC BH: {msesvr}')
