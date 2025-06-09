# Aplica classificação (KNN) no dataset de diabetes e Regressão Linear no dataset de preços de carros.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Parte 1
df = pd.read_csv("diabetes.csv")

X = df.drop(['Outcome'], axis = 1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Classificador KNN
clf_knn = KNeighborsClassifier(n_neighbors = 5)
clf_knn.fit(x_train, y_train)

y_predknn = clf_knn.predict(x_test)

# Métricas
accknn = accuracy_score(y_test, y_predknn)
f1knn = f1_score(y_test, y_predknn)
kappaknn = cohen_kappa_score(y_test, y_predknn)

print(f'Accuracy Score: {accknn}\nF1 Score: {f1knn}\nCohen Kappa Score: {kappaknn}')

# Parte 2
df = pd.read_csv("Nigerian_Car_Prices.csv")

X = df.drop(['Price'], axis = 1)
y = df['Price']

print(f'Valores nulos em cada coluna:\n{X.isnull().sum()}\n')

X.dropna(inplace = True) # Exclui os valores nulos
y = y[X.index]

le = LabelEncoder() # Variáveis: Make, Condition, Fuel, Transmission e Build

X['Make'] = le.fit_transform(X['Make'])
X['Condition'] = le.fit_transform(X['Condition'])
X['Fuel'] = le.fit_transform(X['Fuel'])
X['Transmission'] = le.fit_transform(X['Transmission'])
X['Build'] = le.fit_transform(X['Build'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

reg = LinearRegression().fit(x_train, y_train)
y_pred= reg.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'mae: {mae}\nmse: {mse}\nr2: {r2}')
