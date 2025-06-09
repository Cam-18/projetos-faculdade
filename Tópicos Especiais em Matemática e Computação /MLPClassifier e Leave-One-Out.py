# Avaliação do modelo de Rede Neural (MLPClassifier) no dataset de câncer de mama usando validação Leave-One-Out (LOO)
# e compara duas funções de ativação: ReLU e Logística.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

dados = load_breast_cancer()
X = dados.data
y = dados.target

# Inicializa a validação LeaveOneOut
loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]

clf = MLPClassifier(alpha = 1e-3, hidden_layer_sizes = (5, 10, 15), activation = 'relu')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Métricas com relu
accrelu = metrics.accuracy_score(y_test, y_pred)
f1relu = metrics.f1_score(y_test, y_pred)
print(f'Accuracy score com relu: {accrelu}')
print(f'f1 score com relu: {f1relu}')

# Métricas com Logistic
clf = MLPClassifier(alpha = 1e-3, hidden_layer_sizes = (5, 10, 15), activation = 'logistic')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acclog = metrics.accuracy_score(y_test, y_pred)
f1log = metrics.f1_score(y_test, y_pred)
print(f'Accuracy score com Logistic: {acclog}')
print(f'f1 score com Logistic: {f1log}')
