# Aplica, no dataset Wine, métodos de classificação (SVM e Random Forest) e avalia se a diferença na acurácia é significativa
# Aplica, no dataset mae-results, métodos de classificação (Friedman e Wilcoxon) e avalia se a diferença na acurácia é significativa

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlxtend.evaluate import paired_ttest_5x2cv
from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np
import pandas as pd

# 1

dados = load_wine()
X = dados.data
y = dados.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)

# Método de classificação SVM
clf_svm = svm.SVC()
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

# Método de classificação Random Forest
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f'Acurácia SVM: {acc_svm}')
print(f'Acurácia Random Forest: {acc_rf}\n')

# Avaliar se a diferença na acurácia é significativa
t, p = paired_ttest_5x2cv(estimator1 = clf_svm,
                          estimator2 = clf_rf,
                          X = X, y = y,
                          random_seed = 1)

print(f't statistic: {np.round(t, 2)}')
print(f'p value: {np.round(p, 2)}\n')

if p <= 0.05:
    print('Como p < 0.05, podemos rejeitar a hipótese nula de que ambos os modelos funcionam igualmente bem neste '
          'conjunto de dados. Podemos concluir que os dois algoritmos são significativamente diferentes..\n')
else:
    print('Como p > 0.05, não podemos rejeitar a hipótese nula e podemos concluir que o desempenho dos dois algoritmos '
          'não é significativamente diferente\n')

# 2

df = pd.read_csv('mae-results.csv', sep = '\t')

# Friedman Test
stat, p = friedmanchisquare(df['R1'], df['R2'], df['R3'])
print('Friedman Test')
print(f'Statistics = {stat:.3f}, p = {p:.3f}')

alpha = 0.05
if p <= 0.05:
    print('Como p < 0.05, podemos rejeitar a hipótese nula de que ambos os modelos funcionam igualmente bem neste '
          'conjunto de dados. Podemos concluir que os dois algoritmos são significativamente diferentes..\n')
else:
    print('Como p > 0.05, não podemos rejeitar a hipótese nula e podemos concluir que o desempenho dos dois algoritmos '
          'não é significativamente diferente\n')

# Wilcoxon Test
stat, p = wilcoxon(df['R2'], df['R3'])
print('\nWilcoxon Test')
print(f'Statistics = {stat:.3f}, p = {p:.3f}')

if p <= 0.05:
    print('Como p < 0.05, podemos rejeitar a hipótese nula de que ambos os modelos funcionam igualmente bem neste '
          'conjunto de dados. Podemos concluir que os dois algoritmos são significativamente diferentes..\n')
else:
    print('Como p > 0.05, não podemos rejeitar a hipótese nula e podemos concluir que o desempenho dos dois algoritmos '
          'não é significativamente diferente\n')
