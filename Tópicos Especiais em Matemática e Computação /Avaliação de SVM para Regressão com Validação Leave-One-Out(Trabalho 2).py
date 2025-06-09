# Avaliação de SVM para Regressão em Dados de Diabetes com Validação Leave-One-Out

from sklearn.datasets import load_diabetes # Regressão
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.svm import SVC
import warnings
import time
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)

dados = load_diabetes(scaled = False)
df = pd.DataFrame(dados.data, columns = dados.feature_names) # Transformando em DataFrame

# Definindo X e y
X = dados.data
y = dados.target

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# print(df)

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

# Checando se há valores faltantes
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
      f'atributo:\n{variancia}\n\nValor mínimo de cada atributo:\n{minimo}\n\nValor máximo de cada atributo:\n{maximo}\n')
print(f'Matriz de correlação entre os atributos:\n{matCorrelacao}\n\n')
# Análise dos dados - fim

loo = LeaveOneOut()
svc = SVC()

pg = {'C': [1, 10, 100],
      'gamma': [0.01, 0.1, 1]}

# Resto 7: Leave one out(aula 9) e Máquina de vetor suporte(aula 8)
# GridSerachCV(aula 11.2)

lista = []
for run in range(30):
    st = time.time()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    gs = GridSearchCV(estimator = svc, param_grid = pg, cv = loo, n_jobs = -1, verbose = 0)

    gs.fit(X_train, y_train)

    y_pred = gs.predict(X_test)

    st = time.time() - st

    l = {
        'DATASET': 'Diabetes',
        'MODEL': 'SVM',
        'RUN': run,
        'BEST_PARAMS': gs.best_params_,
        'TIME': st,
        'Y_TRUE': y_test,
        'Y_PRED': y_pred,
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'MSLE': mean_squared_log_error(y_test, y_pred)
    }

    print(f'\n{l}\n')

    lista.append(l)

dfResultados = pd.DataFrame(lista)

dfResultados.to_csv('ResultadosTrabalho2.csv', index = False)

# print(f'DataFrame Resultados:\n{dfResultados}\n')

mediaMAE = dfResultados['MAE'].mean()
mediaMSE = dfResultados['MSE'].mean()
mediaMSLE = dfResultados['MSLE'].mean()

stdMAE = dfResultados['MAE'].std()
stdMSE = dfResultados['MSE'].std()
stdMSLE = dfResultados['MSLE'].std()

print(f'Médias:\nMAE: {mediaMAE}\nMSE: {mediaMSE}\nKappa: {mediaMSLE}\n\nDesvio-Padrão:\nMAE: {stdMAE}\nMSE: {stdMSE}\n'
      f'MSLE: {stdMSLE}')
