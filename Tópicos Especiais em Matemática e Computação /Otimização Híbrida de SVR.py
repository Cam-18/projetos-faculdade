# Otimização Híbrida de SVR para Regressão em Diabetes: Comparando DE, PSO e Pré-processamento

from sklearn.datasets import load_diabetes # Regressão
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn import preprocessing
from scipy.optimize import differential_evolution as de
from pyswarm import pso
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)

dados = load_diabetes(scaled = False)
df = pd.DataFrame(dados.data, columns = dados.feature_names) # Transformando em DataFrame

# Definindo X e y
X = dados.data
y = dados.target

# Normalização(aula 3) de X e y
X_scaled = preprocessing.scale(X)
y_scaled = preprocessing.scale(y)

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

# Método: Máquina de Vetor Suporte(aula 8)   K-Fold(aula 9)     Métricas(aula 9)

def func(x, *args):
    X, y, flag = args
    kf = KFold(n_splits = 5)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    C = x[0]
    svr = SVR(C = C)

    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mabse = median_absolute_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    if flag == 'eval':
        return mse
    else:
        return mae, mabse, mape

lista = []

bounds = [(0.01, 1)]
lb = [(0.01)]
ub = [(1)]

flag = 'eval'
args = (X, y, flag)
args_scaled = (X_scaled, y_scaled, flag)

for run in range(30):
    st = time.time()

    # Antes do pp
    opt_de = de(func = func, bounds = bounds, args = args)
    opt_pso = pso(func = func, lb = lb, ub = ub, args = args) # opt_pso[0]: x   opt_pso[1]: fun

    args1 = (X, y, 'run')

    mae_opt_de, mabse_opt_de, mape_opt_de = func(opt_de['x'],*args1)
    mae_opt_pso, mabse_opt_pso, mape_opt_pso = func(opt_pso[0], *args1)

    # Depois do pp
    opt_de_scaled = de(func = func, bounds = bounds, args = args_scaled)
    opt_pso_scaled = pso(func = func, lb = lb, ub = ub, args = args)

    args_scaled1 = (X_scaled, y_scaled, 'run')

    mae_opt_scaled_de, mabse_opt_scaled_de, mape_opt_scaled_de = func(opt_de_scaled['x'], *args_scaled1)
    mae_opt_scaled_pso, mabse_opt_scaled_pso, mape_opt_scaled_pso = func(opt_pso_scaled[0], *args_scaled1)

    st = time.time() - st

    l = {
        'DATASET': 'Diabetes',
        'MODEL': 'SVM',
        'RUN': run,
        'XOPT_DE': opt_de['x'],
        'XOPT_DE_SCALED': opt_de_scaled['x'],
        'FOPT_DE': opt_de['fun'],
        'FOPT_DE_SCALED': opt_de_scaled['fun'],
        'XOPT_PSO': opt_pso[0],
        'XOPT_PSO_SCALED': opt_pso_scaled[0],
        'FOPT_PSO': opt_pso[1],
        'FOPT_PSO_SCALED': opt_pso_scaled[1],
        'TIME': st,
        'MAEDEantes': mae_opt_de,
        'MAbsEDEantes': mabse_opt_de,
        'MAPEDEantes': mape_opt_de,
        'MAEPSOantes': mae_opt_pso,
        'MAbsEPSOantes': mabse_opt_pso,
        'MAPEPSOantes': mape_opt_pso,
        'MAEDEdepois': mae_opt_scaled_de,
        'MAbsEDEdepois': mabse_opt_scaled_de,
        'MAPEDEdepois': mape_opt_scaled_de,
        'MAEPSOdepois': mae_opt_scaled_pso,
        'MAbsEPSOdepois': mabse_opt_scaled_pso,
        'MAPEPSOdepois': mape_opt_scaled_pso
    }

    print(f'\n{l}\n')

    lista.append(l)

dfResultados = pd.DataFrame(lista)

dfResultados.to_csv('ResultadosTrabalho3.csv', index = False)

# print(f'DataFrame Resultados:\n{dfResultados}\n')

# Média das métricas antes do pp DE
mediaMAEDEantes = dfResultados['MAEDEantes'].mean()
mediaMAbsEDEantes = dfResultados['MAbsEDEantes'].mean()
mediaMAPEDEantes = dfResultados['MAPEDEantes'].mean()

# Média das métricas antes do pp PSO
mediaMAEPSOantes = dfResultados['MAEPSOantes'].mean()
mediaMAbsEPSOantes = dfResultados['MAbsEPSOantes'].mean()
mediaMAPEPSOantes = dfResultados['MAPEPSOantes'].mean()

# Média das métricas depois do pp DE
mediaMAEDEdepois = dfResultados['MAEDEdepois'].mean()
mediaMAbsEDEdepois = dfResultados['MAbsEDEdepois'].mean()
mediaMAPEDEdepois = dfResultados['MAPEDEdepois'].mean()

# Média das métricas depois do pp PSO
mediaMAEPSOdepois = dfResultados['MAEPSOdepois'].mean()
mediaMAbsEPSOdepois = dfResultados['MAbsEPSOdepois'].mean()
mediaMAPEPSOdepois = dfResultados['MAPEPSOdepois'].mean()

# Desvio Padrão das métricas antes do pp DE
stdMAEDEantes = dfResultados['MAEDEantes'].std()
stdMAbsEDEantes = dfResultados['MAbsEDEantes'].std()
stdMAPEDEantes = dfResultados['MAPEDEantes'].std()

# Desvio Padrão das métricas antes do pp PSO
stdMAEPSOantes = dfResultados['MAEPSOantes'].std()
stdMAbsEPSOantes = dfResultados['MAbsEPSOantes'].std()
stdMAPEPSOantes = dfResultados['MAPEPSOantes'].std()

# Desvio Padrão das métricas depois do pp DE
stdMAEDEdepois = dfResultados['MAEDEdepois'].std()
stdMAbsEDEdepois = dfResultados['MAbsEDEdepois'].std()
stdMAPEDEdepois = dfResultados['MAPEDEdepois'].std()

# Desvio Padrão das métricas depois do pp PSO
stdMAEPSOdepois = dfResultados['MAEPSOdepois'].std()
stdMAbsEPSOdepois = dfResultados['MAbsEPSOdepois'].std()
stdMAPEPSOdepois = dfResultados['MAPEPSOdepois'].std()

print(f'Médias e Desvio Padrão:\n\n'
      f'DE antes do pp:\n'
      f'MAE: {mediaMAEDEantes} \u00B1 {stdMAEDEantes}\n'
      f'MAbsE: {mediaMAbsEDEantes} \u00B1 {stdMAbsEDEantes}\n'
      f'MAPE: {mediaMAPEDEantes} \u00B1 {stdMAPEDEantes}\n\n'
      f'DE depois do pp:\n'
      f'MAE: {mediaMAEDEdepois} \u00B1 {stdMAEDEdepois}\n'
      f'MAbsE: {mediaMAbsEDEdepois} \u00B1 {stdMAbsEDEdepois}\n'
      f'MAPE: {mediaMAPEDEdepois} \u00B1 {stdMAPEDEdepois}\n\n')

print(f'PSO antes do pp:\n' 
      f'MAE: {mediaMAEPSOantes} \u00B1 {stdMAEPSOantes}\n'
      f'MAbsE: {mediaMAbsEPSOantes} \u00B1 {stdMAbsEPSOantes}\n'
      f'MAPE: {mediaMAPEPSOantes} \u00B1 {stdMAPEPSOantes}\n\n'
      f'PSO depois do pp:\n'
      f'MAE: {mediaMAEPSOdepois} \u00B1 {stdMAEPSOdepois}\n'
      f'MAbsE: {mediaMAbsEPSOdepois} \u00B1 {stdMAbsEPSOdepois}\n'
      f'MAPE: {mediaMAPEPSOdepois} \u00B1 {stdMAPEPSOdepois}\n\n')

# DE antes
print(f'Linha do melhor valor de MAE para o DE antes:\n'
      f'{dfResultados['MAEDEantes'].idxmin()}\tValor: {dfResultados['MAEDEantes'].min()}\n\n')
print(f'Linha do melhor valor de MAbsE para o DE antes:\n'
      f'{dfResultados['MAbsEDEantes'].idxmin()}\tValor: {dfResultados['MAbsEDEantes'].min()}\n\n')
print(f'Linha do melhor valor de MAPE para o DE antes:\n'
      f'{dfResultados['MAPEDEantes'].idxmin()}\tValor: {dfResultados['MAPEDEantes'].min()}\n\n')

# DE depois
print(f'Linha do melhor valor de MAE para o DE depois:\n'
      f'{dfResultados['MAEDEdepois'].idxmin()}\tValor: {dfResultados['MAEDEdepois'].min()}\n\n')
print(f'Linha do melhor valor de MAbsE para o DE depois:\n'
      f'{dfResultados['MAbsEDEdepois'].idxmin()}\tValor: {dfResultados['MAbsEDEdepois'].min()}\n\n')
print(f'Linha do melhor valor de MAPE para o DE depois:\n'
      f'{dfResultados['MAPEDEdepois'].idxmin()}\tValor: {dfResultados['MAPEDEdepois'].min()}\n\n')

# PSO antes
print(f'Linha do melhor valor de MAE para o PSO antes:\n'
      f'{dfResultados['MAEPSOantes'].idxmin()}\tValor: {dfResultados['MAEPSOantes'].min()}\n\n')
print(f'Linha do melhor valor de MAbsE para o PSO antes:\n'
      f'{dfResultados['MAbsEPSOantes'].idxmin()}\tValor: {dfResultados['MAbsEPSOantes'].min()}\n\n')
print(f'Linha do melhor valor de MAPE para o PSO antes:\n'
      f'{dfResultados['MAPEPSOantes'].idxmin()}\tValor: {dfResultados['MAPEPSOantes'].min()}\n\n')

# PSO depois
print(f'Linha do melhor valor de MAE para o PSO depois:\n'
      f'{dfResultados['MAEPSOdepois'].idxmin()}\tValor: {dfResultados['MAEPSOdepois'].min()}\n\n')
print(f'Linha do melhor valor de MAbsE para o PSO depois:\n'
      f'{dfResultados['MAbsEPSOdepois'].idxmin()}\tValor: {dfResultados['MAbsEPSOdepois'].min()}\n\n')
print(f'Linha do melhor valor de MAPE para o PSO depois:\n'
      f'{dfResultados['MAPEPSOdepois'].idxmin()}\tValor: {dfResultados['MAPEPSOdepois'].min()}\n\n')
