# Análise de regressão linear simples entre o nível de triglicerídeos e a progressão da diabetes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from sklearn.datasets import load_diabetes
from scipy.stats import f

np.random.seed(30)

dados = load_diabetes(scaled = False)
X_ = dados.data
y = dados.target

# print(dados.feature_names)

X = X_[:, 8] # Escolhendo uma coluna de X (Nível de Triglicerídeo)

n = len(X)

df = pd.DataFrame({
    'Nível de Triglicerídeo': X,
    'Progressão da doença': y
})

# print(f'X:\n{X}\n{len(X)}\ny:\n{y}\n{len(y)}\n\n')

def calculo_Sxx(x_barra, x):
    # Somatório de (x_i - x_barra)^2
    Sxx = np.sum((x - x_barra) ** 2)
    return Sxx

def calculo_Syy(y_barra, y):
    # Somatório de (y_i - y_barra)^2
    Syy = np.sum((y - y_barra) ** 2)
    return Syy

def calculo_Sxy(x_barra, x, y_barra, y):
    # Somatório de (x_i - x_barra) * (y_i - y_barra)
    Sxy = np.sum((x - x_barra) * (y - y_barra))
    return Sxy

def calculo_coef_de_correlacao(Sxx, Syy, Sxy):
    coefcorr = Sxy / np.sqrt(Sxx * Syy)
    return coefcorr

def calculo_beta0(y_barra, x_barra, b_chapeu):
    # â = y_barra - b_chapéu . x_barra
    beta_0 = y_barra - (b_chapeu * x_barra)
    return beta_0

def calculo_beta1(Sxx, y, x, x_barra):
    # b_chapéu = somatório y_i (x_i - x_barra) / Sxx
    beta_1 = (np.sum(y * (x - x_barra))) / Sxx
    return beta_1

def calculo_SQE(y_chapeu, y):
    SQE = np.sum((y - y_chapeu) ** 2)
    return SQE

def calculo_SQReg(y_chapeu, y_barra):
    SQReg = np.sum((y_chapeu - y_barra) ** 2)
    return SQReg

# Calculando x_barra e y_barra
x_barra = np.mean(X)
y_barra = np.mean(y)

# Plotar Nível de Triglicerídeo (X)
plt.scatter(range(len(X)), X)
plt.title('Nível de Triglicerídeo')
plt.xlabel('Índice')
plt.ylabel('Nível de Triglicerídeo')
plt.show()

# Plotar Progressão da doença (y)
plt.scatter(range(len(y)), y)
plt.title('Progressão da doença')
plt.xlabel('Índice')
plt.ylabel('Progressão da doença')
plt.show()

# Plotar Nível de Triglicerídeo x Progressão da doença
plt.scatter(X, y)
plt.title('Nível de Triglicerídeo x Progressão da doença')
plt.xlabel('Nível de Triglicerídeo')
plt.ylabel('Progressão da doença')
plt.show()

# Atribuindo os valores das funções à variáveis
Sxx = calculo_Sxx(x_barra, X)
Syy = calculo_Syy(y_barra, y)
Sxy = calculo_Sxy(x_barra, X, y_barra, y)
coef_corr = calculo_coef_de_correlacao(Sxx, Syy, Sxy)
beta_1 = calculo_beta1(Sxx, y, X, x_barra)
beta_0 = calculo_beta0(y_barra, x_barra, beta_1)
variancia_ao_quadrado = Syy / n

# Função da reta y_chapeu = â + b_chapeu . x
y_chapeu = beta_0 + beta_1 * X

# Plotar gráfico com os pontos e a reta calculada anteriormente
plt.scatter(X, y)
plt.plot(X, y_chapeu, color = 'red')
plt.title('Nível de Triglicerídeo x Progressão da doença')
plt.xlabel('Nível de Triglicerídeo')
plt.ylabel('Progressão da doença')
plt.show()

# Calcular resíduos -> e_i = y_i - y_chapeu_i
residuos = y - y_chapeu

# Tabela ANOVA
# SQT, SQE, SQReg
SQT = Syy
SQE = calculo_SQE(y_chapeu, y)
SQReg = calculo_SQReg(y_chapeu, y_barra)
F_0 = SQReg / SQE / (n - 2)

# Dados da tabela
tabelaANOVA = [
    ['Regressão', '1', f'{SQReg:.2f}', f'{SQReg:.2f}', f'{F_0:.4f}'],
    ['Erro', f'{n - 2}', f'{SQE:.2f}', f'{SQE / (n - 2):.2f}', ' '],
    ['Total', f'{n - 1}', f'{SQT:.2f}', ' ', ' ']
]

# Definindo os nomes das colunas
colunas = ['Fonte de Variação', 'GL', 'SQ', 'QM', 'F_0']

# Imprimindo a tabela ANOVA
print(f'\nTabela ANOVA:')
print(tabulate(tabelaANOVA, headers = colunas, tablefmt = 'grid',
               colalign = ('center', 'center', 'center', 'center', 'center')))

# TESTE
print(f'\nSxx = {Sxx:.2f}\n\nSyy = {Syy:.2f}\n\nSxy = {Sxy:.2f}\n\nx_barra = {x_barra:.2f}\n\ny_barra = {y_barra:.2f}\n\n'
      f'Coeficiente de correlação = {coef_corr:.2f}\n\nBeta_0(Coeficiente angular) = {beta_0:.2f}\n\n'
      f'Beta_1(Coeficiente linear) = {beta_1:.2f}\n\nVariância ao quadrado = {variancia_ao_quadrado:.2f}\n\n')


# REMOVENDO OS PONTOS INFLUENTES(OUTLIERS)

# Adiciona uma nova coluna ao DataFrame com os valores dos resíduos
df['residuos'] = residuos

# Filtrar os dados para excluir pontos com resíduos altos (Após a remoção sobraram 243 pontos)
df_filtrado = df[(df['residuos'].values < 50) & (df['residuos'].values > -50)]

# Remover a coluna de resíduos
df_filtrado = df_filtrado.drop(columns = ['residuos'])

# print(f'{df_filtrado}\n{len(df_filtrado)}')

# Remover as linhas correspondentes dos dados
df_sem_pi = df_filtrado

n_sem_pi = len(df_sem_pi)

x_sem_pi = df_sem_pi['Nível de Triglicerídeo']
y_sem_pi = df_sem_pi['Progressão da doença']

# Cálculo do x_barra e y_barra dos dados sem os pontos influentes
x_barra_sem_pi = np.mean(df_sem_pi['Nível de Triglicerídeo'])
y_barra_sem_pi = np.mean(df_sem_pi['Progressão da doença'])

# Atribuindo os valores das funções à variáveis
Sxx_sem_pi = calculo_Sxx(x_barra_sem_pi, x_sem_pi)
Syy_sem_pi = calculo_Syy(y_barra_sem_pi, y_sem_pi)
Sxy_sem_pi = calculo_Sxy(x_barra_sem_pi, x_sem_pi, y_barra_sem_pi, y_sem_pi)
coef_corr_sem_pi = calculo_coef_de_correlacao(Sxx_sem_pi, Syy_sem_pi, Sxy_sem_pi)
beta_1_sem_pi = calculo_beta1(Sxx_sem_pi, y_sem_pi, x_sem_pi, x_barra_sem_pi)
beta_0_sem_pi = calculo_beta0(y_barra_sem_pi, x_barra_sem_pi, beta_1_sem_pi)
variancia_ao_quadrado_sem_pi = Syy / n_sem_pi

# Reta sem os pontos influentes
y_chapeu_sem_pi = beta_0_sem_pi + beta_1_sem_pi * x_sem_pi

# Resíduos sem os pontos influentes
residuos_sem_pi = y_sem_pi - y_chapeu_sem_pi

# Plotar gráfico com os pontos e a reta calculada anteriormente
plt.scatter(df_sem_pi['Nível de Triglicerídeo'], df_sem_pi['Progressão da doença'])
plt.plot(x_sem_pi, y_chapeu_sem_pi, color = 'red')
plt.title('Nível de Triglicerídeo x Progressão da doença (sem pontos influentes)')
plt.xlabel('Nível de Triglicerídeo')
plt.ylabel('Progressão da doença')
plt.show()

# TESTE
print(f'\nSxx_sem_pi = {Sxx_sem_pi:.2f}\n\nSyy_sem_pi = {Syy_sem_pi:.2f}\n\nSxy_sem_pi = {Sxy_sem_pi:.2f}\n\n'
      f'x_barra_sem_pi = {x_barra_sem_pi:.2f}\n\ny_barra_sem_pi = {y_barra_sem_pi:.2f}\n\n'
      f'Coeficiente de correlação_sem_pi = {coef_corr_sem_pi:.2f}\n\n'
      f'Beta_0(Coeficiente linear)_sem_pi = {beta_0_sem_pi:.2f}\n\n'
      f'Beta_1(Coeficiente angular)_sem_pi = {beta_1_sem_pi:.2f}\n\n'
      f'Variância ao quadrado_sem_pi = {variancia_ao_quadrado_sem_pi:.2f}\n')

# Cálculo do valor crítico de F
# Calcular os graus de liberdade
dfn = 1  # Graus de liberdade do numerador (número de grupos - 1)
dfd = n - 2  # Graus de liberdade do denominador (número total de observações - número de grupos)

# Nível de significância
alpha = 0.05

# Valor crítico de F
f_critico = f.ppf(1 - alpha, dfn, dfd)

print(f'\nValor crítico de F: {f_critico:.2f}')
