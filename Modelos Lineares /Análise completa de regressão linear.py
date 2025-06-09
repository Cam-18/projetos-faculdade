# Análise completa de regressão linear, mostrando a relação entre 10 fatores fisiológicos e a progressão da doença

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from sklearn.datasets import load_diabetes
from scipy.stats import f

np.random.seed(9)

dados = load_diabetes(scaled = False)
X = dados.data
y = dados.target

n_colunas = X.shape[1]

# Funções
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

def calculo_beta(x, y):
    # Adicionando coluna de 1's à matriz
    n_linhas = x.shape[0]
    coluna_de_uns = np.ones((n_linhas, 1))
    x_ = np.hstack((coluna_de_uns, x))

    # Obtendo a matriz x transposta
    x_transposta = x_.T

    a = np.linalg.inv(np.linalg.matmul(x_transposta, x_))
    b = np.linalg.matmul(x_transposta, y)

    beta = np.linalg.matmul(a, b)

    return beta

def calculo_y_chapeu(x, beta):
    # Adicionando coluna de 1's
    n_linhas = x.shape[0]
    coluna_de_uns = np.ones((n_linhas, 1))
    x_ = np.hstack((coluna_de_uns, x))

    # Calculando y_chapeu
    y_chapeu = np.linalg.matmul(x_, beta)

    return y_chapeu


# Letra a
# Idade x Progressão da Doença
plt.scatter(X[:, 0], y)
plt.title('Idade x Progressão da doença')
plt.xlabel('Idade')
plt.ylabel('Progressão da doença')
plt.show()

# Sexo x Progressão da Doença
plt.scatter(X[:, 1], y)
plt.title('Sexo x Progressão da doença')
plt.xlabel('Sexo')
plt.ylabel('Progressão da doença')
plt.show()

# IMC x Progressão da Doença
plt.scatter(X[:, 2], y)
plt.title('IMC x Progressão da doença')
plt.xlabel('IMC')
plt.ylabel('Progressão da doença')
plt.show()

# Pressão Arterial x Progressão da Doença
plt.scatter(X[:, 3], y)
plt.title('Pressão Arterial x Progressão da doença')
plt.xlabel('Pressão Arterial')
plt.ylabel('Progressão da doença')
plt.show()

# s1 x Progressão da Doença
plt.scatter(X[:, 4], y)
plt.title('s1 x Progressão da doença')
plt.xlabel('s1')
plt.ylabel('Progressão da doença')
plt.show()

# s2 x Progressão da Doença
plt.scatter(X[:, 5], y)
plt.title('s2 x Progressão da doença')
plt.xlabel('s2')
plt.ylabel('Progressão da doença')
plt.show()

# s3 x Progressão da Doença
plt.scatter(X[:, 6], y)
plt.title('s3 x Progressão da doença')
plt.xlabel('s3')
plt.ylabel('Progressão da doença')
plt.show()

# s4 x Progressão da Doença
plt.scatter(X[:, 7], y)
plt.title('s4 x Progressão da doença')
plt.xlabel('s4')
plt.ylabel('Progressão da doença')
plt.show()

# s5 x Progressão da Doença
plt.scatter(X[:, 8], y)
plt.title('s5 x Progressão da doença')
plt.xlabel('s5')
plt.ylabel('Progressão da doença')
plt.show()

# s6 x Progressão da Doença
plt.scatter(X[:, 9], y)
plt.title('s6 x Progressão da doença')
plt.xlabel('s6')
plt.ylabel('Progressão da doença')
plt.show()

# Letra b
# Médias
y_barra = np.mean(y)
x0_barra = np.mean(X[:, 0])
x1_barra = np.mean(X[:, 1])
x2_barra = np.mean(X[:, 2])
x3_barra = np.mean(X[:, 3])
x4_barra = np.mean(X[:, 4])
x5_barra = np.mean(X[:, 5])
x6_barra = np.mean(X[:, 6])
x7_barra = np.mean(X[:, 7])
x8_barra = np.mean(X[:, 8])
x9_barra = np.mean(X[:, 9])

# Syy
Syy = calculo_Syy(y_barra, y)

# Sxx
Sxx0 = calculo_Sxx(x0_barra, X[:, 0])
Sxx1 = calculo_Sxx(x1_barra, X[:, 1])
Sxx2 = calculo_Sxx(x2_barra, X[:, 2])
Sxx3 = calculo_Sxx(x3_barra, X[:, 3])
Sxx4 = calculo_Sxx(x4_barra, X[:, 4])
Sxx5 = calculo_Sxx(x5_barra, X[:, 5])
Sxx6 = calculo_Sxx(x6_barra, X[:, 6])
Sxx7 = calculo_Sxx(x7_barra, X[:, 7])
Sxx8 = calculo_Sxx(x8_barra, X[:, 8])
Sxx9 = calculo_Sxx(x9_barra, X[:, 9])

# Sxy
Sxy0 = calculo_Sxy(x0_barra, X[:, 0], y_barra, y)
Sxy1 = calculo_Sxy(x1_barra, X[:, 1], y_barra, y)
Sxy2 = calculo_Sxy(x2_barra, X[:, 2], y_barra, y)
Sxy3 = calculo_Sxy(x3_barra, X[:, 3], y_barra, y)
Sxy4 = calculo_Sxy(x4_barra, X[:, 4], y_barra, y)
Sxy5 = calculo_Sxy(x5_barra, X[:, 5], y_barra, y)
Sxy6 = calculo_Sxy(x6_barra, X[:, 6], y_barra, y)
Sxy7 = calculo_Sxy(x7_barra, X[:, 7], y_barra, y)
Sxy8 = calculo_Sxy(x8_barra, X[:, 8], y_barra, y)
Sxy9 = calculo_Sxy(x9_barra, X[:, 9], y_barra, y)

# Cálculo das correlações
corr0 = calculo_coef_de_correlacao(Sxx0, Syy, Sxy0)
corr1 = calculo_coef_de_correlacao(Sxx1, Syy, Sxy1)
corr2 = calculo_coef_de_correlacao(Sxx2, Syy, Sxy2)
corr3 = calculo_coef_de_correlacao(Sxx3, Syy, Sxy3)
corr4 = calculo_coef_de_correlacao(Sxx4, Syy, Sxy4)
corr5 = calculo_coef_de_correlacao(Sxx5, Syy, Sxy5)
corr6 = calculo_coef_de_correlacao(Sxx6, Syy, Sxy6)
corr7 = calculo_coef_de_correlacao(Sxx7, Syy, Sxy7)
corr8 = calculo_coef_de_correlacao(Sxx8, Syy, Sxy8)
corr9 = calculo_coef_de_correlacao(Sxx9, Syy, Sxy9)

print(f'Correlações:\n{corr0}\n{corr1}\n{corr2}\n{corr3}\n{corr4}\n{corr5}\n{corr6}\n{corr7}\n{corr8}\n{corr9}\n')

# Letra c
det0 = corr0 ** 2
det1 = corr1 ** 2
det2 = corr2 ** 2
det3 = corr3 ** 2
det4 = corr4 ** 2
det5 = corr5 ** 2
det6 = corr6 ** 2
det7 = corr7 ** 2
det8 = corr8 ** 2
det9 = corr9 ** 2

print(f'Determinação:\n{det0}\n{det1}\n{det2}\n{det3}\n{det4}\n{det5}\n{det6}\n{det7}\n{det8}\n{det9}\n')

# Letra d
beta = calculo_beta(X, y)
print(f'Beta:\n{beta}\n')

# Letra e
y_chapeu = calculo_y_chapeu(X, beta)
print(f'y_chapeu:\n{y_chapeu}\n')

# Letra f
residuos = y - y_chapeu
print(f'Resíduos:\n{residuos}\n')

# Letra g
plt.hist(residuos)
plt.title('Histograma dos Resíduos')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.show()
