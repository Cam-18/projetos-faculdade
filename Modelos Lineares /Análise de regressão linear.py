# Simulação e análise de regressão linear para modelar a relação entre horas de estudo e notas de alunos

import matplotlib.pyplot as plt
import numpy
import pandas
import statsmodels.api as sm
from tabulate import tabulate

# Função que gera dados para simulação
# Função simula a relação entre as horas de estudo e a nota esperada.
def f(x):
    m = 1.5  # Coeficiente angular (slope)
    b = 2.5  # Coeficiente linear (intercept)
    return m * x + b

def gera_notas(horas_estudo, media_erro = 0, desvio_erro = 4):
    # Gera as notas simuladas
    nota = f(horas_estudo) + numpy.random.normal(media_erro, desvio_erro, len(horas_estudo))

    # Normaliza as notas finais para o intervalo [0, 10]
    nota_normalizada = (nota - min(nota)) / (max(nota) - min(nota)) * 10

    return nota_normalizada


# Modelo Linear
numpy.random.seed(7)
n_obs = 50
horas_estudo = numpy.random.uniform(0, 8, n_obs)
nota = gera_notas(horas_estudo, media_erro = 0, desvio_erro = 4)

# Cria o dataframe com todos os dados
dados = pandas.DataFrame({'Horas': horas_estudo, 'Nota': nota})

X1 = dados[['Horas']]
X1 = sm.add_constant(X1)  # Adiciona uma constante
y1 = dados['Nota']

model1 = sm.OLS(y1, X1).fit() # Cria um modelo de regressão linear e o ajusta (fit) aos dados fornecidos.
sumario1 = model1.summary() # Gera um sumário detalhado do modelo ajustado
intercept1, slope1 = model1.params['const'], model1.params['Horas'] # Extrai os coeficientes estimados do modelo
fit1 = intercept1, slope1 # Armazena os valores do intercepto e do coeficiente angular na tupla fit1

# Gráfico com os resultados
yl = fit1[0] + fit1[1] * horas_estudo # Coeficiente angular + coeficiente linear * x calcula o y_hat que são os valores preditores

plt.scatter(horas_estudo, nota, color = 'blue', label = 'Notas')
plt.plot(horas_estudo, yl, color = 'red', label = 'Reta de Regressão')

for i in range(len(horas_estudo)):
    plt.vlines(x = horas_estudo[i], ymin = yl[i], ymax = nota[i], color = 'black', linestyle = 'dotted')

plt.xlabel('Horas de estudo')
plt.ylabel('Nota')
plt.legend()
plt.show()


# Calcular o erro de cada nota pra reta
erros = numpy.abs(y1 - yl)
dfErros = pandas.DataFrame({'Erro': erros})

print(f'Maior erro com a melhor reta: {dfErros.max()}\n')
print(f'Menor erro com a melhor reta: {dfErros.min()}\n')

# O yl é o ponto que os valores de y1 encostam na reta de regressão

print(f'Erros:\n{erros}')

# Gráfico com uma reta genérica
yl1 = 5 + fit1[1] * horas_estudo

plt.scatter(horas_estudo, nota, color = 'blue', label = 'Notas')
plt.plot(horas_estudo, yl1, color = 'red', label = 'Reta de Regressão')

for i in range(len(horas_estudo)):
    plt.vlines(x = horas_estudo[i], ymin = yl1[i], ymax = nota[i], color = 'black', linestyle = 'dotted')

plt.xlabel('Horas de estudo')
plt.ylabel('Nota')
plt.legend()
plt.show()

# Cálculo dos erros com a reta genérica
erros_gen = numpy.abs(y1 - yl1)
dfErros_gen = pandas.DataFrame({'Erro': erros_gen})

print(f'Maior erro com a reta genérica: {dfErros_gen.max()}\n')
print(f'Menor erro com a reta genérica: {dfErros_gen.min()}\n\n')

print(f'Erros reta genérica:\n{erros_gen}')

df_comparacao = pandas.concat([dfErros, dfErros_gen], axis = 1, ignore_index = True)

print(df_comparacao)
