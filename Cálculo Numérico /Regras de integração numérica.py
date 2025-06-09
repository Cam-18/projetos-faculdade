# Implementação das regras do trapézio e trapézio repetida e simpson e simpson repetida

import re
import math
import numpy as np

# Regra do trapézio e trapézio repetida
def trapezio(f, a, b, n):
    h = (b - a) / n # Calcula o valor de h
    soma = (f(a) + f(b)) / 2 # Soma dos valores das funções de acordo com a fórmula

    # Se o número de partições for maior que 1 o somatório da fórmula do trapézio repetida é resolvido
    for i in range(1, n):
        soma += f(a + i * h)

    return h * soma # Retorna o valor de h vezes o somatório

# Regra de simpson e simpson repetida
def simpson(f, a, b, n):
    # Se certifica de que n vai ser par
    if n % 2:
        n += 1 # Se não for par soma 1 a n

    h = (b - a) / n # Calcula o valor de h
    soma = f(a) + f(b) # Soma dos valores das funções de acordo com a fórmula

    # Resolve os somatórios da fórmula
    # Para os índices pares
    for i in range(1, n, 2):
        soma += 4 * f(a + i * h)

    # Para os índices ímpares
    for i in range(2, n - 1, 2):
        soma += 2 * f(a + i * h)

    return h * soma / 3


# Função para validar a expressão da função do usuário
def valida_funcao(funcao_usuario):
    permitido = re.compile(r'^[0-9x+\-*/(). \^math\.\w+]+$')
    if permitido.match(funcao_usuario):
        funcao_usuario = funcao_usuario.replace('^', '**')
        funcao_usuario = funcao_usuario.replace('sqrt', 'math.sqrt')
        return funcao_usuario
    else:
        raise ValueError('Função inválida! Use apenas números, operadores matemáticos, "math" e "sqrt".')

# Main
print('\nEscolha a regra de integração:')
print('1: Trapézio')
print('2: Simpson\n')
opcao = int(input('Digite o número da regra que deseja usar: '))

# Solicita a função do usuário
funcao_usuario = input('Digite a função: ')

try:
    funcao_usuario = valida_funcao(funcao_usuario) # Utiliza a função para interpretar a função fornecida

    def funcao(x):
        return eval(funcao_usuario) # Resolve a função fornecida

except ValueError as e: # A não ser que a função seja inválida. Se for, a razão de ser inválida é printada
    print(e)
    exit()

# Solicita os valores de a, b e n
a = float(input('Digite o valor de a: '))
b = float(input('Digite o valor de b: '))
n = int(input('Digite o número de partições n: '))

# Analisa a opção escolhida e o valor de n para printar a regra correta
if opcao == 1 and n == 1:
    print('\nTrapézio: ', trapezio(funcao, a, b, n))

elif opcao == 1 and n > 1:
    print('\nTrapézio Repetida: ', trapezio(funcao, a, b, n))

elif opcao == 2 and n == 1:
    print('\nSimpson: ', simpson(funcao, a, b, n))

elif opcao == 2 and n > 1:
    print('\nSimpson Repetida: ', simpson(funcao, a, b, n))

else:
    print('\nOpção inválida')
