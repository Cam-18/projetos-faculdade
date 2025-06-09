# Esse código verifica e transforma uma matriz quadrada em uma matriz diagonalmente dominante

import numpy as np
from itertools import permutations

def diag_dom(matriz):
    n = len(matriz)

    for i in range(n):
        soma_linha = sum(abs(matriz[i][j]) for j in range(n) if j != i)

        if abs(matriz[i][i]) < soma_linha:
            return False

    return True


def perm_valida(matriz):
    n = len(matriz)

    for perm in permutations(range(n)):
        matriz_permutada = matriz[list(perm), :]

        if diag_dom(matriz_permutada):
            return matriz_permutada

    return None


n = int(input('Indique a ordem da matriz: '))

matriz = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        matriz[i][j] = float(input(f'Digite o elemento da posição ({i + 1},{j + 1}): '))

print('\n')

if diag_dom(matriz):
    print(f'A matriz original já atende ao critério de divergência.\n{matriz}')

else:
    print('A matriz original não atende ao critério de divergência.')
    matriz_permutada = perm_valida(matriz)

    if matriz_permutada is not None:
        print(f'Uma permutação que atende ao critério de divergência foi encontrada:\n{matriz_permutada}')

    else:
        print('Nenhuma permutação atende ao critério de divergência.')
