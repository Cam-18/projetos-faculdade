# Esse código calcula a Série de Taylor de uma função matemática que o usuário digitar

import sympy as sp

def taylor_series(funcao, x0 = 0, ordem = 5):
    x = sp.symbols('x')

    serie_taylor = sp.series(funcao, x, x0, ordem).removeO()

    derivadas = [sp.diff(funcao, x, i).subs(x, x0) for i in range(ordem)]
    try:
        rc = sp.limit(sp.Abs(derivadas[ordem - 1]) ** (1 / ordem), x, sp.oo)
    except:
        rc = sp.oo

    ri = rc

    return serie_taylor, rc, ri

entrada = input('Digite a função de x: ')
x = sp.symbols('x')
funcao = sp.sympify(entrada)

a = float(input('Digite o valor de a (ponto de expansão): '))

serie, raio_convergencia, raio_intervalo = taylor_series(funcao, x0 = a)

print(f'\nSérie de Taylor em torno de x = {a}: {serie}')
print(f'Raio de Convergência (RC): {raio_convergencia}')
print(f'Raio de Intervalo (RI): {raio_intervalo}')
