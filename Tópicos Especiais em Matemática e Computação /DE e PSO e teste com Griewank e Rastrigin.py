# Compara dois algoritmos de otimização (DE e PSO) em duas funções de teste conhecidas (Griewank e Rastrigin)

import benchmark_functions as bf
from scipy.optimize import differential_evolution as de
from pyswarm import pso
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action = 'ignore',category = FutureWarning)

# 1

fx1 = bf.Griewank(n_dimensions = 2)

print(f'DE e PSO da função Griewank:\n')

bounds = [(-100, 100), (-100, 100)]
result = de(fx1, bounds)
print(f'result.x: {result.x}, result.fun: {result.fun}')

lb = [(-100), (-100)]
ub = [(100), (100)]
xopt,fopt = pso(fx1, lb, ub)

print(f'\nxopt: {xopt}, fopt: {fopt}')

# 2

fx2 = bf.Rastrigin(n_dimensions = 2)

print(f'\n\nDE e PSO da função Rastrigin:\n')

bounds = [(-10, 10), (-10, 10)]
result = de(fx2, bounds)
print(f'result.x: {result.x}, result.fun: {result.fun}')

lb = [(-10), (-10)]
ub = [(10), (10)]
xopt,fopt = pso(fx2, lb, ub)

print(f'\nxopt: {xopt}, fopt: {fopt}')
