import numpy as np
import matplotlib.pyplot as plt

# Configuração dos gráficos
plt.figure(figsize = (15, 5))

# 1. y = cos(arccos(x)), x ∈ [-1, 1]
x1 = np.linspace(-1, 1, 400)
y1 = np.cos(np.arccos(x1))

plt.subplot(1, 3, 1)
plt.plot(x1, y1, color = 'blue')
plt.title('y = cos(arccos(x))')
plt.xlabel('x')
plt.ylabel('y')

# 2. y = arccos(cos(x)), x ∈ [0, π]
x2 = np.linspace(0, np.pi, 400)
y2 = np.arccos(np.cos(x2))

plt.subplot(1, 3, 2)
plt.plot(x2, y2, color = 'red')
plt.title('y = arccos(cos(x)), x ∈ [0, π]')
plt.xlabel('x')
plt.ylabel('y')

# 3. y = arccos(cos(x)), x ∈ [0, 4π]
x3 = np.linspace(0, 4 * np.pi, 1000)
y3 = np.arccos(np.cos(x3))

plt.subplot(1, 3, 3)
plt.plot(x3, y3, color = 'green')
plt.title('y = arccos(cos(x)), x ∈ [0, 4π]')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
