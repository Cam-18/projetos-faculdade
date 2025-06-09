# Processo de oversampling (equaliza o número de amostras nas classes para lidar com o desbalanceamento do dataset)

import pandas as pd
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

df = pd.DataFrame(X)
df['target'] = y

linhaTarget1, linhaTarget0 = df.target.value_counts() # Conta quantas amostras tem nas colunas target

# Separa por classes
dfT0 = df[df['target'] == 0] # Maligno
dfT1 = df[df['target'] == 1] # Benigno

# Oversampling
max_samples = max(linhaTarget0, linhaTarget1) # Determina o número de amostras da classe majoritária
# Oversampling nas duas classes para igualar ao número da classe majoritária
dfT0_resampled = dfT0.sample(max_samples, replace = True)
dfT1_resampled = dfT1.sample(max_samples, replace = True)

# Combina as duas classes após o oversampling
df_resampled = pd.concat([dfT0_resampled, dfT1_resampled])
df_resampled = df_resampled.sample(frac = 1).reset_index(drop = True)

# Mostra a contagem de amostras por classe após o oversampling
target_count_resampled = df_resampled.target.value_counts()
print(target_count_resampled)
