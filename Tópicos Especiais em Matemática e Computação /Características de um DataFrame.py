# Características de um DataFrame

import pandas as pd

# 1: Cria e printa o DataFrame
df = pd.DataFrame({'Produto': ['Chocolate', 'Banana'],'Quantidade': [200, 80], 'Preço': [3.00, 0.50]})
print(f'DataFrame:\n{df}\n\n')

# 2: Calcula o valor total de cada produto
serie = pd.Series(df['Quantidade'] * df['Preço'])
print(f'Série:\n{serie}\n\n')

# 3: Transforma o arquivo csv "vendas" em um pandas DataFrame
vendas = df.to_csv('vendas.csv', index = False)

# 4:
# a) Mostra os nomes das colunas do df
colunas = df.columns
print(f'Colunas:\n{colunas}\n\n')

# b) Mostra a correlação entre os dados do df
corr = df[['Quantidade', 'Preço']].corr()
print(f'Correlação:\n{corr}\n\n')

# c) Soma as quantidades e o preço dos produtos
soma = df[['Quantidade', 'Preço']].sum()
print(f'Soma:\n{soma}\n\n')

# d) Calcula a média das colunas do df
media = df[['Quantidade', 'Preço']].mean()
print(f'Média:\n{media}\n\n')

# e) Describe retorna várias características do df por exemplo média, máx, min, etc.
describe = df.describe()
print(f'Describe:\n{describe}\n\n')

# 5: Transforma o arquivo csv "ipea_admissoes_caged" em um pandas DataFrame
csv = pd.read_csv('ipea_admissoes_caged.csv')

# a) Calcula a soma das admissões
somaadmi = csv['Admissoes'].sum()
print(f'Soma Admissões:\n{somaadmi}\n\n')

# b) cumsum mostra a soma cumulativa ou seja a soma de cada linha do arquivo até chegar no final.
somacum = csv['Admissoes'].cumsum()
print(f'Soma Cumulativa Admissões:\n{somacum}\n\n')

# c) Mostra o valor mínimo da coluna admissões
minimo = csv['Admissoes'].min()
print(f'Valor Mínimo:\n{minimo}\n\n')

# d) Mostra o valor máximo da coluna data
maximo = csv['Data'].max()
print(f'Maior Data:\n{maximo}\n\n')

# e) Calcula a média das admissões
mediaad = csv['Admissoes'].mean()
print(f'Média Admissões:\n{mediaad}\n\n')

# f) Calcula a mediana das admissões
mediana = csv['Admissoes'].median()
print(f'Mediana Admissoes:\n{mediana}\n\n')

# g) Calcula o desvio padrão das admissões
desviopadrao = csv['Admissoes'].std()
print(f'Desvio Padrão:\n{desviopadrao}\n\n')
