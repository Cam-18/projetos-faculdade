import pandas as pd

# 1:
df = pd.DataFrame({'Produto': ['Chocolate', 'Banana'],'Quantidade': [200, 80], 'Preço': [3.00, 0.50]})
print(f'DataFrame:\n{df}\n\n')

# 2:
serie = pd.Series(df['Quantidade'] * df['Preço'])
print(f'Série:\n{serie}\n\n')

# 3:
vendas = df.to_csv('vendas.csv', index = False)

# 4:
# a)
colunas = df.columns
print(f'Colunas:\n{colunas}\n\n')

# b)
corr = df[['Quantidade', 'Preço']].corr()
print(f'Correlação:\n{corr}\n\n')

# c)
soma = df[['Quantidade', 'Preço']].sum()
print(f'Soma:\n{soma}\n\n')

# d)
media = df[['Quantidade', 'Preço']].mean()
print(f'Média:\n{media}\n\n')

# e) Describe retorna várias características do df por exemplo média, máx, min, etc.
describe = df.describe()
print(f'Describe:\n{describe}\n\n')

# 5:
csv = pd.read_csv('ipea_admissoes_caged.csv')

# a)
somaadmi = csv['Admissoes'].sum()
print(f'Soma Admissões:\n{somaadmi}\n\n')

# b) cumsum mostra a soma cumulativa ou seja a soma de cada linha do arquivo até chegar no final.
somacum = csv['Admissoes'].cumsum()
print(f'Soma Cumulativa Admissões:\n{somacum}\n\n')

# c)
minimo = csv['Admissoes'].min()
print(f'Valor Mínimo:\n{minimo}\n\n')

# d)
maximo = csv['Data'].max()
print(f'Maior Data:\n{maximo}\n\n')

# e)
mediaad = csv['Admissoes'].mean()
print(f'Média Admissões:\n{mediaad}\n\n')

# f)
mediana = csv['Admissoes'].median()
print(f'Mediana Admissoes:\n{mediana}\n\n')

# g)
desviopadrao = csv['Admissoes'].std()
print(f'Desvio Padrão:\n{desviopadrao}\n\n')
