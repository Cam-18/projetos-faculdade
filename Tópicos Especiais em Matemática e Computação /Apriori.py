# Análise de regras de associação usando o algoritmo Apriori com dados de exemplo e com dados reais de vendas

from apyori import apriori
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori as ap

# 1
dataset = \
[['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# min_support: Considera apenas itens que aparecem em pelo menos 10% das transações
# min_confidence: Gera regras com pelo menos 10% de confiança
regras = apriori(dataset, min_support = 0.1, min_confidence = 0.1)

resultados = list(regras)

print(f'{resultados[0]}\n')

# 2
sales_october = pd.read_csv('Sales_October_2019.csv')

# Pré-processamento dos dados
sales_filter = sales_october[['Order ID', 'Product']]
sales_filter = sales_filter.reset_index()

# Transformação para formato one-hot
df_orders = sales_filter.pivot_table(index = 'Order ID', columns = 'Product', aggfunc = 'count')
df_orders = df_orders.fillna(0)

# Conversão para valores booleanos
col_names = df_orders.columns
df_orders = df_orders[col_names].astype('bool')

# Aplicando Apriori
frequent_itemsets = ap(df_orders, min_support = 0.001, max_len = 4, use_colnames = True)
frequent_itemsets_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)

# Mostra o df inteiro
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(f'{frequent_itemsets_rules}')
