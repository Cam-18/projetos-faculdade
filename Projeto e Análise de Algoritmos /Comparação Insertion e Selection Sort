import matplotlib.pyplot as plt
import time
import random
import copy

random.seed(9)

# Função para criação dos vetores
def gera_vetor(tamanho, tipo_ord):
    if tipo_ord == 1: # Cria vetor ordenado de forma crescente
        return list(range(1, tamanho + 1))
    elif tipo_ord == 2: # Cria vetor ordenado de forma decrescente
        return list(range(tamanho, 0, -1))
    elif tipo_ord == 3: # Cria vetor aleatório
        return random.sample(range(1, tamanho + 1), tamanho)


# Selection e Insertion Sort

def insertion_sort(vet):
    n = len(vet)  # Tamanho do vetor

    for k in range(1, n):  # for que vai percorrer todo o vetor
        y = vet[k]  # Guarda o número de vet[k] em y
        i = k - 1  # Inicializa a variável i com a parte ja ordenada do vetor
        while i >= 0 and vet[i] > y:  # Enquanto não chegar no início e vet[i] for maior que y
            vet[i + 1] = vet[i]  # Desloca X[i] para a direita
            i -= 1  # Decrementa i para comparar com o próximo elemento à esquerda
        vet[i + 1] = y  # Coloca y na posição correta
    return vet  # Retorna o vetor ordenado


def selection_sort(vet):
    n = len(vet)  # Tamanho do vetor

    for i in range(n - 1):  # Percorre o vetor até o penúltimo elemento
        menor = vet[i]  # Assume o elemento atual como menor a fim de comparação
        index = i  # Guarda a posição desse elemento
        for j in range(i + 1, n):  # Percorre os elementos não ordenados
            if vet[j] < menor:  # Se o elemento j for menor que o menor
                menor = vet[j]  # Troca e o elemento j passa a ser o novo menor valor
                index = j  # Posição do novo menor valor
        vet[index] = vet[i]  # Troca o elemento na posição index com o elemento na posição i
        vet[i] = menor  # Coloca o menor valor encontrado na posição correta
    return vet  # Retorna o vetor ordenado


# Criação dos vetores
tam = [50, 500, 5000, 50000]
crescente = []
decrescente = []
aleatorio = []

for i in range(4):
    crescente.append(gera_vetor(tam[i], 1))   # Crescente
    decrescente.append(gera_vetor(tam[i], 2)) # Decrescente
    aleatorio.append(gera_vetor(tam[i], 3))   # Aleatório

# Dicionários com os tempos de ordenação
temposCrescente = {
    'Selection': [],
    'Insertion': []
}

temposDecrescente = {
    'Selection': [],
    'Insertion': []
}

temposAleatorio = {
    'Selection': [],
    'Insertion': []
}

# Ordenação
for x in range(4):
    # Cria cópias para o Selection Sort
    vetor_crescente_selection = copy.deepcopy(crescente[x])
    vetor_decrescente_selection = copy.deepcopy(decrescente[x])
    vetor_aleatorio_selection = copy.deepcopy(aleatorio[x])

    # Cria cópias para o Insertion Sort
    vetor_crescente_insertion = copy.deepcopy(crescente[x])
    vetor_decrescente_insertion = copy.deepcopy(decrescente[x])
    vetor_aleatorio_insertion = copy.deepcopy(aleatorio[x])

    # Selection
    # Vetor crescente
    t = time.perf_counter()
    selection_sort(vetor_crescente_selection)
    temposCrescente['Selection'].append(time.perf_counter() - t)

    # Vetor decrescente
    t = time.perf_counter()
    selection_sort(vetor_decrescente_selection)
    temposDecrescente['Selection'].append(time.perf_counter() - t)

    # Vetor aleatório
    t = time.perf_counter()
    selection_sort(vetor_aleatorio_selection)
    temposAleatorio['Selection'].append(time.perf_counter() - t)

    # Insertion
    # Vetor crescente
    t = time.perf_counter()
    insertion_sort(vetor_crescente_insertion)
    temposCrescente['Insertion'].append(time.perf_counter() - t)

    # Vetor decrescente
    t = time.perf_counter()
    insertion_sort(vetor_decrescente_insertion)
    temposDecrescente['Insertion'].append(time.perf_counter() - t)

    # Vetor aleatório
    t = time.perf_counter()
    insertion_sort(vetor_aleatorio_insertion)
    temposAleatorio['Insertion'].append(time.perf_counter() - t)


# Gráfico Crescente
plt.plot(tam, temposCrescente['Selection'], label = 'Selection')
plt.plot(tam, temposCrescente['Insertion'], label = 'Insertion')
plt.title('Vetor Crescente')
plt.xlabel('Tamanho do vetor')
plt.ylabel('Tempo de ordenação')
plt.legend()
plt.show()

# Gráfico decrescente
plt.plot(tam, temposDecrescente['Selection'], label = 'Selection')
plt.plot(tam, temposDecrescente['Insertion'], label = 'Insertion')
plt.title('Vetor Decrescente')
plt.xlabel('Tamanho do vetor')
plt.ylabel('Tempo de ordenação')
plt.legend()
plt.show()

# Gráfico aleatório
plt.plot(tam, temposAleatorio['Selection'], label = 'Selection')
plt.plot(tam, temposAleatorio['Insertion'], label = 'Insertion')
plt.title('Vetor Aleatório')
plt.xlabel('Tamanho do vetor')
plt.ylabel('Tempo de ordenação')
plt.legend()
plt.show()
