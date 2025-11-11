**O projeto foi desenvolvido por Ariadne Evangelista e Arthur Queiroz**  
**DCA3702 – Algoritmos e Estruturas de Dados II**

# Projeto 2: Comparação de Performance: Dijkstra Clássico vs Dijkstra com Min-Heap

<p align="center">
  <img src="./imagem/Capa2.png" alt="Comparação de Tempo Médio de Execução por Tamanho do Grafo"><br>
</p>
<p align="center">

---

## 1. Objetivo
<p align = "justify"> O objetivo é avaliar tempo de execução e pegada de carbono (CO₂) das seguintes abordagens de caminho mínimo: 
  
  - Dijkstra Clássico (O(V² + E))
  - Dijkstra com Min-Heap (O((V + E) * log V)).
    
Reutilizando os códigos já implementados em sala de aula (**dijsktra.ipynb** e **dijsktra_min_heap.ipynb**).</p>

## 2. Preparação do Ambiente

<p align = "justify"> Utilizamos o Google Colab para realização dos testes. Para preparação do ambiente foi necessário instalar o "Codecarbon", "Pandas", "Pandas matplotlib scipy" e importar as seguintes bibliotecas: </p>

```
import networkx as nx
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from codecarbon import EmissionsTracker
```

<p align = "justify">Em seguida, realizamos a fixação das sementes (SEED) para os geradores de números aleatórios (random e numpy). Este passo é fundamental para garantir a reprodutibilidade do experimento: ao usar sempre a mesma semente, asseguramos que a geração dos grafos e a seleção dos nós de partida aleatórios sejam idênticas em todas as execuções do notebook. Isso permite uma comparação e análise mais consistentes dos resultados, conforme fizemos abaixo:</p>

```
SEED = 30
random.seed(SEED)
np.random.seed(SEED)
```

<p align = "justify">Conforme definido nos parâmetros do experimento, incluímos grafos com tamanhos até 50.000 nós, com 15 repetições por tamanho. A execução completa destes testes levou aproximadamente 3h40min. Optamos por limitar o tamanho máximo a 50.000 nós, excluindo o tamanho de 100.000 sugerido na proposta, devido ao tempo estimado de execução consideravelmente maior (aproximadamente o dobro), o que poderia impactar a viabilidade de execução no ambiente do Google Colab, especialmente considerando os limites de tempo de sessão, além disso era necessário uma quantidade de memória maior. Então nosso parâmetros ficaram da seguinte forma:</p>

```
# Define os parâmetros do experimento
GRAPH_SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
REPETITIONS = 15
NUM_SOURCES = 5  # 5 nós aleatórios

# Parâmetro de densidade para grafo GNP
# p=0.01 - cria um grafo esparso e conectado para N grande
PROBABILITY = 0.01

results_data = []

print("\n[INFO] Configuração Inicial concluída. Seeds fixadas e bibliotecas importadas.")
```

## 3. Utilização da Classe MinHeap
<p align = "justify">A codificação do MinHeap foi reutilizado das aulas anteriores de Estrutura de Dados II, conforme solicitado pelo professor.</p>

```
class MinHeap:
    def __init__(self, array):
        self.vertexMap = {idx: idx for idx in range(len(array))}
        self.heap = self.buildHeap(array)

    def isEmpty(self):
        return len(self.heap) == 0

    def buildHeap(self, array):
        firstParentIdx = (len(array) - 2) // 2
        for currentIdx in reversed(range(firstParentIdx + 1)):
            self.siftDown(currentIdx, len(array) - 1, array)
        return array

    def siftDown(self, currentIdx, endIdx, heap):
        childOneIdx = currentIdx * 2 + 1
        while childOneIdx <= endIdx:
            childTwoIdx = currentIdx * 2 + 2 if currentIdx * 2 + 2 <= endIdx else -1

            # Compara a distância (índice 1 do par [vértice, distância])
            if childTwoIdx != -1 and heap[childTwoIdx][1] < heap[childOneIdx][1]:
                idxToSwap = childTwoIdx
            else:
                idxToSwap = childOneIdx

            if heap[idxToSwap][1] < heap[currentIdx][1]:
                self.swap(currentIdx, idxToSwap, heap)
                currentIdx = idxToSwap
                childOneIdx = currentIdx * 2 + 1
            else:
                return

    def siftUp(self, currentIdx, heap):
        parentIdx = (currentIdx - 1) // 2
        # Compara a distância (índice 1)
        while currentIdx > 0 and heap[currentIdx][1] < heap[parentIdx][1]:
            self.swap(currentIdx, parentIdx, heap)
            currentIdx = parentIdx
            parentIdx = (currentIdx - 1) // 2

    def remove(self):
        if self.isEmpty():
            return None
        self.swap(0, len(self.heap) - 1, self.heap)
        vertex, distance = self.heap.pop()
        self.vertexMap.pop(vertex)
        self.siftDown(0, len(self.heap) - 1, self.heap)
        return vertex, distance

    def swap(self, i, j, heap):
        self.vertexMap[heap[i][0]] = j
        self.vertexMap[heap[j][0]] = i
        heap[i], heap[j] = heap[j], heap[i]

    def update(self, vertex, value):
        idx = self.vertexMap[vertex]
        self.heap[idx] = (vertex, value)
        self.siftUp(idx, self.heap)

print("\n[INFO]Iniciando o experimento de comparação de desempenho...")

TRACK_CO2 = True # Ativa medição das emissões de CO2 durante a execução de cada algoritmo
```
## 4. Gerador de Grafos
<p align = "justify"> A função generate_weighted_graph cria grafos aleatórios ponderados e direcionados para experimentos de Dijkstra. Ela começa gerando um grafo aleatório GNP com N nós e probabilidade p de arestas. Em seguida, adiciona pesos aleatórios às arestas e cria arestas reversas com o mesmo peso para simular bidirecionalidade. A função garante a conectividade selecionando o maior componente conectado, se necessário. Por fim, converte o grafo para uma lista de adjacência e retorna o grafo NetworkX, a lista de adjacência, o número real de nós (N_final) e um mapa de nós. Essencialmente, ela prepara grafos adequados para testar e comparar os algoritmos de Dijkstra.</p>

```
def generate_weighted_graph(N, p):

    # Geração do Grafo Aleatório (GNP)
    G = nx.gnp_random_graph(n=N, p=p)

    # Adiciona pesos inteiros positivos
    for u, v in list(G.edges()):
        G[u][v]['weight'] = random.randint(1, 10)
        # Garante que seja direcionado para o Dijkstra, duplicando a aresta com o mesmo peso
        G.add_edge(v, u, weight=G[u][v]['weight'])

    # Garante conectividade
    # Usamos o maior componente conectado se o grafo não for totalmente conectado
    if N > 1 and not nx.is_connected(G.to_undirected()):
        largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
        G = G.subgraph(largest_cc).copy()

    # Atualiza N para o tamanho real do grafo conectado, se necessário
    N_final = len(G.nodes)

    # Converte para Lista de Adjacência
    # A lista deve ter o tamanho N_final, com o índice = nó
    adjacency_list = [[] for _ in range(N_final)]

    # Garante que os nós no grafo networkx sejam remapeados para 0..N-1 se necessário
    node_map = {node: i for i, node in enumerate(G.nodes())}

    for u, v, data in G.edges(data=True):
        u_mapped = node_map[u]
        v_mapped = node_map[v]
        weight = data['weight']
        adjacency_list[u_mapped].append([v_mapped, weight])

    return G, adjacency_list, N_final, node_map
```

## 5. Implementação Clássica do Algoritmo de Dijkstra

<p align = "justify"> A função "getVertexWithMinDistance(distances, visited)" é uma função auxiliar usada pelo algoritmo clássico. O trabalho dela é encontrar, entre todos os vértices que ainda não foram visitados, aquele que possui a menor distância conhecida a partir do nó de partida. Ela faz isso percorrendo a lista completa de distâncias e verificando quais vértices ainda não estão no conjunto.</p>

```
# Função Auxiliar O(V) para o Dijkstra Clássico
def getVertexWithMinDistance(distances, visited):
    currentMinDistance = float("inf")
    vertex = -1

    # Busca linear em todos os vértices não visitados
    for vertexIdx, distance in enumerate(distances):
        if vertexIdx in visited:
            continue

        if distance <= currentMinDistance:
            vertex = vertexIdx
            currentMinDistance = distance

    return vertex, currentMinDistance
```

<p align = "justify"> A função "dijkstrasAlgorithm_Classic(start, edges)" é a implementação principal do algoritmo de Dijkstra Clássico. Ela calcula as distâncias mínimas do nó de partida para todos os outros nós no grafo representado pela lista de adjacência.</p>

```
# Dijkstra Clássico: O(V^2 + E)
def dijkstrasAlgorithm_Classic(start, edges):
    numberOfVertices = len(edges)
    minDistances = [float("inf") for _ in range(numberOfVertices)]
    minDistances[start] = 0
    visited = set()

    while len(visited) != numberOfVertices:
        # Busca O(V)
        vertex, currentMinDistance = getVertexWithMinDistance(minDistances, visited)

        if currentMinDistance == float("inf"):
            break

        visited.add(vertex)

        for edge in edges[vertex]:
            destination, distanceToDestination = edge

            if destination in visited:
                continue

            newPathDistance = currentMinDistance + distanceToDestination
            currentDestinationDistance = minDistances[destination]

            if newPathDistance < currentDestinationDistance:
                minDistances[destination] = newPathDistance

    # Substitui 'inf' por -1
    return list(map(lambda x: -1 if x == float("inf") else x, minDistances))
```

## 6. Implementação do Algoritmo de Dijkstra Otimizado que utiliza um Min-Heap

<p align = "justify"> A função dijkstrasAlgorithm_MinHeap(start, edges) calcula as distâncias mínimas do nó de partida (start) para todos os outros nós no grafo, assim como a versão clássica. No entanto, ela faz isso de uma maneira muito mais eficiente.</p>

<p align = "justify"> A principal diferença é que, em vez de fazer uma busca linear em todos os vértices não visitados para encontrar o de menor distância, esta versão usa a classe MinHeap que definimos anteriormente. O Min-Heap armazena os vértices e suas distâncias conhecidas e permite encontrar e remover o vértice com a menor distância de forma muito rápida (em tempo logarítmico, O(log V)). </p>

```
# Dijkstra Otimizado: O((V + E) * log(V))
def dijkstrasAlgorithm_MinHeap(start, edges):
    numberOfVertices = len(edges)
    minDistances = [float("inf") for _ in range(numberOfVertices)]
    minDistances[start] = 0

    # Inicializa o MinHeap
    minDistancesHeap = MinHeap([(idx, float("inf")) for idx in range(numberOfVertices)])
    minDistancesHeap.update(start, 0) # Atualiza a distância inicial no heap [25]

    while not minDistancesHeap.isEmpty():
        # Extrai o Mínimo (O(log V))
        vertex, currentMinDistance = minDistancesHeap.remove()

        if currentMinDistance == float("inf"):
            break

        for edge in edges[vertex]:
            destination, distanceToDestination = edge

            newPathDistance = currentMinDistance + distanceToDestination
            currentDestinationDistance = minDistances[destination]

            if newPathDistance < currentDestinationDistance:
                minDistances[destination] = newPathDistance
                # Atualiza o heap (O(log V))
                minDistancesHeap.update(destination, newPathDistance)

    # Converte 'inf' para -1
    return list(map(lambda x: -1 if x == float("inf") else x, minDistances))
```

## 7. Experimento
<p align = "justify"> Vamos executar o experimento para cada tamanho de grafo separadamente na seguinte sequência [100, 500, 1000, 5000, 10000, 50000, 100000]. Devido a algumas limitações no colab e de GPU, adaptamos o código para poder verificar o tempo de execução e salvar os registros coletados assim que terminar cada execução em um arquivo .csv e também nos limitamos ao tamano máximo de 50 mil, fizemos isso seguindo a seguinte estrutura de codificação: </p>

```
# Certifique-se de que results_data esteja inicializado se estiver executando pela primeira vez
if 'results_data' not in globals():
    results_data = []

# --- Executar para N = 100 ---

N = 100 # ALTERAMOS O VALOR PARA CADA TAMANHO N

print(f"\n--- Processando grafos com N={N} nós (Repetições: {REPETITIONS}) ---")

start_time_size = time.time() # Capitura do tempo

# A. Geração do Grafo e Lista de Adjacência
G_nx, adj_list, N_actual, node_map = generate_weighted_graph(N, PROBABILITY)

# Garante que as fontes sejam selecionadas a partir dos nós existentes (0 até N_actual - 1)
all_nodes = list(range(N_actual))

for rep in range(REPETITIONS):
    # B. Seleção de 5 Nós de Partida Aleatórios
    source_nodes = random.sample(all_nodes, k=NUM_SOURCES)

    for start_node in source_nodes:
        # 1. DIJKSTRA CLÁSSICO (O(V^2 + E))
        algo_name = 'Dijkstra Clássico (V^2)'
        tracker_classic = EmissionsTracker(output_dir=".", save_to_file=False, log_level="error")
        if TRACK_CO2: tracker_classic.start()
        start_time = time.time()
        dists_classic = dijkstrasAlgorithm_Classic(start_node, adj_list)
        end_time = time.time()
        if TRACK_CO2: emissions_classic = tracker_classic.stop()
        else: emissions_classic = 0.0
        results_data.append({
            'Tamanho_N': N_actual, 'Algoritmo': algo_name, 'Repeticao': rep,
            'No_Fonte': start_node, 'Tempo_s': end_time - start_time, 'CO2_Emissoes_kg': emissions_classic
        })

        # 2. DIJKSTRA COM MIN-HEAP (O((V+E) log V))
        algo_name = 'Dijkstra Min-Heap ((V+E)logV)'
        tracker_heap = EmissionsTracker(output_dir=".", save_to_file=False, log_level="error")
        if TRACK_CO2: tracker_heap.start()
        start_time = time.time()
        dists_heap = dijkstrasAlgorithm_MinHeap(start_node, adj_list)
        end_time = time.time()
        if TRACK_CO2: emissions_heap = tracker_heap.stop()
        else: emissions_heap = 0.0
        results_data.append({
            'Tamanho_N': N_actual, 'Algoritmo': algo_name, 'Repeticao': rep,
            'No_Fonte': start_node, 'Tempo_s': end_time - start_time, 'CO2_Emissoes_kg': emissions_heap
        })

        # 3. REFERÊNCIA NETWORKX
        algo_name = 'NetworkX (Referência)'
        start_node_original = [k for k, v in node_map.items() if v == start_node][0]
        tracker_nx = EmissionsTracker(output_dir=".", save_to_file=False, log_level="error")
        if TRACK_CO2: tracker_nx.start()
        start_time = time.time()
        dists_nx_dict = nx.shortest_path_length(G_nx, source=start_node_original, weight='weight')
        dists_nx = [-1] * N_actual
        for node, dist in dists_nx_dict.items():
             dists_nx[node_map[node]] = dist
        end_time = time.time()
        if TRACK_CO2: emissions_nx = tracker_nx.stop()
        else: emissions_nx = 0.0
        results_data.append({
            'Tamanho_N': N_actual, 'Algoritmo': algo_name, 'Repeticao': rep,
            'No_Fonte': start_node, 'Tempo_s': end_time - start_time, 'CO2_Emissoes_kg': emissions_nx
        })

end_time_size = time.time() # Tempo
print(f"\nColeta de dados para N={N} concluída. Total de resultados coletados até agora: {len(results_data)}")
print(f"Tempo total para N={N}: {end_time_size - start_time_size:.2f} segundos")

# Salvar os resultados parciais após cada tamanho de grafo
df_partial = pd.DataFrame(results_data)
output_partial_filename = '/content/drive/MyDrive/Trabalho AED2 - 3pts/dijkstra_partial_results.csv'
df_partial.to_csv(output_partial_filename, index=False)
print(f"Resultados parciais salvos em: {output_partial_filename}")
```
<p align = "justify"> 
Na codificação acima utilizamos o exemplo N = 100, mas fizemos individualmente para cada tamanho N e salvamos no arquivo dijstra_parcial.results.csv e obtivemos os seguintes resultados para cada item.
</p>

```
--- Processando grafos com N=100 nós (Repetições: 15) ---

Coleta de dados para N=100 concluída. Total de resultados coletados até agora: 289
Tempo total para N=100: 307.42 segundos
Resultados parciais salvos em: /content/drive/MyDrive/Trabalho AED2 - 3pts/dijkstra_partial_results.csv

--- Processando grafos com N=500 nós (Repetições: 15) ---

Coleta de dados para N=500 concluída. Total de resultados coletados até agora: 514
Tempo total para N=500: 307.72 segundos
Resultados parciais salvos em: /content/drive/MyDrive/Trabalho AED2 - 3pts/dijkstra_partial_results.csv

--- Processando grafos com N=1000 nós (Repetições: 15) ---

Coleta de dados para N=1000 concluída. Total de resultados coletados até agora: 739
Tempo total para N=1000: 308.89 segundos
Resultados parciais salvos em: /content/drive/MyDrive/Trabalho AED2 - 3pts/dijkstra_partial_results.csv

--- Processando grafos com N=5000 nós (Repetições: 15) ---

Coleta de dados para N=5000 concluída. Total de resultados coletados até agora: 964
Tempo total para N=5000: 400.10 segundos
Resultados parciais salvos em: /content/drive/MyDrive/Trabalho AED2 - 3pts/dijkstra_partial_results.csv

--- Processando grafos com N=10000 nós (Repetições: 15) ---

Coleta de dados para N=10000 concluída. Total de resultados coletados até agora: 1189
Tempo total para N=10000: 727.00 segundos
Resultados parciais salvos em: /content/drive/MyDrive/Trabalho AED2 - 3pts/dijkstra_partial_results.csv

--- Processando grafos com N=50000 nós (Repetições: 15) ---

Coleta de dados para N=50000 concluída. Total de resultados coletados até agora: 1414
Tempo total para N=50000: 11492.87 segundos
Resultados parciais salvos em: /content/drive/MyDrive/Trabalho AED2 - 3pts/dijkstra_partial_results.csv

```

## 8. Tratamento do Arquivo .CSV gerado

<p align = "justify">O arquivo dijkstra_partial_results.csv é a base de dados completa e bruta com todos os resultados registrados de cada teste individual realizado para os diferentes tamanhos de grafo e algoritmos.
Esta seção foca no tratamento e carregamento inicial desses dados. A célula de código abaixo lê este arquivo CSV, exibindo o total de registros coletados e as primeiras linhas para inspeção. Com base neste DataFrame, podemos iniciar as primeiras análises estatísticas e obter impressões preliminares sobre o desempenho de cada algoritmo em diferentes tamanhos. O código desenvolvido para criação do dataFrame estará disponível no Google Colab. </p>

```
Dados carregados com sucesso do arquivo: dijkstra_partial_results.csv
Total de registros de execução: 1414
```
**Primeiras linhas dos resultados**

| Tamanho_N | Algoritmo                     | Repeticao | No_Fonte | Tempo_s    | CO2_Emissoes_kg |
|-----------|-------------------------------|-----------|----------|------------|-----------------|
| 19        | Dijkstra Clássico (V^2)       | 0         | 17       | 0.000044   | 3.384044e-09    |
| 19        | Dijkstra Min-Heap ((V+E)logV) | 0         | 17       | 0.000108   | 3.528197e-08    |
| 19        | NetworkX (Referência)         | 0         | 17       | 0.001353   | 1.769670e-08    |
| 19        | Dijkstra Clássico (V^2)       | 0         | 4        | 0.000069   | 4.190678e-08    |
| 19        | Dijkstra Min-Heap ((V+E)logV) | 0         | 4        | 0.000104   | 3.138990e-08    |
| 19        | NetworkX (Referência)         | 0         | 4        | 0.000174   | 2.966487e-09    |
| 19        | Dijkstra Clássico (V^2)       | 0         | 1        | 0.000076   | 1.142356e-08    |
| 19        | Dijkstra Min-Heap ((V+E)logV) | 0         | 1        | 0.000102   | 2.930584e-09    |
| 19        | NetworkX (Referência)         | 0         | 1        | 0.000216   | 6.588887e-09    |
| 19        | Dijkstra Clássico (V^2)       | 0         | 10       | 0.000042   | 2.458036e-09    |


**Estatísticas Preliminares (Tempo Médio por Tamanho/Algoritmo)**

| Tamanho_N | Algoritmo                     | mean       | std        |
|-----------|-------------------------------|------------|------------|
| 17        | Dijkstra Clássico (V^2)       | 0.000041   | 0.000010   |
| 17        | Dijkstra Min-Heap ((V+E)logV) | 0.000065   | 0.000017   |
| 17        | NetworkX (Referência)         | 0.000141   | 0.000021   |
| 19        | Dijkstra Clássico (V^2)       | 0.000047   | 0.000012   |
| 19        | Dijkstra Min-Heap ((V+E)logV) | 0.000072   | 0.000017   |
| 19        | NetworkX (Referência)         | 0.000259   | 0.000286   |
| 497       | Dijkstra Clássico (V^2)       | 0.001421   | 0.002051   |
| 497       | Dijkstra Min-Heap ((V+E)logV) | 0.000795   | 0.000339   |
| 497       | NetworkX (Referência)         | 0.002546   | 0.001119   |
| 1000      | Dijkstra Clássico (V^2)       | 0.021720   | 0.023490   |
| 1000      | Dijkstra Min-Heap ((V+E)logV) | 0.002546   | 0.001450   |
| 1000      | NetworkX (Referência)         | 0.007522   | 0.001937   |
| 5000      | Dijkstra Clássico (V^2)       | 0.996384   | 0.825201   |
| 5000      | Dijkstra Min-Heap ((V+E)logV) | 0.031514   | 0.025463   |
| 5000      | NetworkX (Referência)         | 0.167773   | 0.041298   |
| 10000     | Dijkstra Clássico (V^2)       | 4.635076   | 2.665729   |
| 10000     | Dijkstra Min-Heap ((V+E)logV) | 0.090822   | 0.053050   |
| 10000     | NetworkX (Referência)         | 0.677998   | 0.152813   |
| 50000     | Dijkstra Clássico (V^2)       | 122.789473 | 77.884429  |
| 50000     | Dijkstra Min-Heap ((V+E)logV) | 0.994985   | 0.913907   |
| 50000     | NetworkX (Referência)         | 19.091260  | 0.990229   |

## 8. Análise e Resumo Estátistico
<p align = "justify"> Para análise, pegamos os dados brutos de todas as execuções, agrupamos por tamanho do grafo e algoritmo, calculamos as médias, desvios-padrão e a margem do intervalo de confiança para o tempo e as emissões de CO₂. O resultado é salvo em um arquivo CSV de resumo dijkstra_analysis_summary.csv.O código desenvolvido para criação do dataFrame estará disponível no Google Colab.</p>

**Resumo Estatístico (Tempo e CO2 Médios):**
| Tamanho_N | Algoritmo                     | mean_time  | std_time  | ci_margin_time |
|-----------|-------------------------------|------------|-----------|----------------|
| 17        | Dijkstra Clássico (V^2)       | 0.000041   | 0.000010  | 0.000002       |
| 17        | Dijkstra Min-Heap ((V+E)logV) | 0.000065   | 0.000017  | 0.000004       |
| 17        | NetworkX (Referência)         | 0.000141   | 0.000021  | 0.000005       |
| 19        | Dijkstra Clássico (V^2)       | 0.000047   | 0.000012  | 0.000005       |
| 19        | Dijkstra Min-Heap ((V+E)logV) | 0.000072   | 0.000017  | 0.000008       |
| 19        | NetworkX (Referência)         | 0.000259   | 0.000286  | 0.000130       |
| 497       | Dijkstra Clássico (V^2)       | 0.001421   | 0.002051  | 0.000472       |
| 497       | Dijkstra Min-Heap ((V+E)logV) | 0.000795   | 0.000339  | 0.000078       |
| 497       | NetworkX (Referência)         | 0.002546   | 0.001119  | 0.000257       |

## 9. Gráficos

**Visualização: Comparação de Tempo Médio de Execução por Tamanho do Grafo**

<p align="center">
  <img src="./imagem/plotum.png" alt="Comparação de Tempo Médio de Execução por Tamanho do Grafo"><br>
</p>
<p align="center">

**OBS:** Toda vez que a imagem é inserida nesse repositório a imagem corrompe. Caso queira ver como ficou acesse o Link do Colab em Conteúdos Adicionais
  
  O gráfico mostra quanto tempo cada "solucionador" (algoritmo) leva para resolver o problema conforme ele fica maior.

 * Dijkstra Clássico: É método menos eficiente. No gráfico, a linha dele sobe muito rápido. Isso significa que, quando o grafo fica só um pouco maior, o tempo que ele leva para terminar "explode" (aumenta muito, muito rápido). É por isso que ele não conseguiu rodar para o grafo gigante de 100.000 nós.

* Dijkstra com Min-Heap: Esta é uma versão melhorada, mais "inteligente". A linha dele também sobe conforme o grafo aumenta, mas de forma muito mais suave e controlada. Mesmo quando o grafo fica bem grande, o tempo que ele leva aumenta, mas não "explode" como o clássico. Ele lida muito melhor com problemas grandes.

* NetworkX (Referência): A ferramenta já pronta que usamos (NetworkX) usa um método parecido com o Dijkstra com Min-Heap, por isso a linha dele se parece muito com a do Min-Heap, mostrando que também é eficiente para grafos grandes.

As Barras Verticais são as barras de Erro, elas mostram o quanto os tempos de execução variaram para aquele algoritmo e tamanho de grafo específico ao longo das 15 repetições.

  *   Barras Curtas: Significam que os tempos de execução para aquele algoritmo e tamanho foram bastante consistentes entre as repetições. A média calculada é mais "confiável".
  *   Barras Longas: Indicam que houve uma variação maior nos tempos de execução entre as repetições. A média calculada tem uma incerteza maior associada a ela.
  
Observe que, especialmente para o Dijkstra Clássico em tamanhos de grafo maiores, as barras de erro podem se tornar bastante longas, refletindo a grande variabilidade e o comportamento menos previsível do algoritmo quando ele começa a ficar sobrecarregado. Para os algoritmos mais eficientes, as barras tendem a ser mais curtas.
De modo geral, o gráfico mostra que, para grafos pequenos, todos são rápidos. Mas conforme o grafo cresce, o Dijkstra Clássico fica extremamente lento (a linha dispara para cima), enquanto o Dijkstra com Min-Heap e o NetworkX continuam relativamente rápidos (as linhas sobem bem menos).

A escala do gráfico ("logarítmica") ajuda a gente a ver essa diferença enorme de crescimento.
</p>

**Visualização: Gráfico de Emissões Médias de CO₂ vs. Tamanho do Grafo**

<p align="center">
  <img src="./imagem/plotdois.png" alt="Gráfico de Emissões Médias de CO₂ vs. Tamanho do Grafo"><br>
</p>

**OBS:** Toda vez que a imagem é inserida nesse repositório a imagem corrompe. Caso queira ver como ficou acesse o Link do Colab em Conteúdos Adicionais


<p align = "justify">
Assim como o tempo de execução, as emissões de CO₂ (a "pegada de carbono") estão diretamente ligadas ao quanto o computador trabalhou. Se um algoritmo leva mais tempo e usa mais recursos, ele tende a gerar mais emissões.

* **Dijkstra Clássico:** A linha dele para as emissões de CO₂ também sobe muito rapidamente conforme o grafo cresce. Assim como no gráfico de tempo, isso mostra que o algoritmo clássico, por ser menos eficiente para grafos grandes, faz o computador trabalhar muito mais e, consequentemente, gasta mais energia e gera mais CO₂.

* **Dijkstra com Min-Heap:** A linha para as emissões de CO₂ deste algoritmo sobe de forma muito mais suave e controlada, assim como no gráfico de tempo. Isso indica que, por ser mais eficiente e rápido, ele exige menos do computador e tem uma pegada de carbono menor para grafos grandes.

* **NetworkX (Referência):** A linha de CO₂ do NetworkX também segue de perto a do Min-Heap, reforçando que sua eficiência se traduz em menor consumo de energia e menores emissões de CO₂.

Em resumo, o segundo gráfico confirma o que vimos no primeiro: algoritmos mais eficientes (Min-Heap e NetworkX) não só rodam mais rápido, mas também são mais "verdes" ou consomem menos energia (e, portanto, geram menos CO₂ proporcionalmente ao tempo) do que o algoritmo clássico para grafos grandes. A diferença na pegada de carbono se torna enorme conforme o tamanho do problema aumenta. </p>

## Conclusão

<p align = "justify">
Algoritmos mais eficientes são mais rápidos e têm uma pegada de carbono consideravelmente menor para problemas maiores. O Dijkstra Clássico se torna extremamente lento e consome muito mais recursos (gerando mais CO₂) à medida que o tamanho do grafo aumenta, tornando-o inviável para grafos grandes. O Dijkstra com Min-Heap e a implementação do NetworkX demonstram ser muito mais escaláveis. Eles lidam com grafos grandes de forma muito mais eficiente, com um aumento de tempo e emissões de CO₂ significativamente menor em comparação com a versão clássica.

Isso mostra que a escolha do algoritmo correto, considerando sua eficiência (complexidade de tempo), tem um impacto direto e proporcional tanto na performance (rapidez) quanto no consumo de recursos computacionais (e, consequentemente, nas emissões de CO₂). Para problemas do mundo real com grandes conjuntos de dados (grafos), algoritmos otimizados como o Dijkstra com Min-Heap são essenciais não apenas pela velocidade, mas também por serem mais "sustentáveis" computacionalmente.</p>  


## Conteúdo Adicional

Link do Google Colab: https://colab.research.google.com/drive/14HANkKVl0tzKlMecL7OI-mWJqF6U53Rm?usp=sharing

Link do Vídeo: https://youtu.be/hLrdFNs2glM



