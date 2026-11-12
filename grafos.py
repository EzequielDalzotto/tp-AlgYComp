"""
grafos.py
=========
Generación de los grafos urbanos del proyecto "Ubicación de cámaras mediante
Vertex Cover".

Representación
--------------
Un grafo es un diccionario de adyacencias:  dict[nodo] -> set[nodo]
Los nodos son tuplas (i, j): fila i, columna j de la grilla.

  nodo  = intersección
  arista = calle (tramo entre dos intersecciones)

Los algoritmos de cobertura NO modifican el grafo recibido: trabajan sobre
una copia (ver `copiar`).
"""

import random
from collections import deque

# ---------------------------------------------------------------------------
# Utilidades básicas sobre grafos
# ---------------------------------------------------------------------------


def copiar(adj):
    """Copia profunda de un grafo (dict de sets)."""
    return {v: set(vecinos) for v, vecinos in adj.items()}


def aristas(adj):
    """Lista ordenada de aristas (u, v) con u < v, sin repetir."""
    return sorted((u, v) for u in adj for v in adj[u] if u < v)


def cantidad_aristas(adj):
    return sum(len(vecinos) for vecinos in adj.values()) // 2


def grado(adj, v):
    return len(adj[v])


def es_conexo(adj):
    """True si el grafo es conexo (BFS desde un nodo cualquiera)."""
    if not adj:
        return True
    inicio = next(iter(adj))
    visitados = {inicio}
    cola = deque([inicio])
    while cola:
        u = cola.popleft()
        for w in adj[u]:
            if w not in visitados:
                visitados.add(w)
                cola.append(w)
    return len(visitados) == len(adj)


def colorear_bipartito(adj):
    """
    Intenta 2-colorear el grafo (BFS por componente).
    Devuelve un dict nodo -> 0/1, o None si el grafo NO es bipartito
    (es decir, si tiene algún ciclo de longitud impar).
    """
    color = {}
    for inicio in adj:
        if inicio in color:
            continue
        color[inicio] = 0
        cola = deque([inicio])
        while cola:
            u = cola.popleft()
            for w in adj[u]:
                if w not in color:
                    color[w] = 1 - color[u]
                    cola.append(w)
                elif color[w] == color[u]:
                    return None
    return color


def es_bipartito(adj):
    return colorear_bipartito(adj) is not None


def es_diagonal(u, v):
    """True si la arista (u, v) es una diagonal de la grilla."""
    return abs(u[0] - v[0]) == 1 and abs(u[1] - v[1]) == 1


def contar_diagonales(adj):
    return sum(1 for (u, v) in aristas(adj) if es_diagonal(u, v))


# ---------------------------------------------------------------------------
# Grilla pura (caso de control)
# ---------------------------------------------------------------------------


def grilla_pura(n, m):
    """
    Grilla de n filas por m columnas. Aristas entre vecinos horizontales y
    verticales. Cantidad de aristas: n*(m-1) + m*(n-1).

    Es bipartita (tablero de ajedrez: color = (i + j) % 2), por lo que
    Vertex Cover se resuelve en tiempo polinomial (teorema de König).
    """
    adj = {(i, j): set() for i in range(n) for j in range(m)}
    for i in range(n):
        for j in range(m):
            if j + 1 < m:
                adj[(i, j)].add((i, j + 1))
                adj[(i, j + 1)].add((i, j))
            if i + 1 < n:
                adj[(i, j)].add((i + 1, j))
                adj[(i + 1, j)].add((i, j))
    return adj


# ---------------------------------------------------------------------------
# Grilla perturbada
# ---------------------------------------------------------------------------


def grilla_perturbada(n, m, p_d, p_q, seed=None):
    """
    Grilla perturbada:

      1. Se parte de la grilla pura n x m.
      2. Se QUITAN calles: cada arista ortogonal se elimina con probabilidad
         p_q, siempre que el grafo siga siendo conexo.
      3. Se AGREGAN diagonales: en cada manzana (i, j) se agrega, con
         probabilidad p_d, UNA diagonal (principal o secundaria, al azar).

    Por qué este orden: quitar aristas a un grafo bipartito lo deja
    bipartito, así que con p_d = 0 el grafo sigue siendo bipartito aunque
    p_q > 0. Una diagonal une dos nodos del MISMO color del tablero de
    ajedrez; como el grafo (después de quitar calles) es conexo, existe un
    camino de longitud par entre ellos y la diagonal cierra un ciclo IMPAR.
    Entonces basta UNA diagonal para que el grafo deje de ser bipartito.

    Parámetros
    ----------
    p_d, p_q : probabilidades en [0, 1].
    seed     : semilla para reproducibilidad (registrarla en el informe).
    """
    if not (0 <= p_d <= 1 and 0 <= p_q <= 1):
        raise ValueError("p_d y p_q deben estar en [0, 1]")

    rng = random.Random(seed)
    adj = grilla_pura(n, m)

    # 2) quitar calles manteniendo la conexión
    if p_q > 0:
        candidatas = aristas(adj)
        rng.shuffle(candidatas)
        for (u, v) in candidatas:
            if rng.random() < p_q:
                adj[u].discard(v)
                adj[v].discard(u)
                if not es_conexo(adj):          # era un puente: se restituye
                    adj[u].add(v)
                    adj[v].add(u)

    # 3) agregar diagonales (una por manzana, con probabilidad p_d)
    for i in range(n - 1):
        for j in range(m - 1):
            if rng.random() < p_d:
                if rng.random() < 0.5:
                    u, v = (i, j), (i + 1, j + 1)       # diagonal principal
                else:
                    u, v = (i, j + 1), (i + 1, j)       # diagonal secundaria
                adj[u].add(v)
                adj[v].add(u)
    return adj


# ---------------------------------------------------------------------------
# Costos de instalación (desafío extra: Vertex Cover ponderado)
# ---------------------------------------------------------------------------


def asignar_costos(adj, n, m, seed=None, bajo=1, alto=10,
                   avenidas=False, factor=3):
    """
    Asigna a cada intersección un costo entero aleatorio en [bajo, alto].

    Si avenidas=True, las intersecciones de la fila central y de la columna
    central (las "avenidas") cuestan `factor` veces más.
    Devuelve dict nodo -> costo.
    """
    rng = random.Random(seed)
    w = {v: rng.randint(bajo, alto) for v in adj}
    if avenidas:
        fila_c, col_c = n // 2, m // 2
        for (i, j) in w:
            if i == fila_c or j == col_c:
                w[(i, j)] *= factor
    return w
