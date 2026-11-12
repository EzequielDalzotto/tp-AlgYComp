"""
algoritmos.py
=============
Algoritmos para Vertex Cover (cobertura de nodos), escritos desde cero.

  Sin pesos
  ---------
  greedy_grado_max          heurística sin garantía constante (O(log n))
  matching_maximal_2aprox   2-aproximación (matching maximal)
  konig_cobertura           óptimo exacto en grafos BIPARTITOS (grilla pura)
  branching_exacto          óptimo exacto en general (exponencial, con poda)

  Con pesos (desafío extra)
  -------------------------
  greedy_ponderado          greedy por menor costo/grado
  local_ratio_2aprox        2-aproximación ponderada (pricing / local ratio)
  branching_exacto(w=...)   el mismo branching, con costos

Todas las funciones reciben el grafo como dict[nodo] -> set[nodo] y NO lo
modifican. `networkx` aparece solo en `cota_inferior` (matching máximo en
grafos generales) y no se usa ninguna de sus funciones de Vertex Cover.
"""

import heapq
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

from grafos import aristas, cantidad_aristas, colorear_bipartito, copiar

# ===========================================================================
# 1) Greedy por grado máximo
# ===========================================================================


def greedy_grado_max(adj, desempate="indice", seed=None, *, al_elegir=None):
    """
    Elige siempre el nodo de mayor grado del grafo residual, lo agrega a la
    cobertura y le quita sus aristas. Repite hasta que no queden aristas.

    desempate: "indice"    -> ante grados iguales, el menor nodo (determinista)
               "aleatorio" -> prioridad aleatoria fija por nodo (usa `seed`)

    Implementación con heap y entradas "perezosas" (lazy): cada vez que baja
    el grado de un nodo se inserta una entrada nueva; las viejas se descartan
    al sacarlas. Costo total O((V + E) log V).

    No tiene garantía de aproximación constante: en el peor caso la razón
    crece como O(log n).

    Optional observer: al_elegir(new_camera_nodes, None) after each selection.
    """
    g = copiar(adj)
    m_rest = cantidad_aristas(g)
    if desempate == "aleatorio":
        rng = random.Random(seed)
        prioridad = {v: rng.random() for v in g}
    else:
        prioridad = {v: v for v in g}

    heap = [(-len(g[v]), prioridad[v], v) for v in g if g[v]]
    heapq.heapify(heap)
    cobertura = set()

    while m_rest > 0:
        d, _, v = heapq.heappop(heap)
        if v not in g or len(g[v]) != -d:       # entrada vieja
            continue
        vecinos = g.pop(v)
        cobertura.add(v)
        m_rest -= len(vecinos)
        for u in vecinos:
            g[u].discard(v)
            if g[u]:
                heapq.heappush(heap, (-len(g[u]), prioridad[u], u))
        if al_elegir is not None:
            al_elegir((v,), None)
    return cobertura


@dataclass(frozen=True)
class ResultadoGreedyAleatorio:
    cobertura: frozenset
    orden: tuple
    intento_ganador: int
    intentos: int


def greedy_aleatorio(adj, *, seed=None, intentos=10, top_k=3):
    """Sample from the top residual degrees and keep the first best pruned cover.

    The returned selection order precedes pruning so the winner can be replayed.
    """
    if intentos < 1 or top_k < 1:
        raise ValueError("intentos and top_k must be positive")

    semillas = random.Random(seed)
    mejor = None
    for intento in range(1, intentos + 1):
        rng = random.Random(semillas.getrandbits(64))
        residual = copiar(adj)
        orden = []
        while True:
            candidatos = sorted((v for v in residual if residual[v]),
                                key=lambda v: (-len(residual[v]), v))[:top_k]
            if not candidatos:
                break
            v = rng.choice(candidatos)
            orden.append(v)
            for vecino in residual.pop(v):
                residual[vecino].remove(v)

        cobertura = podar_redundantes(adj, orden)
        if mejor is None or len(cobertura) < len(mejor.cobertura):
            mejor = ResultadoGreedyAleatorio(frozenset(cobertura), tuple(orden),
                                             intento, intentos)
    return mejor


# ===========================================================================
# 2) 2-aproximación por matching maximal
# ===========================================================================


def matching_maximal_2aprox(adj, orden="fijo", seed=None, *, al_elegir=None):
    """
    Recorre las aristas; si ninguno de sus extremos está usado, la agrega al
    matching y suma AMBOS extremos a la cobertura.

    Equivale al esquema "tomar una arista cualquiera, agregar ambos extremos y
    eliminar las aristas incidentes", y el resultado depende del orden.

    orden: "fijo"      -> aristas ordenadas lexicográficamente
           "aleatorio" -> aristas en orden aleatorio (usa `seed`)

    Garantía: las aristas elegidas forman un matching maximal M y |C| = 2|M|.
    Como toda cobertura necesita un extremo distinto por cada arista de M,
    OPT >= |M|, y por lo tanto |C| = 2|M| <= 2 * OPT.

    Devuelve (cobertura, matching).

    Optional observer: al_elegir(new_camera_nodes, selected_edge) for accepted edges.
    """
    lista = aristas(adj)
    if orden == "aleatorio":
        random.Random(seed).shuffle(lista)
    usados = set()
    matching = []
    for (u, v) in lista:
        if u not in usados and v not in usados:
            matching.append((u, v))
            usados.add(u)
            usados.add(v)
            if al_elegir is not None:
                al_elegir((u, v), (u, v))
    return set(usados), matching


# ===========================================================================
# 3) Óptimo exacto en grafos bipartitos: matching máximo + König
# ===========================================================================


def _matching_bipartito(adj, L):
    """
    Matching máximo por caminos aumentantes (algoritmo de Kuhn).
    L = un lado de la bipartición. Devuelve dict `pareja` (en ambos sentidos).
    """
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    pareja = {}

    def aumentar(u, visitado):
        for v in adj[u]:
            if v in visitado:
                continue
            visitado.add(v)
            if v not in pareja or aumentar(pareja[v], visitado):
                pareja[u] = v
                pareja[v] = u
                return True
        return False

    for u in L:
        aumentar(u, set())
    return pareja


def konig_cobertura(adj):
    """
    Cobertura MÍNIMA de un grafo bipartito, en tiempo polinomial.

    Construcción constructiva del teorema de König:
      1. Calcular un matching máximo M.
      2. Z = nodos alcanzables desde los nodos de L no emparejados por
         caminos alternantes (arista fuera de M de L a R, arista de M de R a L).
      3. Cobertura = (L \\ Z)  U  (R  n  Z).      Se cumple |cobertura| = |M|.

    Devuelve (cobertura, tamaño_del_matching_máximo).
    Lanza ValueError si el grafo no es bipartito.
    """
    color = colorear_bipartito(adj)
    if color is None:
        raise ValueError("el grafo no es bipartito: König no aplica")
    L = {v for v, c in color.items() if c == 0}
    R = {v for v, c in color.items() if c == 1}

    pareja = _matching_bipartito(adj, L)

    Z = {u for u in L if u not in pareja}
    cola = deque(Z)
    while cola:
        u = cola.popleft()                      # u está en L
        for v in adj[u]:
            if v not in Z and pareja.get(u) != v:   # arista que no es de M
                Z.add(v)
                w = pareja.get(v)               # v está en R (emparejado)
                if w is not None and w not in Z:
                    Z.add(w)
                    cola.append(w)

    cobertura = (L - Z) | (R & Z)
    tam_matching = sum(1 for u in L if u in pareja)
    return cobertura, tam_matching


# ===========================================================================
# 4) Cotas inferiores
# ===========================================================================


def cota_inferior(adj, metodo="maximo"):
    """
    Tamaño de un matching = cota inferior del óptimo (cada arista del matching
    exige un extremo distinto en la cobertura).

      metodo = "maximo"  -> matching máximo (la cota más ajustada).
                            Bipartito: Kuhn propio. General: blossom de
                            networkx (solo se usa como referencia).
      metodo = "maximal" -> matching maximal propio (más débil, sin librerías).
    """
    if metodo == "maximal":
        return len(matching_maximal_2aprox(adj)[1])

    color = colorear_bipartito(adj)
    if color is not None:
        L = {v for v, c in color.items() if c == 0}
        pareja = _matching_bipartito(adj, L)
        return sum(1 for u in L if u in pareja)
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(adj)
        G.add_edges_from(aristas(adj))
        return len(nx.max_weight_matching(G, maxcardinality=True))
    except ImportError:
        return len(matching_maximal_2aprox(adj)[1])


def _cota_matching_ponderada(g, w):
    """
    Cota inferior del costo óptimo: sobre un matching maximal, la suma de
    min(w[u], w[v]) de cada arista. (Cada arista del matching necesita al menos
    uno de sus extremos, y los extremos de distintas aristas son distintos.)
    Con w = 1 es el tamaño del matching.
    """
    usados = set()
    lb = 0
    for u in g:
        if u in usados:
            continue
        for v in g[u]:
            if v not in usados:
                usados.add(u)
                usados.add(v)
                lb += min(w[u], w[v])
                break
    return lb


# ===========================================================================
# 5) Óptimo exacto por branching con poda (con o sin pesos)
# ===========================================================================


class _TiempoAgotado(Exception):
    pass


def branching_exacto(adj, w=None, timeout=None, ub_inicial="auto",
                     ramificacion="grado", reglas_reduccion=True):
    """
    Óptimo exacto de Vertex Cover (ponderado si se pasa `w`) por ramificación
    y poda. Tiempo exponencial en el peor caso.

    Esquema:
        busca(G, C):
            si G no tiene aristas: actualizar la mejor solución
            si costo(C) + cota_inferior(G) >= mejor: podar
            ramificar:
                ramificacion="grado":  v = nodo de mayor grado
                    rama 1: v en la cobertura
                    rama 2: v fuera  =>  todos sus vecinos en la cobertura
                ramificacion="arista": arista (v, u) con v de mayor grado
                    rama 1: v en la cobertura
                    rama 2: u en la cobertura

    Parámetros
    ----------
    w                : dict nodo -> costo (None = todos 1).
    timeout          : segundos máximos (None = sin límite). Si se agota,
                       devuelve la mejor solución hallada y completo=False.
    ub_inicial       : "auto" -> arranca con la 2-aproximación como mejor
                       solución (poda desde el principio); None -> sin solución
                       inicial; o un conjunto de nodos que sea cobertura.
    reglas_reduccion : regla del nodo hoja. Si v tiene grado 1 y su vecino u
                       cumple w[u] <= w[v], existe un óptimo que contiene a u.

    Devuelve dict con: cobertura, tam, costo, nodos (del árbol de búsqueda),
    completo (False si hubo timeout), tiempo.
    """
    t0 = time.perf_counter()
    g = copiar(adj)
    unitario = w is None or all(x == 1 for x in w.values())
    if w is None:
        w = {v: 1 for v in g}

    # --- mejor solución inicial ---
    if ub_inicial == "auto":
        if unitario:
            inicial = matching_maximal_2aprox(adj)[0]
        else:
            inicial = local_ratio_2aprox(adj, w)
    elif ub_inicial is None:
        inicial = None
    else:
        inicial = set(ub_inicial)
    if inicial is not None:
        mejor = set(inicial)
        mejor_costo = sum(w[v] for v in mejor)
    else:
        mejor, mejor_costo = None, float("inf")

    cover = []              # cobertura parcial (pila)
    costo = 0
    m_rest = cantidad_aristas(g)
    nodos = 0
    abortado = False        # True cuando se agota el tiempo: ya no se deshace nada
    limite = None if timeout is None else t0 + timeout

    def tomar(v, pila):
        """Agrega v a la cobertura y lo quita del grafo residual."""
        nonlocal costo, m_rest
        vec = g.pop(v)
        for u in vec:
            g[u].discard(v)
        pila.append((v, vec))
        cover.append(v)
        costo += w[v]
        m_rest -= len(vec)

    def deshacer(pila):
        """Revierte, en orden inverso, todo lo hecho con tomar()."""
        nonlocal costo, m_rest
        if abortado:
            return
        while pila:
            v, vec = pila.pop()
            g[v] = vec
            for u in vec:
                g[u].add(v)
            cover.pop()
            costo -= w[v]
            m_rest += len(vec)

    def rec():
        nonlocal mejor, mejor_costo, nodos, abortado
        nodos += 1
        if limite is not None and (nodos & 127) == 0 \
                and time.perf_counter() > limite:
            abortado = True
            raise _TiempoAgotado()

        pila = []
        try:
            # --- reglas de reducción (nodos hoja) ---
            if reglas_reduccion:
                cambio = True
                while cambio:
                    cambio = False
                    for v in [x for x, vec in g.items() if len(vec) == 1]:
                        if v in g and len(g[v]) == 1:
                            (u,) = tuple(g[v])
                            if w[u] <= w[v]:
                                tomar(u, pila)
                                cambio = True

            # --- caso base ---
            if m_rest == 0:
                if costo < mejor_costo:
                    mejor_costo = costo
                    mejor = set(cover)
                return

            # --- poda por cota inferior ---
            if costo + _cota_matching_ponderada(g, w) >= mejor_costo:
                return

            # --- ramificación ---
            v = max(g, key=lambda x: len(g[x]) / w[x])
            if ramificacion == "arista":
                u = max(g[v], key=lambda x: len(g[x]))
                p1 = []
                tomar(v, p1)
                rec()
                deshacer(p1)
                p2 = []
                tomar(u, p2)
                rec()
                deshacer(p2)
            else:
                vecinos = list(g[v])
                p1 = []
                tomar(v, p1)                    # rama 1: v dentro
                rec()
                deshacer(p1)
                if costo + sum(w[u] for u in vecinos) < mejor_costo:
                    p2 = []                     # rama 2: v fuera
                    for u in vecinos:
                        tomar(u, p2)
                    rec()
                    deshacer(p2)
        finally:
            deshacer(pila)

    completo = True
    try:
        rec()
    except _TiempoAgotado:
        completo = False

    return {
        "cobertura": mejor,
        "tam": None if mejor is None else len(mejor),
        "costo": mejor_costo,
        "nodos": nodos,
        "completo": completo,
        "tiempo": time.perf_counter() - t0,
    }


# ===========================================================================
# 6) Versiones ponderadas (desafío extra)
# ===========================================================================


def costo_cobertura(cobertura, w):
    return sum(w[v] for v in cobertura)


def greedy_ponderado(adj, w, desempate="indice", seed=None):
    """
    Elige el nodo que minimiza costo/grado en el grafo residual (cuánto cuesta
    cada calle que "compra"). Misma estructura de heap perezoso que el greedy
    sin pesos. Sin garantía constante.
    """
    g = copiar(adj)
    m_rest = cantidad_aristas(g)
    if desempate == "aleatorio":
        rng = random.Random(seed)
        prioridad = {v: rng.random() for v in g}
    else:
        prioridad = {v: v for v in g}

    heap = [(w[v] / len(g[v]), prioridad[v], v, len(g[v])) for v in g if g[v]]
    heapq.heapify(heap)
    cobertura = set()
    while m_rest > 0:
        _, _, v, d = heapq.heappop(heap)
        if v not in g or len(g[v]) != d:
            continue
        vecinos = g.pop(v)
        cobertura.add(v)
        m_rest -= len(vecinos)
        for u in vecinos:
            g[u].discard(v)
            if g[u]:
                heapq.heappush(heap, (w[u] / len(g[u]), prioridad[u], u,
                                      len(g[u])))
    return cobertura


def local_ratio_2aprox(adj, w, orden="fijo", seed=None):
    """
    2-aproximación para Vertex Cover ponderado (pricing / local ratio):

        residual <- w
        para cada arista (u, v):
            si residual[u] > 0 y residual[v] > 0:
                d <- min(residual[u], residual[v])
                residual[u] -= d ;  residual[v] -= d
        cobertura <- nodos con residual = 0 (y alguna calle)

    Cada arista queda con un extremo "saturado", así que es una cobertura. El
    costo de cada nodo elegido es la suma de los d que se le restaron, y cada
    d se reparte entre sus dos extremos y es una cota inferior del óptimo
    => costo <= 2 * OPT.

    (El matching maximal "puro" no sirve acá: tomar ambos extremos ignora los
    costos y puede costar mucho más que el doble del óptimo.)
    """
    lista = aristas(adj)
    if orden == "aleatorio":
        random.Random(seed).shuffle(lista)
    residual = dict(w)
    for (u, v) in lista:
        if residual[u] > 0 and residual[v] > 0:
            d = min(residual[u], residual[v])
            residual[u] -= d
            residual[v] -= d
    return {v for v in adj if adj[v] and residual[v] == 0}


def podar_redundantes(adj, cobertura, w=None):
    """
    Quita de la cobertura (de mayor a menor costo) los nodos cuyos vecinos ya
    están todos en ella. Mantiene la validez y nunca empeora el costo.
    """
    cob = set(cobertura)
    for v in sorted(cob, key=lambda x: (-(w[x] if w else 1), x)):
        if all(u in cob for u in adj[v]):
            cob.discard(v)
    return cob
