"""
complejidad.py
==============
Herramientas para el análisis de complejidad del proyecto.

  * Verificador polinomial de certificados (pertenencia a NP).
  * Reducción Vertex Cover <-> Independent Set y su verificación empírica.
  * Fuerza bruta (solo grafos chicos) como referencia independiente.
  * Independent Set exacto, con un algoritmo distinto al de Vertex Cover.
"""

from itertools import combinations

from algoritmos import branching_exacto
from grafos import aristas, copiar


# ---------------------------------------------------------------------------
# 1) Versión de decisión y certificado (pertenencia a NP)
# ---------------------------------------------------------------------------


def es_cobertura(adj, S, K=None):
    """
    VERIFICADOR del problema de decisión "¿existe una cobertura de tamaño <= K?".

    Certificado: un conjunto S de nodos.
    Se comprueba que (si se da K) |S| <= K y que cada arista (u, v) cumple
    u in S o v in S.  Costo: O(|N| + |A|) con S como conjunto hash.
    """
    S = set(S)
    if not S <= set(adj):
        return False
    if K is not None and len(S) > K:
        return False
    return all(u in S or v in S for (u, v) in aristas(adj))


def es_independiente(adj, S):
    """True si ninguna arista del grafo tiene sus dos extremos en S."""
    S = set(S)
    if not S <= set(adj):
        return False
    return all(not (u in S and v in S) for (u, v) in aristas(adj))


def complemento(adj, S):
    """N \\ S: los nodos del grafo que no están en S."""
    return set(adj) - set(S)


# ---------------------------------------------------------------------------
# 2) Reducción Vertex Cover <-> Independent Set
# ---------------------------------------------------------------------------


def reduccion_vc_a_is(adj, K):
    """
    Transformación polinomial:  (G, K)  ->  (G, |N| - K).

    (G, K) es instancia "sí" de Vertex Cover  <=>  (G, |N| - K) es instancia
    "sí" de Independent Set (conjunto independiente de tamaño >= |N| - K).
    Solo se calcula |N| - K: es trivialmente polinomial.
    """
    return adj, len(adj) - K


def reduccion_is_a_vc(adj, K_is):
    """Reducción inversa: (G, K_is) -> (G, |N| - K_is)."""
    return adj, len(adj) - K_is


def verificar_complemento(adj, S):
    """
    Lema: S es cobertura  <=>  N \\ S es independiente.
    Devuelve True si el lema se cumple para este S (sea o no cobertura).
    """
    return es_cobertura(adj, S) == es_independiente(adj, complemento(adj, S))


# ---------------------------------------------------------------------------
# 3) Referencias exactas independientes
# ---------------------------------------------------------------------------


def fuerza_bruta_minima(adj, w=None, max_nodos=22):
    """
    Cobertura de costo mínimo probando todos los subconjuntos por tamaño
    creciente. Solo para grafos CHICOS (por defecto hasta 22 nodos; con pesos
    conviene no pasar de ~18). Devuelve (cobertura, costo).

    Es un algoritmo independiente de `branching_exacto`: sirve para validarlo.
    """
    nodos = list(adj)
    n = len(nodos)
    if n > max_nodos:
        raise ValueError(f"fuerza bruta limitada a {max_nodos} nodos (n={n})")
    idx = {v: i for i, v in enumerate(nodos)}
    masks = [(1 << idx[u]) | (1 << idx[v]) for (u, v) in aristas(adj)]
    costos = [1 if w is None else w[v] for v in nodos]
    min_w = min(costos) if costos else 1

    mejor_costo = float("inf")
    mejor = None
    for k in range(n + 1):
        if mejor is not None and (w is None or k * min_w >= mejor_costo):
            break                       # sin pesos, el primer k que anda es el mínimo
        for comb in combinations(range(n), k):
            m = 0
            for i in comb:
                m |= 1 << i
            if all(m & em for em in masks):
                c = sum(costos[i] for i in comb)
                if c < mejor_costo:
                    mejor_costo = c
                    mejor = {nodos[i] for i in comb}
    return mejor, mejor_costo


def independiente_maximo_exacto(adj):
    """
    Conjunto independiente de tamaño máximo por ramificación (algoritmo
    distinto al de Vertex Cover, a propósito):

        nodo de grado <= 1 -> se incluye (siempre es seguro)
        si no, v de mayor grado:  excluir v   |   incluir v (y sacar sus vecinos)
    """
    def rec(vivos):
        if not vivos:
            return set()
        grados = {v: len(adj[v] & vivos) for v in vivos}
        for v, d in grados.items():
            if d <= 1:
                return {v} | rec(vivos - {v} - (adj[v] & vivos))
        v = max(grados, key=grados.get)
        sin_v = rec(vivos - {v})
        con_v = {v} | rec(vivos - {v} - adj[v])
        return con_v if len(con_v) >= len(sin_v) else sin_v

    return rec(frozenset(adj))


# ---------------------------------------------------------------------------
# 4) Verificación empírica de la reducción
# ---------------------------------------------------------------------------


def verificar_reduccion(adj, coberturas=None, usar_fuerza_bruta=True):
    """
    Comprueba empíricamente, sobre un grafo chico, que

        tau(G) + alpha(G) = |N|

    donde tau = tamaño de la cobertura mínima y alpha = tamaño del
    independiente máximo. Además, para cada cobertura de `coberturas`
    (dict nombre -> conjunto), verifica que su complemento es independiente.

    Devuelve un dict con los resultados.
    """
    n = len(adj)
    res = {"n": n}

    if usar_fuerza_bruta and n <= 22:
        res["tau_fuerza_bruta"] = len(fuerza_bruta_minima(adj)[0])
    res["tau_branching"] = branching_exacto(adj)["tam"]
    indep = independiente_maximo_exacto(adj)
    res["alpha"] = len(indep)
    res["alpha_es_independiente"] = es_independiente(adj, indep)
    res["tau_mas_alpha_igual_n"] = (res["tau_branching"] + res["alpha"] == n)
    if "tau_fuerza_bruta" in res:
        res["fuerza_bruta_coincide"] = (res["tau_fuerza_bruta"]
                                        == res["tau_branching"])

    if coberturas:
        res["complementos"] = {
            nombre: (es_cobertura(adj, S)
                     and es_independiente(adj, complemento(adj, S)))
            for nombre, S in coberturas.items()
        }
    return res
