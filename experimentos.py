"""
experimentos.py
===============
Barridos de parámetros con réplicas reproducibles.

Cada fila del DataFrame resultante es UNA corrida de UN algoritmo sobre UNA
instancia. Columnas principales:

  n, m, nodos_grafo, aristas   tamaño de la grilla
  p_d, p_q, replica, seed      parámetros y semilla de la instancia
  bipartito                    True si el grafo es bipartito
  algoritmo                    greedy | greedy_aleatorio | 2aprox_fijo |
                               2aprox_aleatorio | branching | konig
  tam                          tamaño de la cobertura obtenida
  valida                       la cobertura pasó el verificador es_cobertura
  tiempo_s                     tiempo (promedio de varias repeticiones)
  cota_inf                     matching máximo (cota inferior del óptimo)
  optimo                       óptimo exacto si se conoce (König o branching
                               que terminó), si no NaN
  razon_real                   tam / optimo
  razon_cota                   tam / cota_inf (cota SUPERIOR de la razón real)
  nodos_arbol, completo        solo para branching
"""

import time
import zlib

import pandas as pd

from algoritmos import (branching_exacto, cota_inferior, costo_cobertura,
                        greedy_aleatorio, greedy_grado_max, greedy_ponderado,
                        konig_cobertura, local_ratio_2aprox, matching_maximal_2aprox,
                        podar_redundantes)
from complejidad import es_cobertura
from grafos import (asignar_costos, cantidad_aristas, contar_diagonales,
                    es_bipartito, grilla_perturbada)

NAN = float("nan")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def semilla_instancia(base, n, m, p_d, p_q, replica):
    """Semilla determinista para una instancia. Registrarla en el informe."""
    clave = f"{base}|{n}x{m}|{p_d}|{p_q}|{replica}"
    return zlib.crc32(clave.encode())


def medir(f, repeticiones=1, tiempo_min=0.0):
    """
    Ejecuta f() varias veces y devuelve (resultado, tiempo_promedio_en_s).
    Repite hasta cumplir `repeticiones` y, además, hasta acumular `tiempo_min`
    segundos (útil para algoritmos muy rápidos).
    """
    total, veces, res = 0.0, 0, None
    while veces < repeticiones or total < tiempo_min:
        t0 = time.perf_counter()
        res = f()
        total += time.perf_counter() - t0
        veces += 1
    return res, total / veces


def resumir(df, por, columnas=("tam", "tiempo_s", "razon_real", "razon_cota")):
    """Promedio y desvío estándar de las métricas agrupando por `por`."""
    return df.groupby(por)[list(columnas)].agg(["mean", "std"]).round(4)


# ---------------------------------------------------------------------------
# Experimento principal (sin pesos)
# ---------------------------------------------------------------------------


def correr_instancia(adj, n, m, p_d, p_q, replica, seed,
                     timeout=10.0, repeticiones=3, con_branching=True):
    """Corre todos los algoritmos sobre una instancia y devuelve lista de filas."""
    biparto = es_bipartito(adj)
    lb = cota_inferior(adj)
    base = {
        "n": n, "m": m, "nodos_grafo": len(adj),
        "aristas": cantidad_aristas(adj),
        "diagonales": contar_diagonales(adj),
        "p_d": p_d, "p_q": p_q, "replica": replica, "seed": seed,
        "bipartito": biparto, "cota_inf": lb,
    }
    filas = []

    def registrar(nombre, cob, t, **extra):
        fila = dict(base)
        fila.update({
            "algoritmo": nombre,
            "tam": None if cob is None else len(cob),
            "valida": None if cob is None else es_cobertura(adj, cob),
            "tiempo_s": t,
            "nodos_arbol": NAN, "completo": NAN,
        })
        fila.update(extra)
        filas.append(fila)
        return fila

    # --- heurísticas ---
    cob, t = medir(lambda: greedy_grado_max(adj), repeticiones)
    registrar("greedy", cob, t)

    resultado, t = medir(lambda: greedy_aleatorio(adj, seed=seed), repeticiones)
    registrar("greedy_aleatorio", resultado.cobertura, t)

    cob, t = medir(lambda: matching_maximal_2aprox(adj)[0], repeticiones)
    registrar("2aprox_fijo", cob, t)

    cob, t = medir(lambda: matching_maximal_2aprox(adj, "aleatorio", seed)[0],
                   repeticiones)
    registrar("2aprox_aleatorio", cob, t)

    # --- óptimo exacto ---
    optimo = NAN
    if biparto:
        (cob, _), t = medir(lambda: konig_cobertura(adj), repeticiones)
        registrar("konig", cob, t)
        optimo = len(cob)

    if con_branching:
        r = branching_exacto(adj, timeout=timeout)
        registrar("branching", r["cobertura"], r["tiempo"],
                  nodos_arbol=r["nodos"], completo=r["completo"])
        if r["completo"]:
            optimo = r["tam"]

    # --- razones ---
    for fila in filas:
        fila["optimo"] = optimo
        fila["razon_real"] = (fila["tam"] / optimo
                              if optimo == optimo and optimo > 0 else NAN)
        fila["razon_cota"] = fila["tam"] / lb if lb > 0 else NAN
    return filas


def barrido(tamanos, p_ds, p_q, replicas, semilla_base=2026, timeout=10.0,
            repeticiones=3, con_branching=True, verbose=True):
    """
    Barrido completo.

    tamanos : lista de (n, m)
    p_ds    : lista de probabilidades de diagonal (incluir 0 como control)
    p_q     : probabilidad de quitar calles (fija) o lista con un valor por p_d
    replicas: cantidad de instancias por (tamaño, p_d)
    """
    filas = []
    for (n, m) in tamanos:
        for p_d in p_ds:
            # (con p_d = 0 y p_q = 0 todas las réplicas son la grilla pura;
            #  igual se corren para que los promedios tengan el mismo formato)
            for r in range(replicas):
                seed = semilla_instancia(semilla_base, n, m, p_d, p_q, r)
                adj = grilla_perturbada(n, m, p_d, p_q, seed)
                filas += correr_instancia(adj, n, m, p_d, p_q, r, seed,
                                          timeout, repeticiones,
                                          con_branching)
        if verbose:
            print(f"  listo {n}x{m}")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Experimento ponderado (desafío extra)
# ---------------------------------------------------------------------------


def barrido_ponderado(tamanos, p_ds, p_q, replicas, semilla_base=2026,
                      timeout=10.0, avenidas=False, bajo=1, alto=10,
                      verbose=True):
    """
    Compara, con costos por intersección:
      greedy_pond, local_ratio, local_ratio+poda, branching_pond
    y el costo que tendría la solución SIN pesos (greedy / 2-aprox / óptimo
    sin pesos) evaluada con los costos reales ("ignorar_costos").
    """
    filas = []
    for (n, m) in tamanos:
        for p_d in p_ds:
            for r in range(replicas):
                seed = semilla_instancia(semilla_base, n, m, p_d, p_q, r)
                adj = grilla_perturbada(n, m, p_d, p_q, seed)
                w = asignar_costos(adj, n, m, seed + 1, bajo, alto, avenidas)

                resultados = {}
                cob, t = medir(lambda: greedy_ponderado(adj, w), 3)
                resultados["greedy_pond"] = (cob, t)
                cob, t = medir(lambda: local_ratio_2aprox(adj, w), 3)
                resultados["local_ratio"] = (cob, t)
                resultados["local_ratio+poda"] = (
                    podar_redundantes(adj, cob, w), t)
                r_b = branching_exacto(adj, w=w, timeout=timeout)
                resultados["branching_pond"] = (r_b["cobertura"], r_b["tiempo"])

                # soluciones que ignoran los costos
                resultados["ignorar_costos(greedy)"] = (
                    greedy_grado_max(adj), 0.0)
                r_u = branching_exacto(adj, timeout=timeout)
                resultados["ignorar_costos(optimo)"] = (
                    r_u["cobertura"], r_u["tiempo"])

                optimo = r_b["costo"] if r_b["completo"] else NAN
                for nombre, (cob, t) in resultados.items():
                    costo = costo_cobertura(cob, w)
                    filas.append({
                        "n": n, "m": m, "nodos_grafo": len(adj),
                        "p_d": p_d, "p_q": p_q, "replica": r, "seed": seed,
                        "avenidas": avenidas,
                        "algoritmo": nombre, "tam": len(cob),
                        "costo": costo,
                        "valida": es_cobertura(adj, cob),
                        "tiempo_s": t,
                        "optimo_costo": optimo,
                        "completo": r_b["completo"],
                        "razon_real": (costo / optimo
                                       if optimo == optimo else NAN),
                    })
        if verbose:
            print(f"  listo {n}x{m} (ponderado)")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------


def guardar_csv(df, ruta):
    df.to_csv(ruta, index=False)
    return ruta
