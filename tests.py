"""
tests.py
========
Pruebas rápidas. Ejecutar con:   python tests.py     (o con pytest)
"""

import random

from algoritmos import (branching_exacto, cota_inferior, costo_cobertura,
                        greedy_grado_max, greedy_ponderado, konig_cobertura,
                        local_ratio_2aprox, matching_maximal_2aprox,
                        podar_redundantes)
from complejidad import (es_cobertura, es_independiente, complemento,
                         fuerza_bruta_minima, independiente_maximo_exacto,
                         reduccion_vc_a_is, verificar_complemento,
                         verificar_reduccion)
from grafos import (asignar_costos, cantidad_aristas, contar_diagonales,
                    aristas, es_bipartito, es_conexo, grilla_perturbada,
                    grilla_pura)


def instancias_chicas(cantidad=12):
    """Grillas perturbadas chicas con parámetros variados (reproducibles)."""
    rng = random.Random(7)
    for k in range(cantidad):
        n, m = rng.choice([(3, 3), (3, 4), (4, 4)])
        yield grilla_perturbada(n, m, rng.choice([0.2, 0.4, 0.7]),
                                rng.choice([0.0, 0.1, 0.2]), seed=k)


# ---------------------------------------------------------------------------
# grafos.py
# ---------------------------------------------------------------------------


def test_cantidad_de_aristas_grilla_pura():
    for n, m in [(2, 2), (3, 5), (6, 6), (8, 10)]:
        assert cantidad_aristas(grilla_pura(n, m)) == n * (m - 1) + m * (n - 1)


def test_grilla_pura_es_bipartita():
    assert es_bipartito(grilla_pura(6, 7))


def test_quitar_calles_mantiene_bipartito_y_conexo():
    for seed in range(10):
        g = grilla_perturbada(6, 6, 0.0, 0.2, seed)
        assert es_bipartito(g) and es_conexo(g)
        assert contar_diagonales(g) == 0


def test_una_diagonal_rompe_la_biparticion():
    for seed in range(20):
        g = grilla_perturbada(6, 6, 0.3, 0.1, seed)
        assert es_conexo(g)
        if contar_diagonales(g) > 0:
            assert not es_bipartito(g)


def test_reproducibilidad_con_semilla():
    a = grilla_perturbada(8, 8, 0.2, 0.05, seed=123)
    b = grilla_perturbada(8, 8, 0.2, 0.05, seed=123)
    assert a == b


# ---------------------------------------------------------------------------
# Óptimo en la grilla pura (König) y cota inferior
# ---------------------------------------------------------------------------


def test_optimo_grilla_pura_es_piso_nm_sobre_2():
    for n in range(1, 9):
        for m in range(2, 9):
            g = grilla_pura(n, m)
            cob, tam_m = konig_cobertura(g)
            assert es_cobertura(g, cob)
            assert len(cob) == tam_m == (n * m) // 2, (n, m)


def test_konig_coincide_con_branching_y_fuerza_bruta():
    for n, m in [(2, 3), (3, 3), (4, 4)]:
        g = grilla_pura(n, m)
        k = len(konig_cobertura(g)[0])
        assert k == branching_exacto(g)["tam"] == len(fuerza_bruta_minima(g)[0])


def test_konig_en_grilla_con_calles_quitadas():
    for seed in range(8):
        g = grilla_perturbada(6, 6, 0.0, 0.2, seed)
        cob, tam_m = konig_cobertura(g)
        assert es_cobertura(g, cob) and len(cob) == tam_m
        assert len(cob) == branching_exacto(g)["tam"]


def test_konig_rechaza_no_bipartito():
    g = grilla_perturbada(4, 4, 1.0, 0.0, seed=1)
    try:
        konig_cobertura(g)
        assert False, "debía lanzar ValueError"
    except ValueError:
        pass


def test_cota_inferior_networkx_coincide_con_propia_en_bipartito():
    g = grilla_pura(6, 6)
    assert cota_inferior(g) == 18


# ---------------------------------------------------------------------------
# Heurísticas
# ---------------------------------------------------------------------------


def test_heuristicas_devuelven_coberturas_validas():
    for g in list(instancias_chicas()) + [grilla_perturbada(10, 10, 0.2, 0.05, 3)]:
        assert es_cobertura(g, greedy_grado_max(g))
        assert es_cobertura(g, greedy_grado_max(g, "aleatorio", 5))
        assert es_cobertura(g, matching_maximal_2aprox(g)[0])
        assert es_cobertura(g, matching_maximal_2aprox(g, "aleatorio", 5)[0])


def test_2aprox_cumple_la_garantia_y_el_matching_es_valido():
    for g in instancias_chicas():
        opt = len(fuerza_bruta_minima(g)[0])
        cob, M = matching_maximal_2aprox(g)
        assert len(cob) == 2 * len(M) <= 2 * opt
        extremos = [x for arista in M for x in arista]
        assert len(extremos) == len(set(extremos))      # extremos distintos
        assert len(M) <= opt                            # cota inferior


def test_greedy_nunca_baja_del_optimo():
    for g in instancias_chicas():
        assert len(greedy_grado_max(g)) >= len(fuerza_bruta_minima(g)[0])


# ---------------------------------------------------------------------------
# Branching exacto
# ---------------------------------------------------------------------------


def test_branching_coincide_con_fuerza_bruta():
    for g in instancias_chicas(20):
        r = branching_exacto(g)
        assert r["completo"] and es_cobertura(g, r["cobertura"])
        assert r["tam"] == len(fuerza_bruta_minima(g)[0])


def test_branching_variantes_dan_el_mismo_optimo():
    for g in instancias_chicas():
        base = branching_exacto(g)["tam"]
        assert branching_exacto(g, ramificacion="arista")["tam"] == base
        assert branching_exacto(g, reglas_reduccion=False)["tam"] == base
        assert branching_exacto(g, ub_inicial=None)["tam"] == base


def test_branching_timeout_devuelve_solucion_valida():
    g = grilla_perturbada(15, 15, 0.4, 0.05, seed=9)
    r = branching_exacto(g, timeout=0.05)
    assert es_cobertura(g, r["cobertura"])
    assert r["tam"] <= len(matching_maximal_2aprox(g)[0])


# ---------------------------------------------------------------------------
# Complejidad: certificado y reducción
# ---------------------------------------------------------------------------


def test_verificador_acepta_y_rechaza():
    g = grilla_pura(3, 3)
    assert es_cobertura(g, set(g))
    assert not es_cobertura(g, set())
    assert not es_cobertura(g, set(g), K=3)
    cob = konig_cobertura(g)[0]
    assert es_cobertura(g, cob, K=len(cob))
    assert not es_cobertura(g, cob, K=len(cob) - 1)


def test_lema_complemento_cobertura_independiente():
    rng = random.Random(1)
    for g in instancias_chicas():
        nodos = list(g)
        for _ in range(30):
            S = {v for v in nodos if rng.random() < 0.5}
            assert verificar_complemento(g, S)


def test_tau_mas_alpha_es_n():
    for g in list(instancias_chicas(8)) + [grilla_pura(4, 4),
                                           grilla_perturbada(5, 5, 0.3, 0.05, 2)]:
        cobs = {"greedy": greedy_grado_max(g),
                "2aprox": matching_maximal_2aprox(g)[0]}
        r = verificar_reduccion(g, cobs)
        assert r["tau_mas_alpha_igual_n"]
        assert r["alpha_es_independiente"]
        assert all(r["complementos"].values())
        if "fuerza_bruta_coincide" in r:
            assert r["fuerza_bruta_coincide"]


def test_transformacion_decision():
    g = grilla_pura(4, 4)
    tau = len(konig_cobertura(g)[0])
    alpha = len(independiente_maximo_exacto(g))
    _, k_is = reduccion_vc_a_is(g, tau)
    assert k_is == alpha                      # cobertura <= tau  <=>  indep >= n - tau


# ---------------------------------------------------------------------------
# Versión ponderada
# ---------------------------------------------------------------------------


def test_ponderado_coincide_con_fuerza_bruta():
    for k, g in enumerate(instancias_chicas(8)):
        if len(g) > 16:
            continue
        w = asignar_costos(g, 4, 4, seed=k)
        r = branching_exacto(g, w=w)
        _, opt = fuerza_bruta_minima(g, w, max_nodos=16)
        assert r["completo"] and r["costo"] == opt
        assert costo_cobertura(r["cobertura"], w) == opt


def test_ponderado_2aprox_y_greedy_validos():
    for k, g in enumerate(instancias_chicas(8)):
        if len(g) > 16:
            continue
        w = asignar_costos(g, 4, 4, seed=k)
        opt = branching_exacto(g, w=w)["costo"]
        lr = local_ratio_2aprox(g, w)
        assert es_cobertura(g, lr)
        assert costo_cobertura(lr, w) <= 2 * opt
        assert es_cobertura(g, greedy_ponderado(g, w))
        podada = podar_redundantes(g, lr, w)
        assert es_cobertura(g, podada)
        assert costo_cobertura(podada, w) <= costo_cobertura(lr, w)


def test_sin_pesos_equivale_a_pesos_uno():
    for g in instancias_chicas():
        w1 = {v: 1 for v in g}
        assert branching_exacto(g, w=w1)["tam"] == branching_exacto(g)["tam"]


def test_demo_greedy_registra_la_decision_real():
    adj = {0: {1}, 1: {0, 2, 3}, 2: {1}, 3: {1}}
    pasos = []
    cobertura = greedy_grado_max(adj, al_elegir=lambda *paso: pasos.append(paso))
    assert cobertura == {1}
    assert pasos == [((1,), None)]


def test_demo_matching_registra_las_aristas_elegidas():
    adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    pasos = []
    cobertura, matching = matching_maximal_2aprox(
        adj, al_elegir=lambda *paso: pasos.append(paso))
    assert cobertura == {0, 1, 2, 3}
    assert pasos == [((0, 1), (0, 1)), ((2, 3), (2, 3))]
    assert [arista for _, arista in pasos] == matching


def test_demo_reproduce_las_coberturas_de_los_algoritmos():
    from demo_viva import crear_pasos

    adj = grilla_perturbada(4, 4, 0.4, 0.25, seed=42)
    calles = set(aristas(adj))
    for metodo, esperado in (("Greedy", greedy_grado_max(adj)),
                             ("Matching maximal", matching_maximal_2aprox(adj)[0])):
        pasos = crear_pasos(adj, metodo)
        assert not pasos[0].camaras and not pasos[0].cubiertas
        for anterior, actual in zip(pasos, pasos[1:]):
            assert actual.camaras > anterior.camaras
            assert actual.cubiertas == {e for e in calles if actual.camaras.intersection(e)}
            assert actual.recientes == actual.cubiertas - anterior.cubiertas
        assert pasos[-1].camaras == esperado
        assert pasos[-1].cubiertas == calles


def test_demo_navegacion_conserva_la_instancia_al_cambiar_metodo():
    from demo_viva import Simulacion

    modelo = Simulacion()
    modelo.generar(3, 4, 0.4, 0.2, 19)
    instancia = modelo.adj
    modelo.mover(-1)
    assert modelo.indice == 0
    modelo.mover(1)
    assert modelo.indice == 1
    modelo.cambiar_metodo("Matching maximal")
    assert modelo.adj is instancia and modelo.indice == 0
    for _ in modelo.pasos:
        modelo.mover(1)
    assert modelo.indice == len(modelo.pasos) - 1
    modelo.reiniciar()
    assert modelo.indice == 0 and modelo.adj is instancia
    modelo.generar(3, 4, 0.4, 0.2, 19)
    assert modelo.adj == instancia and modelo.indice == 0


def test_greedy_aleatorio_gana_despues_de_podar_y_conserva_el_primer_empate():
    from algoritmos import greedy_aleatorio

    estrella = {0: {1, 2, 3}, 1: {0}, 2: {0}, 3: {0}}
    primer_intento = greedy_aleatorio(estrella, seed=2, intentos=1)
    ganador = greedy_aleatorio(estrella, seed=2)
    assert primer_intento.cobertura == {1, 2, 3}
    assert ganador.cobertura == {0} and ganador.intento_ganador == 2
    assert ganador.orden == (2, 0)  # A lower-degree leaf was a valid candidate.
    assert ganador.intentos == 10
    assert ganador == greedy_aleatorio(dict(reversed(list(estrella.items()))), seed=2)


def test_greedy_aleatorio_devuelve_coberturas_validas_sin_camaras_redundantes():
    from algoritmos import greedy_aleatorio

    for seed, adj in enumerate(instancias_chicas()):
        resultado = greedy_aleatorio(adj, seed=seed)
        assert es_cobertura(adj, resultado.cobertura)
        assert all(not es_cobertura(adj, resultado.cobertura - {v})
                   for v in resultado.cobertura)


def test_demo_greedy_aleatorio_reproduce_la_poda_del_intento_ganador():
    from demo_viva import crear_pasos

    estrella = {0: {1, 2, 3}, 1: {0}, 2: {0}, 3: {0}}
    pasos = crear_pasos(estrella, "Greedy aleatorio", seed=2)
    assert [paso.camaras for paso in pasos] == [set(), {2}, {0, 2}, {0}]
    assert [paso.nuevas_camaras for paso in pasos[1:3]] == [(2,), (0,)]
    assert pasos[-1].retiradas == {2}
    assert pasos[-1].cubiertas == pasos[-2].cubiertas == set(aristas(estrella))
    assert not pasos[-1].recientes
    assert all(paso.intento_ganador == 2 and paso.intentos == 10 for paso in pasos)


def test_demo_greedy_aleatorio_usa_la_semilla_de_la_instancia():
    from algoritmos import greedy_aleatorio
    from demo_viva import Simulacion

    modelo = Simulacion()
    modelo.generar(3, 4, 0.4, 0.2, 19)
    instancia = modelo.adj
    modelo.cambiar_metodo("Greedy aleatorio")
    assert modelo.adj is instancia and modelo.indice == 0
    esperado = greedy_aleatorio(instancia, seed=19)
    assert modelo.pasos[-1].camaras == esperado.cobertura
    assert modelo.pasos[-1].intento_ganador == esperado.intento_ganador


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pruebas = [(nombre, f) for nombre, f in sorted(globals().items())
               if nombre.startswith("test_") and callable(f)]
    fallos = 0
    for nombre, f in pruebas:
        try:
            f()
            print(f"  ok   {nombre}")
        except Exception as e:                  # noqa: BLE001
            fallos += 1
            print(f"  FALLA {nombre}: {type(e).__name__}: {e}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas pasaron")
    raise SystemExit(1 if fallos else 0)
