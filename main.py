"""
main.py
=======
Punto de entrada. Uso:

    python main.py demo        # visualiza una instancia y compara algoritmos
    python main.py en-vivo     # abre el recorrido gráfico paso a paso
    python main.py rapido      # barrido chico (1-2 minutos), para probar
    python main.py completo    # barrido del informe (puede tardar bastante)

Los resultados (CSV y figuras) se guardan en la carpeta ./resultados
Los parámetros del modo "completo" son ORIENTATIVOS: elegí los tuyos y
justificalos en el informe.
"""

import os
import sys

if __name__ == "__main__" and sys.argv[1:2] == ["en-vivo"]:
    from demo_viva import lanzar

    raise SystemExit(lanzar())

import matplotlib

matplotlib.use("Agg")           # guardar figuras sin abrir ventanas

from algoritmos import (branching_exacto, greedy_grado_max, konig_cobertura,
                        matching_maximal_2aprox)
from complejidad import verificar_reduccion
from experimentos import barrido, barrido_ponderado, guardar_csv, resumir
from grafos import (contar_diagonales, es_bipartito, grilla_perturbada,
                    grilla_pura)
from visualizacion import (comparar_coberturas, graficar_branching_vs_pd,
                           graficar_razon_vs_pd, graficar_tiempos)

SALIDA = "resultados"


def demo():
    os.makedirs(SALIDA, exist_ok=True)

    # --- grilla pura: control ---
    g = grilla_pura(6, 6)
    cob_k, _ = konig_cobertura(g)
    comparar_coberturas(g, {
        "Grilla pura: König (óptimo)": cob_k,
        "Grilla pura: greedy": greedy_grado_max(g),
        "Grilla pura: 2-aprox": matching_maximal_2aprox(g)[0],
    }, guardar=f"{SALIDA}/demo_grilla_pura.png")

    # --- grilla perturbada ---
    p = grilla_perturbada(6, 6, p_d=0.25, p_q=0.05, seed=42)
    print(f"Perturbada 6x6: diagonales={contar_diagonales(p)} "
          f"bipartito={es_bipartito(p)}")
    opt = branching_exacto(p)
    comparar_coberturas(p, {
        "Perturbada: branching (óptimo)": opt["cobertura"],
        "Perturbada: greedy": greedy_grado_max(p),
        "Perturbada: 2-aprox": matching_maximal_2aprox(p)[0],
    }, guardar=f"{SALIDA}/demo_grilla_perturbada.png")

    # --- reducción a Independent Set ---
    chica = grilla_perturbada(4, 4, 0.3, 0.05, seed=3)
    print("Reducción VC <-> IS:", verificar_reduccion(
        chica, {"greedy": greedy_grado_max(chica),
                "2aprox": matching_maximal_2aprox(chica)[0]}))
    print(f"Figuras guardadas en ./{SALIDA}")


def experimento(tamanos_chicos, tamanos_medianos, p_ds, p_q, replicas,
                timeout, tag):
    os.makedirs(SALIDA, exist_ok=True)
    print("Instancias chicas ...")
    chicas = barrido(tamanos_chicos, p_ds, p_q, replicas, timeout=timeout)
    print("Instancias medianas ...")
    medianas = barrido(tamanos_medianos, p_ds, p_q, replicas, timeout=timeout)
    guardar_csv(chicas, f"{SALIDA}/{tag}_chicas.csv")
    guardar_csv(medianas, f"{SALIDA}/{tag}_medianas.csv")

    print("\n== CHICAS: razón real (contra el óptimo) ==")
    print(resumir(chicas, ["n", "p_d", "algoritmo"],
                  ("tam", "razon_real", "tiempo_s")).to_string())
    print("\n== MEDIANAS: razón contra la cota inferior ==")
    print(resumir(medianas, ["n", "p_d", "algoritmo"],
                  ("tam", "razon_cota", "tiempo_s")).to_string())

    graficar_razon_vs_pd(chicas, "razon_real", guardar=f"{SALIDA}/{tag}_razon_chicas.png")
    graficar_razon_vs_pd(medianas, "razon_cota", guardar=f"{SALIDA}/{tag}_razon_medianas.png")
    graficar_tiempos(medianas, guardar=f"{SALIDA}/{tag}_tiempos.png")
    mayor = max(tamanos_medianos)
    graficar_branching_vs_pd(medianas, tam=mayor, guardar=f"{SALIDA}/{tag}_umbral.png")

    print("\n== Desafío extra: ponderado ==")
    pond = barrido_ponderado([tamanos_chicos[-1]], p_ds, p_q, replicas,
                             timeout=timeout)
    guardar_csv(pond, f"{SALIDA}/{tag}_ponderado.csv")
    print(resumir(pond, ["p_d", "algoritmo"], ("costo", "razon_real")).to_string())
    print(f"\nResultados guardados en ./{SALIDA}")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if modo == "demo":
        demo()
    elif modo == "rapido":
        experimento([(4, 4), (6, 6)], [(8, 8), (10, 10)], [0.0, 0.1, 0.4],
                    0.05, replicas=3, timeout=3.0, tag="rapido")
    elif modo == "completo":
        experimento([(4, 4), (5, 5), (6, 6)],
                    [(8, 8), (10, 10), (12, 12), (15, 15)],
                    [0.0, 0.05, 0.1, 0.2, 0.4], 0.05, replicas=10,
                    timeout=30.0, tag="completo")
    else:
        print(__doc__)
