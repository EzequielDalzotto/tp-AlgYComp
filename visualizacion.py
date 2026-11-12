"""
visualizacion.py
================
Dibujo de la grilla con la cobertura resaltada y gráficas de resultados.
Usa matplotlib (y pandas para las gráficas de resultados).
"""

import matplotlib

import matplotlib.pyplot as plt

from grafos import aristas, es_diagonal

COLOR_CAMARA = "#d62728"
COLOR_NODO = "#cfd8dc"
COLOR_CALLE = "#607d8b"
COLOR_DIAGONAL = "#1f77b4"


def dibujar_grilla(adj, cobertura=None, titulo=None, ax=None, guardar=None,
                   etiquetas=False, costos=None):
    """
    Dibuja la grilla (pura o perturbada).

      - Nodos con cámara (cobertura): rojo.
      - Calles ortogonales: gris. Diagonales agregadas: azul, más gruesas.
      - Calles quitadas: simplemente no aparecen.
      - costos: dict nodo -> costo; si se pasa y etiquetas=True se muestra.

    Devuelve el eje de matplotlib.
    """
    cobertura = set(cobertura) if cobertura is not None else set()
    propio = ax is None
    if propio:
        filas = max(i for i, _ in adj) + 1
        cols = max(j for _, j in adj) + 1
        _, ax = plt.subplots(figsize=(max(3.5, cols * 0.55),
                                      max(3.5, filas * 0.55)))

    pos = {(i, j): (j, -i) for (i, j) in adj}
    for (u, v) in aristas(adj):
        (x1, y1), (x2, y2) = pos[u], pos[v]
        if es_diagonal(u, v):
            ax.plot([x1, x2], [y1, y2], color=COLOR_DIAGONAL, lw=1.6, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color=COLOR_CALLE, lw=1.0, zorder=1)

    xs = [pos[v][0] for v in adj]
    ys = [pos[v][1] for v in adj]
    colores = [COLOR_CAMARA if v in cobertura else COLOR_NODO for v in adj]
    tam = 120 if len(adj) <= 64 else 45
    ax.scatter(xs, ys, c=colores, s=tam, zorder=2, edgecolors="#37474f",
               linewidths=0.6)

    if etiquetas and costos is not None:
        for v in adj:
            ax.annotate(str(costos[v]), pos[v], ha="center", va="center",
                        fontsize=6, zorder=3)

    ax.set_aspect("equal")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=9)
    if guardar and propio:
        plt.savefig(guardar, dpi=150, bbox_inches="tight")
    return ax


def comparar_coberturas(adj, soluciones, guardar=None, costos=None):
    """
    Dibuja varias coberturas de la MISMA instancia lado a lado.
    soluciones: dict nombre -> conjunto de nodos.
    """
    k = len(soluciones)
    filas = max(i for i, _ in adj) + 1
    cols = max(j for _, j in adj) + 1
    fig, axes = plt.subplots(1, k, figsize=(k * max(3.2, cols * 0.5),
                                            max(3.2, filas * 0.5) + 0.4))
    if k == 1:
        axes = [axes]
    for ax, (nombre, cob) in zip(axes, soluciones.items()):
        extra = ""
        if costos is not None:
            extra = f" | costo {sum(costos[v] for v in cob)}"
        dibujar_grilla(adj, cob, f"{nombre}\n{len(cob)} cámaras{extra}", ax=ax,
                       costos=costos, etiquetas=costos is not None)
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Gráficas de resultados (a partir del DataFrame de experimentos.py)
# ---------------------------------------------------------------------------


def graficar_razon_vs_pd(df, columna_razon="razon_cota", tam=None,
                         guardar=None):
    """
    Razón de aproximación promedio (con desvío) de cada algoritmo en función
    de p_d. `columna_razon`: "razon_real" (contra el óptimo) o "razon_cota"
    (contra la cota inferior; es una cota superior de la razón real).
    """
    datos = df.copy()
    if tam is not None:
        datos = datos[(datos["n"] == tam[0]) & (datos["m"] == tam[1])]
    datos = datos.dropna(subset=[columna_razon])
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg, sub in datos.groupby("algoritmo"):
        g = sub.groupby("p_d")[columna_razon].agg(["mean", "std"]).fillna(0)
        ax.errorbar(g.index * 100, g["mean"], yerr=g["std"], marker="o",
                    capsize=3, label=alg)
    ax.axhline(2.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("p_d: diagonales agregadas (% de manzanas)")
    ax.set_ylabel(f"razón de aproximación ({columna_razon})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if tam:
        ax.set_title(f"Grilla {tam[0]}x{tam[1]}")
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=150)
    return fig


def graficar_tiempos(df, algoritmos=None, eje_x="nodos_grafo", logy=True,
                     guardar=None):
    """Tiempo promedio de cada algoritmo según el tamaño de la instancia."""
    datos = df.copy()
    if algoritmos:
        datos = datos[datos["algoritmo"].isin(algoritmos)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg, sub in datos.groupby("algoritmo"):
        g = sub.groupby(eje_x)["tiempo_s"].mean()
        ax.plot(g.index, g.values, marker="o", label=alg)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("cantidad de nodos")
    ax.set_ylabel("tiempo promedio (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=150)
    return fig


def graficar_branching_vs_pd(df, tam=None, guardar=None):
    """
    Umbral de perturbación: tiempo medio del branching y proporción de
    instancias que terminan (sin timeout) en función de p_d.
    """
    datos = df[df["algoritmo"] == "branching"].copy()
    if tam is not None:
        datos = datos[(datos["n"] == tam[0]) & (datos["m"] == tam[1])]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    g = datos.groupby("p_d").agg(tiempo=("tiempo_s", "mean"),
                                 completo=("completo", "mean"))
    ax1.plot(g.index * 100, g["tiempo"], "o-", color="#1f77b4")
    ax1.set_xlabel("p_d: diagonales agregadas (% de manzanas)")
    ax1.set_ylabel("tiempo medio del branching (s)", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(g.index * 100, g["completo"] * 100, "s--", color="#d62728")
    ax2.set_ylabel("instancias resueltas sin timeout (%)", color="#d62728")
    ax2.set_ylim(-5, 105)
    ax1.grid(alpha=0.3)
    if tam:
        ax1.set_title(f"Grilla {tam[0]}x{tam[1]}")
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=150)
    return fig
