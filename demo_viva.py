"""Trace the existing coverage algorithms and replay their choices on a grid."""

import sys
import textwrap
from dataclasses import dataclass

from algoritmos import greedy_grado_max, matching_maximal_2aprox
from grafos import aristas, es_diagonal, grilla_perturbada

METODOS = {"Greedy": greedy_grado_max, "Matching maximal": matching_maximal_2aprox}


@dataclass(frozen=True)
class Paso:
    camaras: frozenset
    cubiertas: frozenset
    recientes: frozenset
    nuevas_camaras: tuple = ()
    arista_elegida: tuple | None = None


def crear_pasos(adj, metodo):
    """Capture immutable display states from the production algorithm's decisions."""
    pendientes = set(aristas(adj))
    camaras, cubiertas = set(), set()
    pasos = [Paso(frozenset(), frozenset(), frozenset())]

    def registrar(nuevas, elegida):
        recientes = {e for e in pendientes if any(v in e for v in nuevas)}
        camaras.update(nuevas)
        pendientes.difference_update(recientes)
        cubiertas.update(recientes)
        pasos.append(Paso(frozenset(camaras), frozenset(cubiertas),
                          frozenset(recientes), nuevas, elegida))

    METODOS[metodo](adj, al_elegir=registrar)
    return pasos


class Simulacion:
    """Keep the current instance and its playback independent of Matplotlib."""

    def __init__(self):
        self.metodo = "Greedy"
        self.generar(4, 4, 0.0, 0.0, 42)

    def generar(self, filas, columnas, p_d, p_q, semilla):
        if not (2 <= filas <= 8 and 2 <= columnas <= 8):
            raise ValueError("Las filas y columnas deben estar entre 2 y 8.")
        if not (0 <= p_d <= 1 and 0 <= p_q <= 1):
            raise ValueError("Las probabilidades deben estar entre 0 y 1.")
        self.adj = grilla_perturbada(filas, columnas, p_d, p_q, seed=semilla)
        self.calles = aristas(self.adj)
        self.filas, self.columnas, self.semilla = filas, columnas, semilla
        self.cambiar_metodo(self.metodo)

    def cambiar_metodo(self, metodo):
        self.pasos = crear_pasos(self.adj, metodo)
        self.metodo = metodo
        self.reiniciar()

    def mover(self, desplazamiento):
        self.indice = max(0, min(self.indice + desplazamiento, len(self.pasos) - 1))

    def reiniciar(self):
        self.indice = 0


class DemoViva:
    """One desktop figure with controls to regenerate and replay a grid."""

    def __init__(self):
        from matplotlib import pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, TextBox

        self.modelo = Simulacion()
        self.fig = plt.figure(figsize=(11, 7), facecolor="#ffffff")
        self.fig.canvas.manager.set_window_title("Vertex Cover · Demo en vivo")
        self.ax = self.fig.add_axes([0.055, 0.28, 0.55, 0.57])
        self.fig.text(0.06, 0.935, "Cámaras en la ciudad", fontsize=23, weight="bold")
        self.fig.text(0.06, 0.89, "Vertex Cover paso a paso", fontsize=13, color="#40566b")
        self.info = self.fig.text(0.06, 0.245, "", fontsize=11, weight="bold")
        self.explicacion = self.fig.text(0.06, 0.205, "", fontsize=11, va="top")
        self.error = self.fig.text(0.68, 0.125, "", fontsize=10, color="#a12436", va="top")

        def campo(etiqueta, y, valor):
            self.fig.text(0.68, y + 0.045, etiqueta, fontsize=10)
            return TextBox(self.fig.add_axes([0.68, y, 0.26, 0.04]), "", initial=valor)

        self.fig.text(0.68, 0.89, "Configurar instancia", fontsize=13, weight="bold")
        self.filas = campo("Filas (2 a 8)", 0.80, "4")
        self.columnas = campo("Columnas (2 a 8)", 0.70, "4")
        self.diagonales = campo("Agregar diagonales (0 a 1)", 0.60, "0")
        self.quitar_calles = campo("Quitar calles (0 a 1)", 0.50, "0")
        self.semilla = campo("Semilla", 0.40, "42")
        self.generar = Button(self.fig.add_axes([0.68, 0.32, 0.26, 0.05]), "Generar instancia")
        self.generar.on_clicked(self._generar)

        self.fig.text(0.68, 0.265, "Algoritmo", fontsize=12, weight="bold")
        self.selector = RadioButtons(self.fig.add_axes([0.68, 0.15, 0.26, 0.10]), list(METODOS))
        self.selector.on_clicked(self._cambiar_metodo)

        self.anterior = Button(self.fig.add_axes([0.06, 0.06, 0.14, 0.06]), "Anterior")
        self.siguiente = Button(self.fig.add_axes([0.23, 0.06, 0.14, 0.06]), "Siguiente")
        self.reiniciar = Button(self.fig.add_axes([0.40, 0.06, 0.14, 0.06]), "Reiniciar")
        self.anterior.on_clicked(lambda _: self._mover(-1))
        self.siguiente.on_clicked(lambda _: self._mover(1))
        self.reiniciar.on_clicked(self._reiniciar)
        self._dibujar()

    def _generar(self, _):
        try:
            filas, columnas = int(self.filas.text), int(self.columnas.text)
            p_d = float(self.diagonales.text.replace(",", "."))
            p_q = float(self.quitar_calles.text.replace(",", "."))
            semilla = int(self.semilla.text)
        except ValueError:
            self.error.set_text("Ingresá enteros para tamaño y semilla,\ny números para las probabilidades.")
            self.fig.canvas.draw_idle()
            return
        try:
            self.modelo.generar(filas, columnas, p_d, p_q, semilla)
        except ValueError as exc:
            self.error.set_text(textwrap.fill(str(exc), 34))
            self.fig.canvas.draw_idle()
            return
        self.error.set_text("")
        self._dibujar()

    def _cambiar_metodo(self, metodo):
        self.modelo.cambiar_metodo(metodo)
        self._dibujar()

    def _mover(self, desplazamiento):
        self.modelo.mover(desplazamiento)
        self._dibujar()

    def _reiniciar(self, _):
        self.modelo.reiniciar()
        self._dibujar()

    def _dibujar(self):
        from matplotlib.lines import Line2D

        modelo = self.modelo
        paso = modelo.pasos[modelo.indice]
        total = len(modelo.pasos) - 1
        colores = {"pendiente": "#8191a1", "cubierta": "#16795f", "reciente": "#c66a1a",
                   "elegida": "#6744ae", "camara": "#165c91"}
        self.ax.clear()
        for calle in modelo.calles:
            ((i, j), (k, l)) = calle
            if calle == paso.arista_elegida:
                color, ancho, nivel = colores["elegida"], 4.0, 3
            elif calle in paso.recientes:
                color, ancho, nivel = colores["reciente"], 3.0, 2
            elif calle in paso.cubiertas:
                color, ancho, nivel = colores["cubierta"], 2.2, 1
            else:
                color, ancho, nivel = colores["pendiente"], 1.4, 0
            self.ax.plot([j, l], [-i, -k], color=color, lw=ancho, zorder=nivel,
                         linestyle="--" if es_diagonal(*calle) else "-")

        for (i, j) in modelo.adj:
            camara = (i, j) in paso.camaras
            nueva = (i, j) in paso.nuevas_camaras
            self.ax.scatter(j, -i, s=155 if nueva else 105, zorder=4,
                            facecolors=colores["camara"] if camara else "#ffffff",
                            edgecolors="#172d42", linewidths=2.2 if nueva else 0.8)
            self.ax.annotate(f"{i},{j}", (j, -i), xytext=(0, 8),
                             textcoords="offset points", ha="center", fontsize=8, zorder=5)

        self.ax.set_aspect("equal")
        self.ax.set_xlim(-0.45, modelo.columnas - 0.55)
        self.ax.set_ylim(-modelo.filas + 0.55, 0.45)
        self.ax.axis("off")
        self.ax.set_title(f"{modelo.metodo}  ·  Grilla {modelo.filas} × {modelo.columnas}"
                          f"  ·  Semilla {modelo.semilla}", fontsize=11)
        marcas = [Line2D([0], [0], color=colores[k], lw=2, label=etiqueta)
                  for k, etiqueta in (("pendiente", "Pendiente"), ("cubierta", "Cubierta"),
                                      ("reciente", "Recién cubierta"), ("elegida", "Elegida"))]
        marcas.append(Line2D([0], [0], color=colores["camara"], marker="o", ls="",
                             label="Cámara"))
        self.ax.legend(handles=marcas, loc="lower center", bbox_to_anchor=(0.5, -0.055),
                       ncol=3, fontsize=8, frameon=False)

        self.info.set_text(f"Paso {modelo.indice}/{total}   ·   {len(paso.camaras)} cámaras"
                           f"   ·   {len(paso.cubiertas)}/{len(modelo.calles)} calles cubiertas")
        if not modelo.indice:
            detalle = "Todavía no se eligieron cámaras. Avanzá para ver la primera decisión."
        elif paso.arista_elegida:
            u, v = paso.arista_elegida
            detalle = (f"Matching eligió la calle {u} – {v}: ninguno de sus extremos tenía "
                       f"cámara. Se cubrieron {len(paso.recientes)} calles nuevas.")
        else:
            detalle = (f"Greedy eligió {paso.nuevas_camaras[0]}: ningún otro cruce cubría "
                       f"más de {len(paso.recientes)} calles pendientes.")
        if modelo.indice == total:
            detalle += " Cobertura completa."
        self.explicacion.set_text(textwrap.fill(detalle, 70))
        self.anterior.set_active(modelo.indice > 0)
        self.siguiente.set_active(modelo.indice < total)
        self.reiniciar.set_active(modelo.indice > 0)
        self.fig.canvas.draw_idle()


def lanzar():
    """Open the local interactive window without forcing the batch backend."""
    from matplotlib import pyplot as plt

    demo = DemoViva()
    if demo.fig.canvas.required_interactive_framework is None:
        plt.close(demo.fig)
        print("La demo en vivo necesita un backend gráfico de Matplotlib (p. ej., TkAgg).",
              file=sys.stderr)
        return 1
    plt.show()
    return 0
