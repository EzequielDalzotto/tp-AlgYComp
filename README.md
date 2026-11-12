# Ubicación de cámaras de vigilancia mediante Vertex Cover

Implementación del proyecto (Python 3.10+). Los algoritmos de cobertura están
escritos desde cero; `networkx` solo se usa como referencia para el matching
máximo en grafos generales (cota inferior), nunca para resolver Vertex Cover.

## Instalación

    pip install -r requirements.txt

## Archivos

| Archivo | Contenido |
|---|---|
| `grafos.py` | Grilla pura, grilla perturbada (`p_d`, `p_q`, semilla), costos, bipartición |
| `algoritmos.py` | Greedy por grado máximo, 2-aprox por matching maximal, König, branching con poda (con o sin pesos), greedy y local ratio ponderados |
| `complejidad.py` | Verificador `es_cobertura` (certificado en NP), reducción Vertex Cover <-> Independent Set, fuerza bruta y verificación empírica |
| `experimentos.py` | Barridos con réplicas y semillas, tiempos, razones de aproximación, CSV |
| `visualizacion.py` | Dibujo de la grilla con la cobertura y gráficas de resultados |
| `demo_viva.py` | Reproducción gráfica de greedy, matching maximal y greedy aleatorio |
| `tests.py` | Pruebas automáticas (`python tests.py`) |
| `main.py` | `python main.py demo`, `en-vivo`, `rapido` o `completo` |
| `notebook.ipynb` | Experimentos del informe, en el orden del proyecto |

## Uso rápido

    python tests.py            # verifica que todo funcione
    python main.py demo        # figuras de la grilla pura y la perturbada
    python main.py en-vivo     # ventana interactiva para la exposición
    python main.py rapido      # barrido chico (~1 min)

```python
from grafos import grilla_pura, grilla_perturbada
from algoritmos import greedy_grado_max, matching_maximal_2aprox, konig_cobertura, branching_exacto
from complejidad import es_cobertura

g = grilla_perturbada(8, 8, p_d=0.1, p_q=0.05, seed=42)
cob = greedy_grado_max(g)
print(len(cob), es_cobertura(g, cob))
print(branching_exacto(g, timeout=10))     # óptimo exacto con límite de tiempo
```

## Demo en vivo

En una sesión de escritorio, ejecutá `python main.py en-vivo`. **Siguiente** avanza una
decisión, **Anterior** vuelve una y **Reiniciar** conserva el mapa. Al cambiar de
algoritmo, se vuelve al paso cero sobre la misma instancia.

**Greedy aleatorio** elige al azar entre hasta tres cruces con más calles
pendientes. Hace diez intentos con la semilla de la instancia y retira cámaras
redundantes en cada uno. Se reproduce únicamente el primer intento que logró
la menor cantidad de cámaras. Si hubo retiradas en ese intento, se muestran
en el último paso.

Para comparar perturbaciones, cambiá las dimensiones (2–8), las probabilidades
(0–1) o la semilla y pulsá **Generar instancia**. Por ejemplo: probá la grilla
pura inicial y después `0,4` para diagonales y `0,25` para quitar calles.
Las diagonales se dibujan con líneas discontinuas y las calles quitadas no
aparecen. La demo muestra las decisiones registradas por los algoritmos del
proyecto; no mide rendimiento ni genera archivos en `resultados/`.

Si no se abre la ventana, comprobá que tu instalación de Python tenga un
backend gráfico de Matplotlib, como TkAgg, y que no estés forzando `Agg`.

## Decisiones de implementación

* Un grafo es `dict[nodo] -> set[nodo]` con nodos `(fila, columna)`. Los algoritmos no modifican el grafo recibido.
* **Perturbación:** primero se quitan calles (manteniendo la conexión) y después se agregan diagonales. Quitar calles no rompe la bipartición; una sola diagonal sí.
* **Greedy:** heap con entradas perezosas, O((V+E) log V). Desempate fijo o aleatorio con semilla.
* **2-aprox:** el resultado depende del orden de las aristas (fijo o aleatorio con semilla).
* **Branching:** ramifica en "v dentro" / "vecinos de v dentro", poda con la cota del matching y arranca con la 2-aprox como mejor solución. Incluye una regla de reducción para nodos hoja y `timeout`.
* **Ponderado:** el branching usa como cota la suma de `min(w[u], w[v])` sobre un matching. La 2-aprox ponderada es *local ratio*, porque el matching maximal puro ignora los costos.
* **Razones:** `razon_real = tam / óptimo` (óptimo conocido) y `razon_cota = tam / cota inferior` (cota superior de la razón real, para cuando el exacto no termina).

## Para el informe

Los barridos `rapido` y `completo` incluyen `greedy_aleatorio` como un método
distinto de `2aprox_aleatorio`. Su tiempo es el promedio de ejecuciones completas:
cada repetición usa la misma semilla de la instancia y mide los diez intentos,
incluida la poda. Los CSV y gráficos históricos solo se actualizan al volver a
ejecutar el barrido correspondiente.

Los parámetros de `main.py completo` y del notebook son orientativos. Elegí los tuyos (tamaños, `p_d`, `p_q`, réplicas, costos), registrá la semilla base y justificá cada decisión.
