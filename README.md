# Modelo de predicción del Balón de Oro

### Para ver los resultados del modelo visita: https://app-apppy-drjtfhstmpfw9umpncevzw.streamlit.app/

## 1. Objetivo y resumen

Este proyecto construye un ranking de candidatos al Balón de Oro a partir de sus estadísticas por competición y temporada. El modelo aprende de los rankings históricos qué jugador debería quedar por encima de otro y transforma las estadísticas de cada candidato en una puntuación (`score_modelo`). Cuanto mayor es la puntuación, más arriba queda el jugador en el ranking predicho.

La implementación utiliza una **regresión logística por pares** (*pairwise logistic regression*), con coeficientes diferentes para cada posición y restricciones monótonas sobre los signos de los coeficientes. La optimización numérica se realiza con `scipy.optimize.minimize` y `L-BFGS-B`; Optuna selecciona los hiperparámetros mediante validación temporal.

El modelo no predice directamente un puesto real ni una probabilidad calibrada de ganar el Balón de Oro. Aprende a ordenar jugadores de forma coherente con los enfrentamientos entre posiciones observados en los rankings históricos.

### Componentes principales

1. Carga y limpieza básica de `data/dataset_ml_base.csv`.
2. Agrupación de estadísticas por jugador, temporada y categoría de competición.
3. Normalización mediante *z-scores* calculados por temporada.
4. Construcción de una matriz de características con un bloque independiente por posición.
5. Generación de pares de jugadores y ponderación estática inspirada en Delta-NDCG.
6. Ajuste de una regresión logística pareada regularizada y con restricciones monótonas.
7. Selección de hiperparámetros con dos búsquedas de Optuna.
8. Backtest independiente de las temporadas 2020/21–2024/25.
9. Entrenamiento final con temporadas anteriores a 2025/26 y generación de la predicción 2025/26.
10. Exportación del ranking, métricas, coeficientes, importancia descriptiva y comparaciones de jugadores.

---

### Rutas configuradas

```text
data/
├── dataset_ml_base.csv
└── ml/
    ├── resumen_backtest_logistic_solo_temporada.csv
    ├── optuna_trials_solo_temporada_evaluacion_hasta_2019_20.csv
    ├── optuna_best_params_solo_temporada_evaluacion_hasta_2019_20.csv
    ├── optuna_trials_solo_temporada_prediccion_2025_26.csv
    ├── optuna_best_params_solo_temporada_prediccion_2025_26.csv
    ├── prediccion_solo_temporada_backtest_2020_21.csv
    ├── prediccion_solo_temporada_backtest_2021_22.csv
    ├── prediccion_solo_temporada_backtest_2022_23.csv
    ├── prediccion_solo_temporada_backtest_2023_24.csv
    ├── prediccion_solo_temporada_backtest_2024_25.csv
    ├── prediccion_solo_temporada_2025_26.csv
    ├── coeficientes_logistic_solo_temporada.csv
    ├── importancia_estadisticas_solo_temporada.csv
    ├── comparacion_mbappe_lamine_solo_temporada.csv
    └── comparacion_cubarsi_lamine_solo_temporada.csv
```

Los rankings de backtest se generan únicamente para las temporadas de `BACKTEST_TEMPORADAS` que estén presentes en el dataset. La predicción final está definida por `TEMPORADA_PREDICCION = '2025/26'`.

---

## 3. Datos de entrada y unidad de análisis

El archivo de entrada es `data/dataset_ml_base.csv`. Se espera que contenga, como mínimo, campos compatibles con los utilizados en el código, entre ellos:

- `temporada`
- `nombre`
- `club`
- `posicion`
- `competicion`
- `posicion_balon_oro`
- `resultado_competicion`
- estadísticas numéricas de rendimiento

El script elimina las filas que no tienen `competicion` o `posicion`. La posición se utiliza tal como aparece en el CSV: el código **no convierte ni agrupa automáticamente** etiquetas como `Attacker`, `Forward` o `Striker`. Por ello, las categorías de posición deben venir normalizadas en el dataset; etiquetas distintas se tratarán como posiciones distintas.

La clave de agregación del jugador es:

```python
CLAVE_JUGADOR = ['temporada', 'nombre']
```

---

## 4. Qué estadísticas utiliza realmente el modelo

El modelo no crea una única característica genérica como `goals`. Genera una característica por **categoría de competición y estadística**, agrega sus valores por jugador y temporada y después normaliza cada característica. Por ejemplo:

```text
5_grandes_ligas_goals__zscore
uefa_champions_league_goals__zscore
fifa_world_cup_goals__zscore
5_grandes_ligas_accuratePassesPercentage__zscore
uefa_champions_league_resultado_competicion__zscore
```

La lista exacta de características depende de las columnas presentes, de las competiciones del CSV y de las categorías resultantes de la normalización de nombres. Las competiciones que no figuran en `CATEGORIAS_COMPETICION` se convierten en una categoría propia a partir de su nombre limpiado.

### 4.1 Estadísticas aditivas

Cuando la columna está presente en el dataset, se agregan sumando los valores del jugador en cada categoría de competición:

| Estadística | Interpretación general |
|---|---|
| `goalsAssistsSum` | Suma de goles y asistencias, normalmente proporcionada por el dataset o calculada si la columna no existe. |
| `bigChancesCreated` | Grandes ocasiones creadas. |
| `bigChancesMissed` | Grandes ocasiones falladas. |
| `keyPasses` | Pases clave. |
| `successfulDribbles` | Regates completados. |
| `accuratePasses` | Pases precisos. |
| `totalAttemptAssist` | Intentos que generan una asistencia de tiro, según la definición de origen. |
| `aerialDuelsWon` | Duelos aéreos ganados. |
| `totalDuelsWon` | Duelos totales ganados. |
| `tacklesWon` | Entradas ganadas/completadas. |
| `interceptions` | Intercepciones. |
| `outfielderBlocks` | Bloqueos realizados por jugadores de campo. |
| `ballRecovery` | Recuperaciones de balón. |
| `dispossessed` | Veces que el jugador pierde la posesión al ser desposeído. |
| `dribbledPast` | Veces que un rival supera al jugador con un regate. |
| `clearances` | Despejes. |
| `errorLeadToGoal` | Errores que terminan en gol rival. |
| `errorLeadToShot` | Errores que terminan en un tiro rival. |
| `saves` | Paradas de portero. |
| `goalsPrevented` | Goles evitados, de acuerdo con la fuente. |
| `goalsConceded` | Goles encajados. |
| `highClaims` | Salidas del portero para atrapar balones altos. |
| `punches` | Despejes de puños del portero. |
| `successfulRunsOut` | Salidas exitosas del portero, según la fuente. |
| `cleanSheet` | Partidos sin encajar gol, según cómo lo codifique el dataset. |
| `ownGoals` | Goles en propia puerta. |
| `penaltyConceded` | Penaltis concedidos. |
| `possessionWonAttThird` | Recuperaciones/posesiones ganadas en el tercio ofensivo. |

Las estadísticas aditivas se suman por categoría. No se promedian entre partidos ni entre competiciones.

### 4.2 Porcentajes y tasas

El modelo calcula seis porcentajes a partir de numeradores y denominadores agregados:

| Característica | Numerador | Denominador |
|---|---|---|
| `accuratePassesPercentage` | `accuratePasses` | `totalPasses` |
| `successfulDribblesPercentage` | `successfulDribbles` | `totalDribbles` |
| `aerialDuelsWonPercentage` | `aerialDuelsWon` | `totalAerialDuels` |
| `totalDuelsWonPercentage` | `totalDuelsWon` | `totalDuels` |
| `tacklesWonPercentage` | `tacklesWon` | `totalTackles` |
| `goalConversionPercentage` | `goals` | `totalShots` |

La fórmula usada al agregar una categoría es:

```text
porcentaje agregado = 100 × suma(numeradores) / suma(denominadores)
```

Así, el porcentaje global no es la media simple de porcentajes de distintas filas: se calcula a partir de los totales. Si el denominador es cero, se asigna 0; si no se dispone de datos suficientes, puede quedar como `NaN`.

Si faltan columnas de denominador, el código intenta inferirlas usando el numerador y el porcentaje disponible. Si el porcentaje viene expresado entre 0 y 1, lo convierte a escala 0–100 antes de inferir el denominador.

### 4.3 Goles y asistencias por minuto

También se construye:

```text
goalsAssistsPerMins = suma(goalsAssistsSum) / suma(minutesPlayed)
```

Se calcula por categoría de competición. Si el total de minutos es cero o falta la información necesaria, la tasa queda como dato ausente y después recibe el tratamiento de `NaN` descrito en la sección de normalización.

### 4.4 Resultado de la competición

El resultado de cada competición también es una característica numérica del modelo. El código convierte los textos de `resultado_competicion` a valores ordinales y, para cada jugador y categoría de competición, conserva el **máximo** valor ordinal observado.

La conversión definida por `MAPA_RESULTADO_ORDINAL` incluye, entre otros, estos valores:

| Resultado | Valor |
|---|---:|
| Ganador | 1,00 |
| Final | 0,90 |
| Partido por el tercer puesto | 0,80 |
| Semifinales | 0,80 |
| Cuartos de final | 0,70 |
| Octavos de final | 0,60 |
| Dieciseisavos de final | 0,50 |
| Ronda intermedia / Repechaje | 0,40 |
| Fase de grupos | 0,30 |
| 7.ª o 6.ª ronda | 0,20 |
| 5.ª o 4.ª ronda | 0,10 |
| 3.ª o 2.ª ronda | 0,05 |

Además, los valores textuales numéricos se mapean así: `1` → 1,00; `2` → 0,90; `3` → 0,80; `4` → 0,70; `5` → 0,60; `6` → 0,50; `7` → 0,40; `8` → 0,30; `9` → 0,20; `10` → 0,10; `11` → 0,075; `17` → 0,05; `20` → 0,025. Estos valores son los que define el diccionario, con independencia de cómo se interpreten en el origen. El diccionario del código es la referencia definitiva para todas las equivalencias.

El resultado se transforma en un *z-score* igual que las demás características. Por ello, el modelo no usa directamente el valor bruto `1.0`, sino su posición relativa dentro de la distribución de esa característica y temporada.

**Advertencia:** el máximo ordinal dentro de una categoría puede combinar resultados de varias competiciones agrupadas en la misma categoría, por ejemplo las cinco grandes ligas. La interpretación de `resultado_competicion` depende de que el dataset lo codifique de manera coherente para las distintas clases de competición.

---

## 5. Categorías de competición

El diccionario `CATEGORIAS_COMPETICION` agrupa competiciones que considera comparables. Esto permite que varias competiciones compartan una familia de características, mientras que otras mantienen un bloque propio.

### Ligas

- `5_grandes_ligas`: Bundesliga, LaLiga, Ligue 1, Premier League y Serie A.
- `ligas_secundarias`: Campeonato Brasileiro Série A, Eredivisie, Liga Portugal y Saudi Pro League.

### Copas nacionales y supercopas

- `copa_nacional_principal_5grandes`: Copa del Rey, Coppa Italia, Coupe de France, DFB Pokal y FA Cup.
- `copa_de_la_liga_5grandes`: Coupe de la Ligue y EFL Cup.
- `supercopa_5grandes_ligas`: Community Shield, Franz Beckenbauer Supercup, Supercopa de España, Supercoppa Italiana y Trophée des Champions.
- `copa_nacional_principal_ligas_secundarias`: KNVB Beker y King's Cup.
- `supercopa_ligas_secundarias`: Johan Cruijff Schaal, Saudi Super Cup y Supertaça.

### Competiciones continentales y de clubes

- `uefa_champions_league`: UEFA Champions League.
- `uefa_europa_league`: UEFA Europa League.
- `uefa_conference_league`: UEFA Conference League.
- `fifa_club_world_cup`: FIFA Club World Cup.
- `fifa_intercontinental_cup`: FIFA Intercontinental Cup.
- `concacaf_champions_cup`: CONCACAF Champions Cup.
- `competiciones_anual_continental_clubes`: AFC Champions League Two y UEFA Super Cup.

### Selecciones nacionales

- `fifa_world_cup`: FIFA World Cup.
- `clasificacion_world_cup`: World Cup Qual. AFC, CAF, CONMEBOL y UEFA.
- `torneos_selecciones_continentales_tier1`: Copa América y EURO.
- `torneos_selecciones_continentales_tier2`: AFC Asian Cup, Africa Cup of Nations y UEFA Nations League.
- `confederations_cup`: Confederations Cup.
- `olympic_games`: Olympic Games.
- `clasificacion_torneos_continentales`: Africa Cup of Nations Qual. y EURO, Qualification.

Las competiciones no reconocidas por el diccionario reciben una categoría propia derivada de su nombre después de limpiar caracteres especiales. Por tanto, añadir una nueva competición al dataset puede generar nuevas características automáticamente.

---

## 6. Reglas de normalización con *z-scores*

La normalización es una parte central del sistema. Se realiza **por temporada y por característica**: no se calcula una única media histórica para todos los años.

Para cada temporada y característica, se calcula:

\[
z = \frac{x - \mu_{temporada, característica}}{\sigma_{temporada, característica}}
\]

Donde:

- `x` es el valor agregado del jugador en esa característica.
- `μ` es la media de los valores usados para el cálculo en esa temporada.
- `σ` es la desviación estándar poblacional (`ddof=0`) de esos valores.

Interpretación orientativa:

- `z = 0`: valor situado en la media calculada.
- `z > 0`: valor superior a esa media.
- `z < 0`: valor inferior a esa media.
- `z = 1`: una desviación estándar por encima de la media.
- `z = -1`: una desviación estándar por debajo de la media.

Los *z-scores* permiten que escalas diferentes —por ejemplo, goles, pases y porcentajes— entren en una función lineal sin que el número bruto de una estadística domine únicamente por su unidad de medida. La comparación es relativa a la distribución de cada característica en cada temporada; un `z-score` no representa una cantidad fija de goles o acciones.

### 6.1 Qué jugadores participan en la media y desviación

Para cada característica, el código identifica la categoría de competición a la que pertenece y comprueba si cada jugador participó en esa categoría.

**Regla A: valor observado.** Si la estadística tiene un valor numérico, ese valor se utiliza para calcular la media y la desviación.

**Regla B: jugador que no participó en la competición.** Si el valor es `NaN` y el jugador no figura como participante en la categoría correspondiente, el código sustituye ese `NaN` por `0` **solo al calcular la media y la desviación**.

**Regla C: jugador que sí participó, pero no tiene la estadística.** Si el jugador participó y el valor de la estadística es `NaN`, ese dato se excluye del cálculo de la media y la desviación.

**Regla D: cinco estadísticas exclusivas de portero.** Para `saves`, `goalsPrevented`, `highClaims`, `punches` y `successfulRunsOut`, todos los `NaN` cuentan como cero al calcular la media y desviación, independientemente de la participación.

### 6.2 Qué valor recibe finalmente un `NaN`

Una vez calculados la media y la desviación, **todo valor que originalmente era `NaN` recibe un z-score de `0`**. En la matriz del modelo también se convierten en cero los posibles valores `NaN`, infinitos positivos e infinitos negativos.

Esto significa que un dato ausente se representa en el modelo como una contribución neutral, equivalente a estar en el centro de la escala normalizada. Hay un matiz importante: aunque algunos `NaN` se hayan tratado como cero para calcular `μ` y `σ`, su valor final **no** es `(0 - μ) / σ`; se fuerza a `z = 0`.

Por tanto, el modelo no distingue mediante esa característica entre:

- un valor observado que coincide exactamente con la media; y
- un valor que estaba ausente y fue imputado con `z = 0`.

Esta decisión evita que los `NaN` lleguen a la optimización, pero puede ocultar diferencias entre ausencia real de participación, falta de cobertura estadística y un dato verdaderamente medio.

### 6.3 Desviación nula o falta de valores

Si no hay valores válidos para calcular parámetros, o si la desviación es cero o prácticamente cero (`<= 1e-12`), los *z-scores* de esa característica se quedan en cero. No hay variación que permita distinguir jugadores con esa característica en esa temporada.

### 6.4 Normalización durante la validación

El código calcula los parámetros de normalización dentro de cada temporada utilizando el conjunto de jugadores que aparece en esa temporada, no una media y desviación calculadas exclusivamente en el bloque de entrenamiento histórico. No utiliza las posiciones reales para calcular los *z-scores*, pero sí utiliza la distribución de características de los candidatos de la temporada evaluada.

Esto es una normalización relativa dentro de la edición: puede ser apropiada si en el momento de predecir se conocen todos los candidatos y sus estadísticas, pero no equivale a un proceso de predicción estrictamente inductivo que normaliza cada temporada únicamente con parámetros estimados a partir de temporadas anteriores.

---

## 7. Tratamiento específico de los porteros

El código intenta evitar que determinadas estadísticas de jugadores de campo se apliquen a porteros.

Las estadísticas permitidas explícitamente para porteros son:

- `saves`
- `goalsPrevented`
- `goalsConceded`
- `highClaims`
- `punches`
- `successfulRunsOut`
- `cleanSheet`
- `ownGoals`
- `penaltyConceded`
- `errorLeadToGoal`
- `errorLeadToShot`
- `accuratePassesPercentage`
- `resultado_competicion` (se permite siempre)

Las cinco estadísticas exclusivas de portero —paradas, goles evitados, balones altos atrapados, puñetazos y salidas exitosas— reciben además la regla especial de imputación para la normalización explicada anteriormente.

El filtrado se aplica a las filas agregadas cuyo valor de `posicion` es exactamente `Goalkeeper`. Como las etiquetas de posición no se mapean en este script, cualquier variante del texto debe normalizarse previamente en el dataset.

---

## 8. Cómo se incorporan las posiciones al modelo

El modelo construye una copia de cada característica para cada posición. Si hay `P` posiciones y `F` características base, el modelo tiene `P × F` coeficientes.

Los nombres internos siguen este patrón:

```text
Forward::5_grandes_ligas_goals__zscore
Midfielder::5_grandes_ligas_goals__zscore
Defender::5_grandes_ligas_goals__zscore
Goalkeeper::5_grandes_ligas_goals__zscore
```

Cada jugador activa exclusivamente el bloque correspondiente a su posición. Los bloques de las demás posiciones valen cero para ese jugador.

La puntuación de un jugador `i` es:

\[
S_i = \sum_k x_{ik}\,w_{posición(i),k}
\]

Donde `x` son los *z-scores* y `w` son los coeficientes aprendidos para la posición del jugador.

Esto permite que, por ejemplo, la misma característica tenga una magnitud de coeficiente diferente para un delantero y para un centrocampista. Sin embargo, no se introduce una penalización que obligue a que los coeficientes de posiciones distintas sean parecidos: cada posición tiene sus propios coeficientes y recibe la regularización L2 general.

Los pares de entrenamiento se forman entre **todos los jugadores de una misma temporada**, incluidos los de posiciones diferentes. Así, el modelo aprende en un único objetivo cómo ordenar conjuntamente delanteros, centrocampistas, defensas y porteros, no cuatro rankings aislados.

No hay un término independiente (*intercepto*) explícito. En una comparación por diferencias de puntuación, un intercepto común se cancelaría al restar el score de un jugador al del otro.

---

## 9. Cómo aprende: comparaciones por pares

### 9.1 Creación de los pares

En cada temporada, el script toma los jugadores que tienen una `posicion_balon_oro` conocida y genera todas las parejas posibles sin repetir el orden de la pareja. Si hay `n` jugadores, antes de eliminar empates se generan hasta:

\[
\frac{n(n-1)}{2}
\]

pares.

Se descartan las parejas empatadas en la posición real. Para cada pareja `(i, j)`:

- Las entradas de la matriz son la diferencia de características: `X[i] - X[j]`.
- La etiqueta vale `1` si `i` quedó por delante de `j`, porque su posición real es numéricamente menor.
- La etiqueta vale `0` en el caso contrario.

El modelo aprende a dar una diferencia de puntuación positiva al jugador que terminó mejor clasificado.

### 9.2 Probabilidad logística del duelo

Para dos jugadores, la probabilidad que representa el modelo es:

\[
P(i \text{ por delante de } j) = \sigma(S_i - S_j)
\]

con la función sigmoide:

\[
\sigma(t) = \frac{1}{1 + e^{-t}}
\]

Si `S_i - S_j` es positivo, la probabilidad logística del duelo supera el 50 %. Si es negativo, queda por debajo del 50 %. La magnitud de la diferencia determina cuán extrema es esa probabilidad.

### 9.3 Función de pérdida

El ajuste minimiza una pérdida logística binaria ponderada y añade regularización L2:

\[
L(w) = \frac{\sum_p a_p\,\ell(y_p,\,x_p w)}{\sum_p a_p}
       + \frac{1}{2C}\sum_j w_j^2
\]

Donde:

- `p` identifica un duelo.
- `a_p` es el peso de ese duelo.
- `y_p` indica cuál de los dos jugadores quedó por delante.
- `x_p` es la diferencia entre sus vectores de características.
- `w` es el vector de coeficientes.
- `C` controla la fuerza de la regularización.

La primera parte penaliza los duelos que el modelo ordena mal o con poca separación. La regularización L2 penaliza coeficientes de gran magnitud y ayuda a limitar el sobreajuste.

### 9.4 Peso de cada duelo y aproximación a Delta-NDCG

Los pares no pesan todos lo mismo. El código asigna un peso estático inspirado en Delta-NDCG para dar más importancia relativa a ciertos intercambios de posiciones que afectan más a la calidad de un ranking.

El procedimiento es:

1. Convierte la posición oficial de cada jugador en un rango relativo dentro de la edición, de `1` a `n`.
2. Asigna relevancias lineales entre `4` para el rango superior y `0` para el último.
3. Convierte la relevancia en ganancia con `2^relevancia - 1`.
4. Calcula descuentos `1 / log2(rango + 1)`.
5. Calcula una diferencia de ganancia-descuento para cada pareja, la divide por el IDCG cuando corresponde y aplica una raíz cúbica (`np.cbrt`).
6. Normaliza los pesos para que la suma de pesos de cada temporada sea `1`.

Normalizar por temporada evita que una edición con muchos más pares contribuya automáticamente con más peso total que otra edición. En la función de pérdida, la suma total de pesos es la suma de las contribuciones normalizadas de las temporadas incluidas.

**No es LambdaRank ni una implementación dinámica de Delta-NDCG.** Los pesos se calculan una sola vez con el ranking real; no se recalculan en cada iteración en función del ranking provisional del modelo. En concreto, tanto las ganancias como los descuentos usados por esta fórmula se derivan de los rangos reales relativos, no de las posiciones actuales predichas.

---

## 10. Restricciones monótonas

El código define estas estadísticas como negativas:

```python
ESTADISTICAS_NEGATIVAS = {
    'bigChancesMissed',
    'dispossessed',
    'dribbledPast',
    'errorLeadToGoal',
    'errorLeadToShot',
    'goalsConceded',
    'ownGoals',
    'penaltyConceded',
}
```

Para cada característica cuyo nombre termina en una de estas estadísticas —después de quitar `__zscore`— el coeficiente tiene límite superior `-epsilon`. Para todas las demás características el coeficiente tiene límite inferior `+epsilon`.

En términos prácticos:

- Estadística definida como positiva: `coeficiente >= epsilon`.
- Estadística definida como negativa: `coeficiente <= -epsilon`.

Las restricciones son **duras**: el optimizador no puede elegir el signo contrario. Esto significa que, manteniendo constantes las demás características, aumentar una variable positiva no puede reducir el score a través de su contribución lineal; aumentar una variable negativa no puede aumentarlo.

La lista de estadísticas negativas es la única excepción: todo lo demás queda tratado como positivo por defecto. Esto incluye los porcentajes, las tasas por minuto y `resultado_competicion`.

### El papel de `epsilon`

`epsilon` define la magnitud mínima absoluta de cada coeficiente. Con `epsilon = 0`, las restricciones permiten que un coeficiente llegue a cero. Sin embargo, **la búsqueda de Optuna de este script no prueba cero**: usa exclusivamente `[1e-6, 1e-5, 1e-4, 1e-3, 1e-2]`. Por tanto, en los modelos que selecciona esta configuración, todos los coeficientes quedan obligados a estar al menos a esa distancia de cero, con el signo establecido.

Esto es importante: la regularización no puede anular por completo una característica mientras `epsilon` sea estrictamente positivo. Una característica con poca utilidad puede acabar con un peso mínimo impuesto por la restricción. Si se quiere permitir que el modelo descarte características por completo, habría que incluir `epsilon = 0` en la búsqueda o fijarlo manualmente.

Estas reglas no deciden qué peso es mayor en términos absolutos. El entrenamiento sigue buscando la magnitud que mejor ajusta los duelos históricos, dentro de los límites permitidos y bajo regularización.

---

## 11. Optimizador y regularización

El vector de coeficientes se optimiza con `scipy.optimize.minimize` usando:

```python
method='L-BFGS-B'
jac=True
maxiter=10_000_000
maxls=100
```

La función de pérdida y su gradiente se calculan manualmente. Se utiliza `np.logaddexp` para evaluar de forma estable la pérdida logística y `expit` para la sigmoide. Los logits de la sigmoide se recortan al intervalo `[-50, 50]` al calcular el gradiente y las probabilidades, para reducir problemas numéricos.

### Hiperparámetros buscados

| Hiperparámetro | Valores/rango | Función |
|---|---|---|
| `C` | Entre `0.1` y `100`, en escala logarítmica | Regula la penalización L2. Menor `C` implica regularización más fuerte; mayor `C`, más débil. |
| `ftol` | Valores de `1e-15` a `1e-5` | Criterio de tolerancia relativo para la mejora de la función objetivo. |
| `gtol` | Valores de `1e-15` a `1e-5` | Tolerancia asociada al gradiente proyectado en L-BFGS-B. |
| `epsilon` | `1e-6`, `1e-5`, `1e-4`, `1e-3`, `1e-2` | Límite mínimo de la magnitud de cada coeficiente, además de su signo. |

`MAX_ITER_OPTIMIZACION` establece un máximo de diez millones de iteraciones; es un límite máximo, no una promesa de que se realicen todas. La optimización puede terminar antes por los criterios de convergencia.

El código considera fallido un ensayo si el optimizador devuelve `resultado.success == False`: se lanza un error y Optuna poda ese ensayo. Por tanto, conviene revisar cuántos ensayos se completan, se podan y qué mensajes de convergencia aparecen.

---

## 12. Las dos fases de Optuna y la separación temporal

Optuna maximiza la media del `NDCG@30` en temporadas de validación temporal. Para cada temporada usada como validación, el entrenamiento solo contiene temporadas estrictamente anteriores a ella (*walk-forward*).

### Fase A: elegir parámetros para la evaluación independiente

- Límite de temporadas candidatas para validación: `2019/20`, incluido.
- Se conservan solo folds con temporadas anteriores para entrenar y al menos dos posiciones reales distintas tanto en entrenamiento como en validación.
- Se solicitan `500` ensayos.
- Los parámetros seleccionados se guardan en `optuna_best_params_solo_temporada_evaluacion_hasta_2019_20.csv`.

Con los parámetros de esta fase se genera el backtest independiente de `2020/21`, `2021/22`, `2022/23`, `2023/24` y `2024/25`. Esas temporadas no se emplean como folds de validación en la Fase A.

### Fase B: elegir parámetros para la predicción 2025/26

- Se consideran como folds todas las temporadas históricas elegibles anteriores a `2025/26`.
- En cada fold se entrena con las temporadas anteriores al fold y se valida sobre esa temporada.
- Se solicitan otros `500` ensayos.
- Los mejores parámetros se guardan en `optuna_best_params_solo_temporada_prediccion_2025_26.csv`.

Después se entrena un modelo final con todas las temporadas etiquetadas anteriores a `2025/26` y se calculan los scores de los candidatos de `2025/26`.

Las dos fases tienen finalidades diferentes: la Fase A permite evaluar el procedimiento sobre temporadas posteriores que no intervinieron en su búsqueda de parámetros; la Fase B reutiliza la historia disponible para elegir los parámetros de la predicción final.

El muestreador es `TPESampler()` sin semilla fija. En consecuencia, una nueva ejecución puede explorar otros ensayos y no necesariamente seleccionar exactamente los mismos parámetros.

---

## 13. Métricas de evaluación

Las métricas se calculan comparando los scores del modelo con las posiciones reales disponibles.

### NDCG@30

Mide la calidad del orden, dando más valor a colocar arriba a los jugadores con mayor relevancia. En este código, las posiciones reales se convierten en rangos relativos dentro de la edición y se transforman en relevancias lineales entre `4` y `0`. Las ganancias se calculan exponencialmente (`2^rel - 1`), y la evaluación utiliza `k = min(30, número de jugadores evaluados)`.

Esta es la métrica que Optuna intenta maximizar.

### Spearman

Mide la correlación por rangos entre la posición real (invertida para que los puestos mejores tengan mayor valor) y el score del modelo. Está cerca de `1` cuando existe una relación monótona positiva fuerte entre el orden real invertido y los scores.

### MAE y RMSE de la posición predicha

El código ordena los scores de mayor a menor y asigna puestos predichos secuenciales de `1` a `n`.

- `MAE`: error absoluto medio entre el puesto real numérico y el puesto predicho.
- `RMSE`: raíz del error cuadrático medio, que penaliza más los errores grandes.

Debe interpretarse con cuidado si hay puestos oficiales ausentes, empates o solo una parte de los candidatos tiene posición real conocida: la posición predicha se calcula como un orden secuencial entre las filas evaluadas, mientras que MAE/RMSE comparan contra el valor numérico original de `posicion_balon_oro`.

### Resumen de entrenamiento y validación

Para cada temporada del backtest, se calculan métricas de entrenamiento promediando sus resultados por edición y métricas de validación para la temporada correspondiente. `gap_ndcg30` es la diferencia entre el NDCG medio del conjunto de entrenamiento y el NDCG de validación; una brecha grande puede ser una señal de sobreajuste, aunque por sí sola no lo demuestra.

---

## 14. Scores y probabilidades exportadas

### `score_modelo` y `score_ranking`

Ambos campos contienen el score logístico nativo del modelo. En esta versión:

```python
scores_finales_ranking = scores_finales_modelo
```

Por tanto, no hay una segunda transformación del score: `score_ranking` es una copia de `score_modelo`. El orden final se obtiene ordenando de mayor a menor.

Los scores son valores relativos y no tienen una escala de puntos con interpretación universal. No se deben comparar directamente entre temporadas como si estuvieran calibrados en una escala común, especialmente porque las características se normalizan de forma independiente por temporada.

### `prob_duelo_contra_lider` y `prob_duelo_contra_anterior`

Para cada jugador, una vez ordenado el ranking:

- `prob_duelo_contra_lider` usa `sigmoid(score_jugador - score_líder)`.
- `prob_duelo_contra_anterior` usa `sigmoid(score_jugador - score_del_jugador_inmediatamente_anterior)`.

El primer jugador no tiene un líder ni un jugador anterior, por lo que ambas celdas quedan vacías/`NaN`.

Estas cifras son probabilidades logísticas **derivadas de la diferencia de scores** para esos duelos concretos. El código no aplica calibración externa, no ajusta una proyección Bradley–Terry adicional y no calcula probabilidades calibradas de ganar el Balón de Oro. Tampoco son probabilidades de terminar exactamente en el puesto mostrado.

La función `calcular_porcentajes_softmax` está definida, pero no se utiliza en el flujo de predicción actual.

---

## 15. Interpretación de coeficientes e importancia

Al terminar, los coeficientes se reorganizan por posición y característica. Para cada posición, el código calcula una medida descriptiva de impacto:

\[
I_{pos,k} = \operatorname{media}_{i \in pos}\bigl(|z_{ik}|\bigr)\,|w_{pos,k}|
\]

Es decir:

1. Calcula el valor absoluto de los *z-scores* de la característica para los jugadores de esa posición en `train_final`.
2. Calcula su media.
3. La multiplica por el valor absoluto del coeficiente correspondiente.

Después normaliza esa medida dentro de cada posición para producir `peso_relativo_porcentaje`, de modo que los impactos sumen el 100 % por posición cuando su suma es positiva.

### Columnas de los CSV

- `modelo_normalizacion`: etiqueta `zscore_solo_temporada`.
- `posicion`: posición cuyo bloque de coeficientes se está describiendo.
- `caracteristica`: nombre de característica normalizada, incluido `__zscore`.
- `estadistica`: nombre sin el sufijo `__zscore`.
- `coeficiente`: valor del coeficiente ajustado.
- `signo_monotonico`: signo esperado según la restricción configurada.
- `impacto_medio_absoluto`: medida `media(abs(z)) * abs(coeficiente)`.
- `caracteristicas_utilizadas_posicion`: número de características cuyo impacto supera la tolerancia configurada.
- `coeficientes_no_nulos_posicion`: número de coeficientes cuya magnitud supera la tolerancia configurada.
- `peso_relativo_porcentaje`: impacto de la característica como porcentaje del impacto total de esa posición.

**Esta medida no es una importancia causal ni una prueba de qué ocurriría al eliminar una característica.** Tampoco es la magnitud del coeficiente sin más: combina el coeficiente con la variación observada en los jugadores de esa posición. Variables correlacionadas pueden repartirse la contribución; el impacto no garantiza que una variable sea independiente de las demás.

Los archivos `coeficientes_logistic_solo_temporada.csv` e `importancia_estadisticas_solo_temporada.csv` contienen el mismo DataFrame en esta versión; son dos copias del mismo contenido con nombres diferentes.

---

## 16. Comparaciones individuales

Se generan dos archivos de comparación para los candidatos de `2025/26`:

- `comparacion_mbappe_lamine_solo_temporada.csv`
- `comparacion_cubarsi_lamine_solo_temporada.csv`

Para cada característica se exportan:

- posición de cada jugador;
- valor bruto agregado;
- *z-score* de la temporada;
- coeficiente correspondiente a la posición de cada jugador;
- contribución de la característica al score de cada jugador (`z-score × coeficiente`);
- diferencia de contribuciones entre el jugador A y el B;
- valor absoluto de esa diferencia.

Las filas se ordenan de mayor a menor diferencia absoluta de contribución, facilitando localizar las características que más separan a los jugadores en este modelo.

Hay que distinguir entre la contribución individual de una característica y la diferencia total de score: la diferencia total se obtiene sumando las diferencias de contribución de todas las características. Una contribución positiva o negativa describe cómo esa variable suma o resta a la puntuación lineal respecto del centro de la normalización; no demuestra una relación causal.

---

## 17. Archivos que genera el script

| Archivo | Contenido |
|---|---|
| `resumen_backtest_logistic_solo_temporada.csv` | Métricas de entrenamiento y validación, brecha de NDCG, parámetros utilizados y ruta del ranking generado para cada temporada de backtest. |
| `optuna_trials_solo_temporada_evaluacion_hasta_2019_20.csv` | Historial de los ensayos de la Fase A, con valores de objetivo, parámetros, estado, duración y atributos del ensayo. |
| `optuna_best_params_solo_temporada_evaluacion_hasta_2019_20.csv` | Mejor resultado y parámetros encontrados para la evaluación independiente. |
| `optuna_trials_solo_temporada_prediccion_2025_26.csv` | Historial de ensayos de la Fase B. |
| `optuna_best_params_solo_temporada_prediccion_2025_26.csv` | Mejores parámetros elegidos para la predicción 2025/26. |
| `prediccion_solo_temporada_backtest_YYYY_YY.csv` | Ranking predicho para cada temporada de evaluación independiente. |
| `prediccion_solo_temporada_2025_26.csv` | Ranking final de candidatos de 2025/26. |
| `coeficientes_logistic_solo_temporada.csv` | Coeficientes y medidas descriptivas de impacto, por posición y característica. |
| `importancia_estadisticas_solo_temporada.csv` | Copia del archivo anterior. |
| `comparacion_mbappe_lamine_solo_temporada.csv` | Diferencias por característica entre Mbappé y Lamine Yamal si ambos se encuentran en los datos. |
| `comparacion_cubarsi_lamine_solo_temporada.csv` | Diferencias por característica entre Cubarsí y Lamine Yamal si ambos se encuentran en los datos. |

Los nombres de los archivos del backtest transforman la barra `/` de la temporada en guion bajo, por ejemplo `2020_21`.

---

## 18. Resumen del flujo completo

```text
dataset_ml_base.csv
        │
        ▼
Limpieza básica y normalización de tipos
        │
        ▼
Agregación por temporada + jugador + categoría de competición
        │
        ├── Estadísticas aditivas
        ├── Porcentajes calculados desde totales
        ├── Goles/asistencias por minuto
        └── Resultado ordinal de competición
        │
        ▼
Z-scores por temporada y característica
        │
        ▼
Matriz con un bloque de coeficientes por posición
        │
        ▼
Pares de jugadores etiquetados por el orden real
        │
        ▼
Pesos estáticos inspirados en Delta-NDCG
        │
        ▼
Regresión logística pareada + L2 + restricciones monótonas
        │
        ├── Optuna Fase A → parámetros para backtest independiente
        ├── Backtest 2020/21–2024/25
        └── Optuna Fase B → parámetros para predicción final
                               │
                               ▼
                  Entrenamiento con historia anterior a 2025/26
                               │
                               ▼
                  Score por candidato y ranking descendente
                               │
                               ▼
                  CSV de ranking, métricas, pesos y comparaciones
```

En pocas palabras: el modelo no aplica una tabla fija de puntos por goles, asistencias o títulos. Aprende coeficientes a partir de comparaciones entre jugadores históricos, pero lo hace bajo dos supuestos explícitos: la importancia puede cambiar según la posición y el signo de cada coeficiente está restringido por la lista de estadísticas positivas y negativas configurada en el código.
