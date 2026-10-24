# Predicción del Balón de Oro mediante Machine Learning

Proyecto de ciencia de datos que utiliza estadísticas futbolísticas históricas para generar rankings del Balón de Oro. El modelo aprende a comparar jugadores por parejas a partir de sus estadísticas y de las posiciones que obtuvieron en ediciones anteriores.

La solución combina dos modelos de regresión logística *pairwise*, restricciones monotónicas sobre determinadas estadísticas, optimización de hiperparámetros con Optuna y un sistema de ensamble que transforma las comparaciones entre jugadores en un ranking único.

## Contenido

- [Objetivo del proyecto](#objetivo-del-proyecto)
- [Cómo funciona el modelo](#cómo-funciona-el-modelo)
- [Datos de entrada](#datos-de-entrada)
- [Preparación de las estadísticas](#preparación-de-las-estadísticas)
- [Normalización y valores ausentes](#normalización-y-valores-ausentes)
- [Regresión logística por parejas](#regresión-logística-por-parejas)
- [Restricciones monotónicas](#restricciones-monotónicas)
- [Ensamble de los modelos](#ensamble-de-los-modelos)
- [Optimización con Optuna](#optimización-con-optuna)
- [Backtest y predicción final](#backtest-y-predicción-final)
- [Métricas de evaluación](#métricas-de-evaluación)
- [Interpretabilidad y comparaciones](#interpretabilidad-y-comparaciones)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Archivos generados](#archivos-generados)
- [Limitaciones](#limitaciones)

## Objetivo del proyecto

El propósito es aprender qué características estadísticas ayudan a ordenar a los jugadores según sus resultados históricos en el Balón de Oro y utilizar ese aprendizaje para generar rankings de temporadas posteriores.

En lugar de predecir directamente un puesto mediante un modelo de regresión convencional, el sistema aprende **qué jugador debería quedar por encima de otro**. Después combina las comparaciones para obtener una clasificación completa.

El modelo tiene en cuenta:

- Estadísticas de los jugadores agrupadas por tipo de competición.
- Diferencias estadísticas entre jugadores.
- El contexto de la posición del futbolista.
- El resultado alcanzado en las competiciones.
- Restricciones que impiden invertir el signo esperado de determinadas estadísticas.
- La calidad del ranking obtenida en temporadas históricas.

La temporada objetivo definida en el código es `2025/26`. Las temporadas configuradas para el backtest independiente son `2020/21`, `2021/22`, `2022/23`, `2023/24` y `2024/25`.

## Cómo funciona el modelo

El proceso general es el siguiente:

```text
Dataset histórico
       │
       ▼
Preparación de los datos y las estadísticas
       │
       ▼
Agrupación de competiciones y resultados
       │
       ▼
Agregación por jugador, temporada y categoría
       │
       ▼
Generación de dos versiones normalizadas
       │
       ├──────────────────┐
       ▼                  ▼
Modelo posicional    Modelo global
       │                  │
       └────────┬─────────┘
                ▼
     Comparaciones por parejas
                │
                ▼
   Regresión logística pairwise
   con restricciones monotónicas
                │
                ▼
     Probabilidades de ensamble
                │
                ▼
   Proyección Bradley–Terry
                │
                ▼
        Ranking final
                │
                ▼
 Rankings, métricas y análisis CSV
```

## Datos de entrada

El código necesita un dataset en esta ruta:

```text
data/dataset_ml_base.csv
```

Entre las columnas que utiliza se encuentran:

| Columna | Función |
|---|---|
| `temporada` | Identifica la edición o temporada. |
| `nombre` | Identifica al jugador para agrupar sus estadísticas. |
| `club` | Se conserva como información descriptiva. |
| `posicion` | Controla el bloque de coeficientes y la compatibilidad de las estadísticas. |
| `competicion` | Permite clasificar y agregar las estadísticas. |
| `resultado_competicion` | Se transforma en una variable ordinal. |
| `posicion_balon_oro` | Posición histórica utilizada como objetivo de aprendizaje. |

La posición real se interpreta de la forma habitual: **un valor numérico más pequeño significa un resultado mejor**. Por ejemplo, el puesto 2 es mejor que el puesto 10.

Al cargar los datos se eliminan las filas sin competición o sin posición futbolística. Las columnas relevantes se convierten al tipo de dato adecuado.

En la temporada `2025/26`, las posiciones del Balón de Oro menores o iguales que cero se convierten en valores ausentes, ya que el código las interpreta como posiciones todavía desconocidas.

## Preparación de las estadísticas

### 1. Agrupación de competiciones

El programa agrupa las competiciones en categorías antes de generar las características del modelo. Así, las estadísticas de varios torneos pueden combinarse en una misma variable.

Algunos ejemplos de las categorías implementadas son:

| Categoría | Ejemplos |
|---|---|
| `5_grandes_ligas` | Bundesliga, LaLiga, Ligue 1, Premier League y Serie A |
| `ligas_secundarias` | Eredivisie, Liga Portugal, Campeonato Brasileiro Série A y Saudi Pro League |
| `copa_nacional_principal_5grandes` | Copa del Rey, FA Cup, Coppa Italia, Coupe de France y DFB-Pokal |
| `uefa_champions_league` | UEFA Champions League |
| `uefa_europa_league` | UEFA Europa League |
| `uefa_conference_league` | UEFA Conference League |
| `fifa_world_cup` | FIFA World Cup |
| `torneos_selecciones_continentales_tier1` | Copa América y EURO |

El código también contempla supercopas, torneos de clubes FIFA, otras copas nacionales y competiciones de selecciones.

Si una competición no aparece en el diccionario de categorías, se utiliza su nombre normalizado como categoría alternativa.

Las características se crean con nombres similares a estos:

```text
5_grandes_ligas_goalsAssistsSum
uefa_champions_league_keyPasses
fifa_world_cup_resultado_competicion
```

### 2. Estadísticas aditivas

Las estadísticas aditivas se suman por jugador, temporada y categoría de competición. La lista configurada en el código es:

- `goalsAssistsSum`
- `bigChancesCreated`
- `bigChancesMissed`
- `keyPasses`
- `successfulDribbles`
- `accuratePasses`
- `totalAttemptAssist`
- `aerialDuelsWon`
- `totalDuelsWon`
- `tacklesWon`
- `interceptions`
- `outfielderBlocks`
- `ballRecovery`
- `dispossessed`
- `dribbledPast`
- `clearances`
- `errorLeadToGoal`
- `errorLeadToShot`
- `saves`
- `goalsPrevented`
- `goalsConceded`
- `highClaims`
- `punches`
- `successfulRunsOut`
- `cleanSheet`
- `ownGoals`
- `penaltyConceded`
- `possessionWonAttThird`

Si `goalsAssistsSum` no existe, pero están disponibles `goals` y `assists`, el código puede crearla sumando ambas columnas.

Conviene distinguir entre las columnas auxiliares y las características que utiliza realmente el modelo. Por ejemplo, `expectedGoals` y `expectedAssists` se convierten a valores numéricos si existen, pero **no forman parte de la lista de estadísticas aditivas utilizada como característica**.

Tampoco se incorporan directamente `appearances`, goles y asistencias como características independientes. Goles y asistencias intervienen en variables derivadas como `goalsAssistsSum` y `goalConversionPercentage`; los minutos se utilizan en el cálculo de `goalsAssistsPerMins`.

### 3. Cálculo correcto de los porcentajes

El modelo utiliza seis porcentajes:

| Característica | Numerador | Denominador |
|---|---|---|
| `accuratePassesPercentage` | `accuratePasses` | `totalPasses` |
| `successfulDribblesPercentage` | `successfulDribbles` | `totalDribbles` |
| `aerialDuelsWonPercentage` | `aerialDuelsWon` | `totalAerialDuels` |
| `totalDuelsWonPercentage` | `totalDuelsWon` | `totalDuels` |
| `tacklesWonPercentage` | `tacklesWon` | `totalTackles` |
| `goalConversionPercentage` | `goals` | `totalShots` |

Cuando se agregan los datos de distintas competiciones, el programa no calcula una media simple de los porcentajes. Suma primero los numeradores y denominadores y, a continuación, calcula el porcentaje:

\[
\text{Porcentaje} =
100 \times
\frac{\text{Numerador total}}{\text{Denominador total}}
\]

Por ejemplo, el porcentaje global de pases precisos se calcula a partir de todos los pases precisos y todos los pases intentados. De este modo, una competición con muchos intentos no recibe el mismo peso que otra con muy pocos.

Cuando falta un denominador, el código intenta inferirlo usando el numerador y el porcentaje disponibles. Si un porcentaje está expresado en una escala de 0 a 1, se convierte a la escala de 0 a 100.

### 4. Goles y asistencias por minuto

Se genera también la característica:

```text
goalsAssistsPerMins
```

La fórmula utilizada es:

\[
\text{goalsAssistsPerMins} =
\frac{\text{Goles y asistencias totales}}
{\text{Minutos jugados totales}}
\]

Es una tasa por minuto, no por cada 90 minutos: el código no multiplica el resultado por 90. Si los minutos totales son cero, evita la división y deja el valor ausente.

### 5. Resultado de las competiciones

El resultado alcanzado en cada competición se convierte en un valor ordinal. Algunos de los valores configurados son:

| Resultado | Valor ordinal |
|---|---:|
| Ganador | 1,00 |
| Final | 0,90 |
| Semifinales | 0,80 |
| Partido por el tercer puesto | 0,80 |
| Cuartos de final | 0,70 |
| Octavos de final | 0,60 |
| Fase de grupos | 0,30 |

El diccionario del código incluye otras rondas y algunos resultados expresados mediante números. Los valores que no estén contemplados quedan ausentes.

Si un jugador tiene varios resultados en competiciones de la misma categoría, el modelo conserva el valor ordinal máximo. Así, por ejemplo, si un jugador alcanza una final en una competición y queda eliminado antes en otra de esa misma categoría, se conserva el resultado más alto.

## Normalización y valores ausentes

Las estadísticas tienen escalas diferentes. Para hacerlas comparables, el programa utiliza una normalización *z-score*:

\[
z = \frac{x-\mu}{\sigma}
\]

Donde \(x\) es el valor del jugador, \(\mu\) es la media del grupo de referencia y \(\sigma\) es su desviación estándar poblacional.

El código genera dos versiones normalizadas de las mismas características.

### Modelo posicional

La media y la desviación se calculan por combinación de:

- Temporada.
- Posición.
- Característica.

Por ejemplo, las estadísticas de un delantero se normalizan tomando como referencia a los jugadores de su posición durante esa temporada.

### Modelo global

La media y la desviación se calculan por combinación de:

- Temporada.
- Característica.

En esta versión, la referencia incluye a todos los jugadores de la temporada, no solo a los de la misma posición.

**Los dos modelos aprenden comparaciones entre jugadores de toda la temporada.** Lo que cambia es la forma de normalizar las estadísticas, no el conjunto de posiciones que pueden enfrentarse.

### Máscaras para datos ausentes

Cada característica normalizada tiene una máscara asociada, como `__mask`, que determina si debe participar en la puntuación del jugador.

La lógica distingue entre estadísticas disponibles, valores que pueden tratarse como media normalizada y estadísticas que no son aplicables a una posición o competición. Al construir la matriz de características, el código multiplica cada valor normalizado por su máscara.

De esta manera, un valor ausente cuya máscara esté desactivada no se introduce directamente en el duelo como una diferencia estadística.

### Tratamiento de porteros

El código limita las estadísticas disponibles para los porteros. Entre las permitidas se encuentran:

- Paradas.
- Goles evitados.
- Goles encajados.
- Despejes de puños.
- Salidas.
- Porterías a cero.
- Algunos errores y el porcentaje de pases precisos.
- Resultados de competición.

Las estadísticas exclusivas de portero no se aplican como características a los jugadores de campo. Esto evita que una estadística no pertinente para una posición influya como si su ausencia fuese un rendimiento negativo.

## Regresión logística por parejas

El componente principal es una regresión logística *pairwise* con restricciones monotónicas.

En lugar de aprender únicamente un valor objetivo independiente para cada jugador, el modelo aprende relaciones del tipo:

> Dadas las estadísticas de dos jugadores de una misma edición, ¿cuál debería quedar mejor clasificado?

### Cómo se generan los duelos

Para cada temporada de entrenamiento se seleccionan los jugadores cuya posición real es conocida. Después se generan parejas sin repetición.

Se descartan las parejas en las que ambos jugadores tengan la misma posición real, puesto que no permiten determinar cuál quedó por encima del otro.

Para cada pareja:

- El objetivo es `1` si el primer jugador obtuvo una posición real mejor.
- El objetivo es `0` si el segundo jugador quedó mejor.
- Las características utilizadas son las diferencias entre sus estadísticas normalizadas.

La idea matemática simplificada es:

\[
P(A>B)=
\sigma\left(
\boldsymbol{\beta}^{T}
(\mathbf{x}_A-\mathbf{x}_B)
\right)
\]

Donde:

- \(\mathbf{x}_A-\mathbf{x}_B\) representa la diferencia entre las características de ambos jugadores.
- \(\boldsymbol{\beta}\) contiene los coeficientes aprendidos.
- \(\sigma\) es la función sigmoide, que devuelve valores entre 0 y 1.

### Coeficientes específicos por posición

El modelo crea un bloque de coeficientes por cada posición. Cada jugador activa el bloque correspondiente a la suya, por lo que una misma estadística puede tener coeficientes diferentes en delanteros, centrocampistas, defensas y porteros.

Esto permite aprender una relación distinta entre las estadísticas y el ranking según la posición sin tener que entrenar un modelo completamente separado para cada una.

### Regularización

La función de pérdida combina la pérdida logística ponderada de los duelos con una penalización L2:

\[
L =
L_{\text{logística ponderada}} +
\frac{1}{2C}\sum_j \beta_j^2
\]

La regularización limita la magnitud de los coeficientes y ayuda a evitar el sobreajuste. En términos generales, un `C` menor aplica una regularización más fuerte y un `C` mayor permite coeficientes de mayor magnitud.

La optimización se realiza mediante `L-BFGS-B` de SciPy, que permite establecer límites para los coeficientes.

## Restricciones monotónicas

El modelo incorpora restricciones que fijan el signo de determinados coeficientes.

Las estadísticas definidas como negativas son:

- `bigChancesMissed`
- `dispossessed`
- `dribbledPast`
- `errorLeadToGoal`
- `errorLeadToShot`
- `goalsConceded`
- `ownGoals`
- `penaltyConceded`

Para esas características, el coeficiente debe ser menor o igual que `-epsilon`. El resto se considera positivo por defecto y sus coeficientes deben ser mayores o iguales que `epsilon`.

La finalidad es impedir que el modelo aprenda una relación contraria a la regla establecida. Por ejemplo, más errores graves no pueden recibir un coeficiente positivo por una correlación accidental presente en el histórico.

Estas restricciones son duras: el optimizador no puede invertir los signos definidos en el código. Esto aporta coherencia con las reglas del proyecto, pero también limita lo que el modelo puede aprender a partir de los datos.

## Ponderación de las parejas

No todas las comparaciones tienen necesariamente el mismo impacto en la calidad del ranking. Para reflejarlo, el código calcula pesos inspirados en **Delta-NDCG**.

Utiliza el orden real de los jugadores, sus relevancias, las ganancias y los descuentos por posición para aproximar cuánto cambiaría el NDCG al intercambiar dos jugadores. Aplica una raíz cúbica a la diferencia relativa y normaliza los pesos para que cada temporada aporte un peso total equivalente.

Esto evita que una temporada con un número mucho mayor de parejas domine el entrenamiento únicamente por tener más combinaciones posibles.

La implementación es una aproximación estática inspirada en Delta-NDCG, no una versión dinámica de LambdaRank. Los pesos no se recalculan en cada iteración a partir del ranking provisional del modelo.

## Ensamble de los modelos

El sistema combina dos modelos:

1. **Modelo posicional:** utiliza las características normalizadas por temporada y posición.
2. **Modelo global:** utiliza las características normalizadas por temporada.

Ambos producen scores que permiten calcular probabilidades de duelo.

### 1. Combinar probabilidades

El peso configurado para el modelo posicional es `0.5`:

\[
P_{\text{ensamble}}(A>B) =
0.5\,P_{\text{posicional}}(A>B) +
0.5\,P_{\text{global}}(A>B)
\]

El programa calcula estas probabilidades para las parejas de jugadores y construye una matriz de probabilidades de ensamble. La diagonal, que representa la comparación de un jugador consigo mismo, se fija en 0,5.

### 2. Convertir las comparaciones en un ranking consistente

La combinación de dos modelos puede producir ciclos: A puede superar a B, B a C y C a A. Un ranking único necesita una puntuación que establezca un orden coherente.

Para resolverlo, el código utiliza una proyección basada en el modelo **Bradley–Terry**. Este procedimiento busca una utilidad escalar por jugador que se aproxime a las probabilidades originales de la matriz combinada.

Las utilidades resultantes se centran alrededor de cero y se utilizan como `score_ensamble`. Los jugadores se ordenan de mayor a menor puntuación.

### Interpretación de las columnas

| Columna | Significado |
|---|---|
| `score_modelo_posicional` | Score lineal del modelo con normalización por temporada y posición. |
| `score_modelo_global` | Score lineal del modelo con normalización por temporada. |
| `score_ensamble` | Utilidad de la proyección Bradley–Terry que se utiliza para ordenar a los jugadores. |
| `posicion_predicha` | Puesto asignado por el ranking, empezando en 1. |

El `score_ensamble` no es la media de los dos scores originales ni una probabilidad. Es la utilidad calculada por la proyección que permite ordenar las comparaciones de forma transitiva.

### Probabilidades contra el líder y contra el jugador anterior

Una vez ordenados los candidatos, el programa calcula:

\[
P(i>\text{líder}) =
\sigma(s_i-s_{\text{líder}})
\]

\[
P(i>\text{anterior}) =
\sigma(s_i-s_{i-1})
\]

Estas probabilidades se derivan de las utilidades del ensamble. El primer clasificado no tiene un líder ni un jugador anterior, por lo que sus valores quedan ausentes.

No representan la probabilidad de ganar el premio completo ni se calibran mediante una fase externa con probabilidades observadas.

## Optimización con Optuna

El código realiza dos búsquedas separadas para evitar seleccionar los parámetros del backtest usando los resultados de las propias temporadas que se quieren evaluar.

En ambas fases se optimiza la media del `NDCG@30` del ensamble utilizando validación cronológica (*walk-forward*). Para validar una temporada, solo se entrena con temporadas anteriores que tienen posiciones reales conocidas.

### Fase A: parámetros del backtest

La primera búsqueda utiliza temporadas candidatas hasta `2019/20`. Los folds deben tener datos suficientes tanto para el entrenamiento como para la validación.

Los hiperparámetros son:

| Parámetro | Rango o valores |
|---|---|
| `C` | Entre 1 y 75, mediante muestreo logarítmico |
| `ftol` | Valores discretos entre `1e-18` y `1e-1` |
| `gtol` | Valores discretos entre `1e-18` y `1e-1` |
| `epsilon` | Valores entre `1e-10` y `1e-2` |

Se solicitan 500 ensayos a Optuna y se utiliza una semilla `42` para el sampler TPE. Los parámetros seleccionados se reservan para la evaluación independiente de 2020/21–2024/25.

### Fase B: parámetros para la predicción final

La segunda búsqueda utiliza las temporadas históricas elegibles anteriores a `2025/26` como folds de validación cronológica. También solicita 500 ensayos y optimiza la media del `NDCG@30`.

Los parámetros ganadores de esta fase se utilizan para entrenar los modelos finales destinados a la predicción de `2025/26`.

Las dos búsquedas pueden ser costosas: cada ensayo puede entrenar dos modelos en varios folds temporales y cada entrenamiento genera parejas de jugadores.

## Backtest y predicción final

### Backtest independiente

El código evalúa estas temporadas:

- 2020/21
- 2021/22
- 2022/23
- 2023/24
- 2024/25

Para cada edición, los dos modelos se entrenan con las temporadas anteriores disponibles y etiquetadas. Se utilizan los parámetros obtenidos en la Fase A, sin utilizar los resultados de la edición evaluada para elegirlos.

Un matiz importante: el código construye el ranking de backtest con jugadores cuya `posicion_balon_oro` es conocida. Por tanto, la evaluación se realiza sobre los jugadores etiquetados, que no tienen por qué ser todos los nominados de esa edición.

### Predicción de 2025/26

Para el ranking final, el programa:

1. Selecciona temporadas históricas anteriores a `2025/26` con posiciones reales conocidas.
2. Entrena los modelos posicional y global con los parámetros de la Fase B.
3. Predice todos los candidatos disponibles en el dataset para `2025/26`, incluso si su puesto real todavía es desconocido.
4. Combina las probabilidades de duelo y proyecta el resultado a un ranking transitivo.
5. Exporta el ranking y los scores.

Los datos de resultado de `2025/26` no se usan como etiquetas para entrenar esta predicción. La evaluación de ese ranking necesitará una posición real fiable y una evaluación posterior.

## Métricas de evaluación

### NDCG@30

Evalúa la calidad del orden, dando mayor importancia a colocar correctamente a los jugadores mejor clasificados. El código convierte las posiciones reales en valores de relevancia y calcula la métrica hasta los primeros 30 puestos, o hasta el número de jugadores disponibles si es menor.

Cuanto más cerca de 1, mejor. Es la métrica principal utilizada por Optuna.

### Correlación de Spearman

Mide la relación entre el orden real y el score del modelo. El código utiliza el negativo de la posición real para que un score mayor se corresponda con un puesto mejor.

Un valor alto y positivo indica que los jugadores mejor clasificados tienden a obtener scores más altos.

### MAE

El error absoluto medio entre la posición predicha y la posición real. En general, un valor menor es mejor.

### RMSE

La raíz del error cuadrático medio entre el puesto predicho y el real. Penaliza los errores grandes más que el MAE. También se busca un valor bajo.

El código calcula métricas por temporada y, en el resumen de backtest, guarda los resultados del ensamble, los de cada modelo por separado y la diferencia entre NDCG de entrenamiento y validación.

## Interpretabilidad y comparaciones

Al finalizar el entrenamiento de predicción, se exportan los coeficientes de ambos modelos por posición.

Para estimar la importancia de una característica, el código utiliza:

\[
\text{Impacto medio absoluto} =
\operatorname{media}(|z|)\times|\text{coeficiente}|
\]

Aquí, \(z\) es la característica normalizada después de aplicar su máscara.

Después calcula el porcentaje de importancia relativa de cada característica dentro de cada combinación de modelo y posición, dividiendo su impacto entre la suma de impactos de ese grupo.

Estas medidas permiten explorar qué variables intervienen más en el cálculo del modelo. No demuestran causalidad ni garantizan que una estadística, de forma aislada, determine la posición final.

El programa también genera comparaciones de contribuciones entre:

- Kylian Mbappé y Lamine Yamal.
- Pau Cubarsí y Lamine Yamal.

Siempre que encuentre a ambos jugadores en la predicción final, exporta sus valores reales, valores normalizados, coeficientes y contribuciones de ambos modelos. Las filas se ordenan por la mayor diferencia absoluta de contribución.

## Instalación y ejecución

### Requisitos

El programa utiliza Python y estas bibliotecas:

- NumPy
- pandas
- SciPy
- scikit-learn
- Optuna

Instálalas mediante:

```bash
python -m pip install numpy pandas scipy scikit-learn optuna
```

### Estructura del proyecto

```text
proyecto-balon-de-oro/
├── data/
│   ├── dataset_ml_base.csv
│   └── ml/
└── nombre_del_script.py
```

`nombre_del_script.py` es un marcador de posición: sustitúyelo por el nombre real del archivo que contiene el código del modelo. El programa crea el directorio `data/ml/` cuando no existe.

### Ejecutar

Desde la raíz del proyecto:

```bash
python nombre_del_script.py
```

La ejecución incluye las dos búsquedas de Optuna, el backtest y el entrenamiento final. Debido a los 500 ensayos de cada búsqueda y a la generación de parejas, puede requerir bastante tiempo y memoria.

Para cambiar la temporada objetivo, modifica `TEMPORADA_PREDICCION` en la sección de configuración. Revisa también `BACKTEST_TEMPORADAS` y los límites de las búsquedas si necesitas adaptar el diseño experimental a otro periodo.

## Archivos generados

Los resultados se guardan en `data/ml/`.

| Archivo | Contenido |
|---|---|
| `prediccion_ensamble_backtest_2020_21.csv` | Ranking de backtest de 2020/21, si se puede evaluar. |
| `prediccion_ensamble_backtest_2021_22.csv` | Ranking de backtest de 2021/22. |
| `prediccion_ensamble_backtest_2022_23.csv` | Ranking de backtest de 2022/23. |
| `prediccion_ensamble_backtest_2023_24.csv` | Ranking de backtest de 2023/24. |
| `prediccion_ensamble_backtest_2024_25.csv` | Ranking de backtest de 2024/25. |
| `prediccion_ensamble_2025_26.csv` | Ranking final de la temporada objetivo. |
| `resumen_backtest_logistic_posicion_pairwise.csv` | Métricas de entrenamiento y validación por temporada, parámetros y referencias a los rankings. |
| `optuna_trials_evaluacion_hasta_2019_20.csv` | Historial de ensayos de Optuna de la Fase A. |
| `optuna_best_params_evaluacion_hasta_2019_20.csv` | Mejores parámetros de la Fase A. |
| `optuna_trials_prediccion_2025_26.csv` | Historial de ensayos de Optuna de la Fase B. |
| `optuna_best_params_prediccion_2025_26.csv` | Mejores parámetros de la Fase B. |
| `coeficientes_logistic_posicion_pairwise.csv` | Coeficientes, impacto medio absoluto e importancia por posición. |
| `importancia_estadisticas_por_posicion.csv` | El mismo resumen de importancia exportado con otro nombre. |
| `comparacion_mbappe_lamine_posicion_pairwise.csv` | Comparación de contribuciones entre Mbappé y Lamine Yamal, si ambos están disponibles. |
| `comparacion_cubarsi_lamine_posicion_pairwise.csv` | Comparación de contribuciones entre Cubarsí y Lamine Yamal, si ambos están disponibles. |

No todos los CSV de backtest se generan necesariamente: el código omite temporadas que no estén disponibles o que no tengan datos suficientes. Las comparaciones individuales tampoco se exportan si no se encuentran ambos jugadores.

## Limitaciones

1. **El modelo aprende resultados históricos.** No aprende directamente los votos individuales ni demuestra causalidad entre una estadística y el resultado del premio.
2. **La calidad depende del dataset.** Etiquetas incorrectas, estadísticas ausentes o definiciones diferentes entre fuentes pueden afectar al ranking.
3. **La identidad se agrupa por nombre y temporada.** La clave de agregación es `(temporada, nombre)`. Si existen homónimos o nombres inconsistentes, podrían agruparse filas de forma incorrecta.
4. **Las categorías de competición son reglas manuales.** Distintas competiciones pueden terminar agregadas en una misma categoría.
5. **Los resultados ordinales son definidos manualmente.** Los valores para rondas y resultados se establecen en el diccionario del código.
6. **Las restricciones monotónicas limitan el aprendizaje.** El modelo no puede invertir el signo fijado para las estadísticas clasificadas como negativas.
7. **La importancia no es causal.** Los coeficientes y la medida de impacto sirven para interpretar el modelo, pero no indican qué ocurriría si se modificara una estadística manteniendo todo lo demás constante en el mundo real.
8. **El backtest filtra posiciones reales conocidas.** Sus resultados no equivalen necesariamente a evaluar el ranking de todos los nominados.
9. **Las probabilidades no tienen calibración externa.** Se derivan de los scores y de la proyección Bradley–Terry, pero no hay una fase separada de calibración supervisada.
10. **El coste computacional puede ser alto.** El número de parejas crece aproximadamente como \(n(n-1)/2\) para \(n\) jugadores de una temporada.
11. **No todas las columnas numéricas son características.** Algunas columnas se convierten a formato numérico para ayudar a calcular variables derivadas, pero no se introducen directamente en el modelo.

## Resumen

Este proyecto combina agregación por categoría de competición, normalización contextual, aprendizaje por parejas y restricciones monotónicas para crear rankings del Balón de Oro. Dos regresiones logísticas *pairwise* —una con normalización por temporada y posición y otra con normalización por temporada— generan probabilidades de duelo que se combinan y se proyectan mediante Bradley–Terry para producir un ranking coherente.

Optuna selecciona los hiperparámetros con validación cronológica, el backtest evalúa las temporadas 2020/21–2024/25 y el entrenamiento final genera la predicción configurada para 2025/26.
