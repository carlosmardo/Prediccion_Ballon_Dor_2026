import os
import re
import warnings

import numpy as np
import pandas as pd
import optuna
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET = 'data/dataset_ml_base.csv'
OUTPUT_DIR = 'data/ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ITER_OPTIMIZACION = 10000000

TEMPORADA_PREDICCION = '2025/26'
# Evaluación independiente: estas temporadas NO se usan en la primera búsqueda de Optuna.
BACKTEST_TEMPORADAS = [
    '2020/21',
    '2021/22',
    '2022/23',
    '2023/24',
    '2024/25',
]

# Primera búsqueda: valida únicamente temporadas hasta 2019/20.
TEMPORADA_LIMITE_OPTUNA_EVALUACION = '2019/20'

# Segunda búsqueda: usa como validación todas las temporadas históricas disponibles
# anteriores a TEMPORADA_PREDICCION, mediante validación temporal walk-forward.

N_TRIALS_OPTUNA_EVALUACION = 500
N_TRIALS_OPTUNA_PREDICCION_FINAL = 500

BACKTEST_FILE = os.path.join(
    OUTPUT_DIR, 'resumen_backtest_logistic_solo_temporada.csv'
)
OPTUNA_EVALUACION_TRIALS_FILE = os.path.join(
    OUTPUT_DIR, 'optuna_trials_solo_temporada_evaluacion_hasta_2019_20.csv'
)
OPTUNA_EVALUACION_BEST_FILE = os.path.join(
    OUTPUT_DIR, 'optuna_best_params_solo_temporada_evaluacion_hasta_2019_20.csv'
)
OPTUNA_PREDICCION_TRIALS_FILE = os.path.join(
    OUTPUT_DIR, 'optuna_trials_solo_temporada_prediccion_2025_26.csv'
)
OPTUNA_PREDICCION_BEST_FILE = os.path.join(
    OUTPUT_DIR, 'optuna_best_params_solo_temporada_prediccion_2025_26.csv'
)
COEFICIENTES_FILE = os.path.join(
    OUTPUT_DIR, 'coeficientes_logistic_solo_temporada.csv'
)
PESO_ESTADISTICAS_FILE = os.path.join(
    OUTPUT_DIR, 'importancia_estadisticas_solo_temporada.csv'
)
COMPARACION_DUELO_FILE = os.path.join(
    OUTPUT_DIR, 'comparacion_mbappe_lamine_solo_temporada.csv'
)
COMPARACION_DUELO_CUBARSI_FILE = os.path.join(
    OUTPUT_DIR, 'comparacion_cubarsi_lamine_solo_temporada.csv'
)
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

MAPEO_PORCENTAJES = {
    'accuratePassesPercentage': ('accuratePasses', 'totalPasses'),
    'successfulDribblesPercentage': ('successfulDribbles', 'totalDribbles'),
    'aerialDuelsWonPercentage': ('aerialDuelsWon', 'totalAerialDuels'),
    'totalDuelsWonPercentage': ('totalDuelsWon', 'totalDuels'),
    'tacklesWonPercentage': ('tacklesWon', 'totalTackles'),
    'goalConversionPercentage': ('goals', 'totalShots'),
}

ESTADISTICAS_ADITIVAS_SOLICITADAS = [
    'goalsAssistsSum',
    'bigChancesCreated',
    'bigChancesMissed',
    'keyPasses',
    'successfulDribbles',
    'accuratePasses',
    'totalAttemptAssist',
    'aerialDuelsWon',
    'totalDuelsWon',
    'tacklesWon',
    'interceptions',
    'outfielderBlocks',
    'ballRecovery',
    'dispossessed',
    'dribbledPast',
    'clearances',
    'errorLeadToGoal',
    'errorLeadToShot',

    'saves',
    'goalsPrevented',
    'goalsConceded',
    'highClaims',
    'punches',
    'successfulRunsOut',

    'cleanSheet',
    'ownGoals',
    'penaltyConceded',
    'possessionWonAttThird',
]

ESTADISTICAS_PORCENTUALES = list(MAPEO_PORCENTAJES)

ESTADISTICAS_PERMITIDAS_PORTERO = {
    'saves',
    'goalsPrevented',
    'goalsConceded',
    'highClaims',
    'punches',
    'successfulRunsOut',
    'cleanSheet',
    'ownGoals',
    'penaltyConceded',
    'errorLeadToGoal',
    'errorLeadToShot',
    'accuratePassesPercentage',
}

ESTADISTICAS_EXCLUSIVAS_PORTERO = {
    'saves',
    'goalsPrevented',
    'highClaims',
    'punches',
    'successfulRunsOut',
}


# ============================================================
# 1. UTILIDADES
# ============================================================

def imprimir_separador(titulo):
    print('\n' + '=' * 100)
    print(titulo)
    print('=' * 100)


def limpiar_nombre_columna(columna):
    columna = re.sub(r'[^A-Za-z0-9_]+', '_', str(columna))
    return re.sub(r'_+', '_', columna).strip('_')


def convertir_temporada_a_orden(temporada):
    match = re.match(r'^\s*(\d{4})', str(temporada))
    return int(match.group(1)) if match else 10**9


def columna_estadistica_negativa(nombre_feature):
    nombre = re.sub(r'__zscore$', '', str(nombre_feature))
    return any(
        nombre == stat or nombre.endswith(f'_{stat}')
        for stat in ESTADISTICAS_NEGATIVAS
    )


def obtener_signo_restriccion(nombre_feature):
    nombre = re.sub(r'__zscore$', '', str(nombre_feature))
    if columna_estadistica_negativa(nombre):
        return -1
    return 1



def calcular_porcentajes_softmax(logits):
    logits_array = np.asarray(logits, dtype=np.float64)
    if len(logits_array) == 0:
        return logits_array
    logits_estables = logits_array - np.max(logits_array)
    exp_vals = np.exp(np.clip(logits_estables, -50.0, 50.0))
    suma_exp = np.sum(exp_vals)
    if suma_exp < 1e-12:
        return np.ones_like(logits_array) / len(logits_array) * 100.0
    return (exp_vals / suma_exp) * 100.0


def relevancia_ndcg(rank, n):
    if pd.isna(rank):
        return 0.0
    rel = (n - rank) / max(n - 1, 1) * 4.0
    return max(0.0, rel)


# ============================================================
# 2. CARGAR DATASET Y PREPARAR COLUMNAS
# ============================================================

imprimir_separador('CARGANDO DATASET')
if not os.path.exists(DATASET):
    raise FileNotFoundError(f'No se encontró el dataset en {DATASET!r}.')

df = pd.read_csv(DATASET).copy()
# Conserva solo filas con competición y posición presentes; no modifica sus etiquetas.
df = df.dropna(subset=['competicion', 'posicion']).copy()
df['temporada'] = df['temporada'].astype(str).str.strip()
df['nombre'] = df['nombre'].astype(str).str.strip()
df['club'] = df['club'].fillna('').astype(str).str.strip()
df['posicion'] = df['posicion'].astype(str).str.strip()
df['competicion'] = df['competicion'].astype(str).str.strip()
df['posicion_balon_oro'] = pd.to_numeric(df['posicion_balon_oro'], errors='coerce')

# En 2025/26, los valores <=0 significan que el puesto todavía se desconoce.
mascara_objetivo = df['temporada'].eq(TEMPORADA_PREDICCION)
mascara_objetivo_sin_posicion = (
    mascara_objetivo
    & df['posicion_balon_oro'].notna()
    & (df['posicion_balon_oro'] <= 0)
)
if mascara_objetivo_sin_posicion.any():
    df.loc[mascara_objetivo_sin_posicion, 'posicion_balon_oro'] = np.nan

# Las posiciones ya vienen normalizadas en el CSV; no se renombran ni se mapean.
POSICIONES = sorted(df['posicion'].dropna().astype(str).str.strip().unique().tolist())

# ============================================================
# 3. PREPARAR COLUMNAS NUMÉRICAS (SIN NORMALIZAR NOMBRES DE POSICIÓN)
# ============================================================

COLUMNAS_AUXILIARES_NUMERICAS = sorted({
    'goals', 'assists', 'goalsAssistsSum', 'minutesPlayed',
    'totalPasses', 'totalDribbles', 'totalAerialDuels',
    'totalDuels', 'totalTackles', 'totalShots', 'expectedGoals',
    'expectedAssists', *ESTADISTICAS_PORCENTUALES,
    *ESTADISTICAS_ADITIVAS_SOLICITADAS,
})

for col in COLUMNAS_AUXILIARES_NUMERICAS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for pct, (num, den) in MAPEO_PORCENTAJES.items():
    if num in df.columns and pct in df.columns:
        numerador = pd.to_numeric(df[num], errors='coerce')
        porcentaje = pd.to_numeric(df[pct], errors='coerce')
        pct_validos = porcentaje.dropna()
        if not pct_validos.empty and pct_validos.max() <= 1.0:
            porcentaje = porcentaje * 100.0
        denominador_inferido = numerador / (porcentaje / 100.0).where(porcentaje > 0)
        
        mascara_ceros = (numerador == 0) & (porcentaje == 0) & numerador.notna() & porcentaje.notna()
        denominador_inferido = denominador_inferido.mask(mascara_ceros, 0.0)

        if den not in df.columns:
            df[den] = denominador_inferido
        else:
            denominador_original = pd.to_numeric(df[den], errors='coerce')
            df[den] = denominador_original.fillna(denominador_inferido)

if 'goalsAssistsSum' not in df.columns and {'goals', 'assists'}.issubset(df.columns):
    df['goalsAssistsSum'] = df[['goals', 'assists']].sum(axis=1, min_count=1)

df = df.copy()
estadisticas_aditivas = [col for col in ESTADISTICAS_ADITIVAS_SOLICITADAS if col in df.columns]
estadisticas_porcentuales = list(MAPEO_PORCENTAJES)

# ============================================================
# 4. CATEGORIZACIÓN DE COMPETICIONES Y RESULTADOS
# ============================================================

CATEGORIAS_COMPETICION = {
    'Bundesliga': '5_grandes_ligas',
    'LaLiga': '5_grandes_ligas',
    'Ligue 1': '5_grandes_ligas',
    'Premier League': '5_grandes_ligas',
    'Serie A': '5_grandes_ligas',
    'Campeonato Brasileiro Série A': 'ligas_secundarias',
    'Eredivisie': 'ligas_secundarias',
    'Liga Portugal': 'ligas_secundarias',
    'Saudi Pro League': 'ligas_secundarias',
    'Copa del Rey': 'copa_nacional_principal_5grandes',
    'Coppa Italia': 'copa_nacional_principal_5grandes',
    'Coupe de France': 'copa_nacional_principal_5grandes',
    'DFB Pokal': 'copa_nacional_principal_5grandes',
    'FA Cup': 'copa_nacional_principal_5grandes',
    'Coupe de la Ligue': 'copa_de_la_liga_5grandes',
    'EFL Cup': 'copa_de_la_liga_5grandes',
    'Community Shield': 'supercopa_5grandes_ligas',
    'Franz Beckenbauer Supercup': 'supercopa_5grandes_ligas',
    'Supercopa de España': 'supercopa_5grandes_ligas',
    'Supercoppa Italiana': 'supercopa_5grandes_ligas',
    'Trophée des Champions': 'supercopa_5grandes_ligas',
    'KNVB Beker': 'copa_nacional_principal_ligas_secundarias',
    "King's Cup": 'copa_nacional_principal_ligas_secundarias',
    'Johan Cruijff Schaal': 'supercopa_ligas_secundarias',
    'Saudi Super Cup': 'supercopa_ligas_secundarias',
    'Supertaça': 'supercopa_ligas_secundarias',
    'UEFA Champions League': 'uefa_champions_league',
    'UEFA Europa League': 'uefa_europa_league',
    'UEFA Conference League': 'uefa_conference_league',
    'FIFA Club World Cup': 'fifa_club_world_cup',
    'FIFA Intercontinental Cup': 'fifa_intercontinental_cup',
    'CONCACAF Champions Cup': 'concacaf_champions_cup',
    'AFC Champions League Two': 'competiciones_anual_continental_clubes',
    'UEFA Super Cup': 'competiciones_anual_continental_clubes',
    'FIFA World Cup': 'fifa_world_cup',
    'World Cup Qual. AFC': 'clasificacion_world_cup',
    'World Cup Qual. CAF': 'clasificacion_world_cup',
    'World Cup Qual. CONMEBOL': 'clasificacion_world_cup',
    'World Cup Qual. UEFA': 'clasificacion_world_cup',
    'Copa América': 'torneos_selecciones_continentales_tier1',
    'EURO': 'torneos_selecciones_continentales_tier1',
    'AFC Asian Cup': 'torneos_selecciones_continentales_tier2',
    'Africa Cup of Nations': 'torneos_selecciones_continentales_tier2',
    'UEFA Nations League': 'torneos_selecciones_continentales_tier2',
    'Confederations Cup': 'confederations_cup',
    'Olympic Games': 'olympic_games',
    'Africa Cup of Nations Qual.': 'clasificacion_torneos_continentales',
    'EURO, Qualification': 'clasificacion_torneos_continentales',
}

MAPA_RESULTADO_ORDINAL = {
    'Ganador': 1.00,
    'Final': 0.9,
    'Partido por el tercer puesto': 0.80,
    'Semifinales': 0.80,
    'Cuartos de final': 0.7,
    'Octavos de final': 0.6,
    'Dieciseisavos de final': 0.5,
    'Ronda intermedia': 0.4,
    'Repechaje': 0.4,
    'Fase de Grupos': 0.3,
    '7ª Ronda': 0.2,
    '6ª Ronda': 0.2,
    '5ª Ronda': 0.1,
    '4ª Ronda': 0.1,
    '3ª Ronda': 0.05,
    '2ª Ronda': 0.05,
    '1': 1.00,
    '2': 0.9,
    '3': 0.8,
    '4': 0.7,
    '5': 0.6,
    '6': 0.5,
    '7': 0.4,
    '8': 0.3,
    '9': 0.2,
    '10': 0.1,
    '11': 0.075,
    '17': 0.05,
    '20': 0.025
}

df['categoria_competicion'] = df['competicion'].map(CATEGORIAS_COMPETICION)
mascara_categoria_desconocida = df['categoria_competicion'].isna()
df.loc[mascara_categoria_desconocida, 'categoria_competicion'] = (
    df.loc[mascara_categoria_desconocida, 'competicion'].apply(limpiar_nombre_columna)
)
categorias_unicas = sorted(df['categoria_competicion'].astype(str).unique())

resultado_texto = df['resultado_competicion'].astype(str).str.strip()
df['__resultado_ordinal'] = resultado_texto.map(MAPA_RESULTADO_ORDINAL).astype(float)
df = df.copy()

competiciones_jugador = (
    df.groupby(['temporada', 'nombre'])['categoria_competicion']
    .unique()
    .to_dict()
)


# ============================================================
# 5. AGREGACIÓN Y FILTRADO DE ESTADÍSTICAS PARA PORTEROS
# ============================================================

imprimir_separador('AGREGANDO ESTADÍSTICAS POR JUGADOR Y TEMPORADA')

CLAVE_JUGADOR = ['temporada', 'nombre']
COLUMNAS_IDENTIDAD = [c for c in ['id_sofascore', 'nacionalidad', 'posicion', 'posicion_balon_oro', 'dataset', 'club'] if c in df.columns]

df_identidad = df[CLAVE_JUGADOR + COLUMNAS_IDENTIDAD].groupby(CLAVE_JUGADOR, as_index=False, sort=True).first()
lista_agregados = [df_identidad.set_index(CLAVE_JUGADOR)]
features_base_creadas = []

for categoria in categorias_unicas:
    sub = df.loc[df['categoria_competicion'].eq(categoria)].copy()
    if sub.empty:
        continue

    cat_limpia = limpiar_nombre_columna(categoria)
    grupo = sub.groupby(CLAVE_JUGADOR, sort=True)
    agregado = pd.DataFrame(index=grupo.size().index)

    for stat in estadisticas_aditivas:
        if stat in sub.columns:
            nombre_feature = f'{cat_limpia}_{stat}'
            agregado[nombre_feature] = grupo[stat].sum(min_count=1)
            features_base_creadas.append(nombre_feature)

    for pct, (num, den) in MAPEO_PORCENTAJES.items():
        if num in sub.columns and den in sub.columns:
            columna_pct = f'{cat_limpia}_{pct}'
            num_sum = grupo[num].sum(min_count=1)
            den_sum = grupo[den].sum(min_count=1)
            porcentaje_agregado = np.where(den_sum > 0, 100.0 * num_sum / den_sum, np.where(den_sum == 0, 0.0, np.nan))
            agregado[columna_pct] = pd.Series(porcentaje_agregado, index=num_sum.index)
            features_base_creadas.append(columna_pct)

    if 'goalsAssistsSum' in sub.columns and 'minutesPlayed' in sub.columns:
        suma_goles_asistencias = grupo['goalsAssistsSum'].sum(min_count=1)
        suma_minutos = grupo['minutesPlayed'].sum(min_count=1)
        nombre_feature = f'{cat_limpia}_goalsAssistsPerMins'
        agregado[nombre_feature] = suma_goles_asistencias / suma_minutos.replace(0, np.nan)
        features_base_creadas.append(nombre_feature)

    col_resultado = f'{cat_limpia}_resultado_competicion'
    agregado[col_resultado] = grupo['__resultado_ordinal'].max()
    features_base_creadas.append(col_resultado)

    lista_agregados.append(agregado)

df_agrupado = pd.concat(lista_agregados, axis=1).reset_index()
features_base_creadas = list(dict.fromkeys(features_base_creadas))

def es_stat_permitida_portero(stat_base):
    if stat_base == 'resultado_competicion' or stat_base.endswith('_resultado_competicion'):
        return True
    return stat_base in ESTADISTICAS_PERMITIDAS_PORTERO

for col in features_base_creadas:
    stat_base = None
    for s in list(estadisticas_aditivas) + list(MAPEO_PORCENTAJES.keys()) + ['goalsAssistsPerMins', 'resultado_competicion']:
        if col.endswith(f'_{s}') or col == s:
            stat_base = s
            break
    
    if stat_base and not es_stat_permitida_portero(stat_base):
        mascara_gk = df_agrupado['posicion'].eq('Goalkeeper')
        df_agrupado.loc[mascara_gk, col] = np.nan


# ============================================================
# 6. GENERAR Z-SCORES SOLO POR TEMPORADA
# ============================================================

imprimir_separador('GENERANDO Z-SCORES SOLO POR TEMPORADA CON TRATAMIENTO CONDICIONAL DE NaN')

categorias_feature = sorted(
    {limpiar_nombre_columna(c) for c in categorias_unicas},
    key=len,
    reverse=True,
)

ESTADISTICAS_BASE_FEATURE = sorted(
    set(estadisticas_aditivas)
    | set(MAPEO_PORCENTAJES.keys())
    | {'goalsAssistsPerMins', 'resultado_competicion'},
    key=len,
    reverse=True,
)


def obtener_stat_base_feature(feature):
    """Obtiene el nombre de la estadística al final del nombre de una feature."""
    for stat in ESTADISTICAS_BASE_FEATURE:
        if feature.endswith(f'_{stat}') or feature == stat:
            return stat
    return None


def generar_features_normalizadas(df_base):
    """Genera z-scores usando la media y desviación de cada temporada y feature.

    Reglas para calcular media/desviación:
    - Si el jugador participó en la competición, sus NaN se excluyen del cálculo.
    - Las NaN de estadísticas exclusivas de portero cuentan como cero en el cálculo,
      independientemente de si el jugador participó en la competición.
    - Si el jugador no participó en la competición, sus NaN cuentan como cero
      para calcular media/desviación.

    Después de calcular media y desviación, todo valor originalmente NaN recibe
    z-score 0, es decir, se trata como si estuviera en la media.
    """
    features_transformadas = []
    partes_procesadas = []

    for _, grupo_original in df_base.groupby(['temporada'], sort=False):
        grupo = grupo_original.copy()
        temporada = str(grupo['temporada'].iloc[0]).strip()
        nuevas = {}

        for feature in features_base_creadas:
            valores = pd.to_numeric(grupo[feature], errors='coerce').replace(
                [np.inf, -np.inf], np.nan
            )
            mascara_nan = valores.isna()

            competicion_de_feature = next(
                (cat for cat in categorias_feature if feature.startswith(f'{cat}_')),
                None,
            )
            if competicion_de_feature is None:
                raise ValueError(
                    f'No se pudo identificar la categoría de la característica: {feature}'
                )

            # Determina si cada jugador jugó la competición asociada a esta feature.
            jugo_competicion = np.array([
                competicion_de_feature in [
                    limpiar_nombre_columna(c)
                    for c in competiciones_jugador.get((temporada, nombre), [])
                ]
                for nombre in grupo['nombre']
            ], dtype=bool)

            stat_base = obtener_stat_base_feature(feature)
            es_estadistica_exclusiva_portero = stat_base in ESTADISTICAS_EXCLUSIVAS_PORTERO

            valores_para_calculo = valores.copy()
            if es_estadistica_exclusiva_portero:
                # En estas cinco estadísticas, todo NaN cuenta como un cero para
                # calcular los parámetros de normalización.
                valores_para_calculo = valores_para_calculo.fillna(0.0)
            else:
                # Los NaN de quien NO jugó la competición cuentan como cero para
                # media/desviación. Los NaN de quien sí la jugó quedan excluidos.
                mascara_nan_no_jugo = mascara_nan.to_numpy() & (~jugo_competicion)
                valores_para_calculo.loc[mascara_nan_no_jugo] = 0.0

            valores_calculo_validos = valores_para_calculo.dropna()
            zscore = pd.Series(0.0, index=grupo.index, dtype=np.float64)

            if not valores_calculo_validos.empty:
                media = float(valores_calculo_validos.mean())
                desviacion = float(valores_calculo_validos.std(ddof=0))

                if np.isfinite(desviacion) and desviacion > 1e-12:
                    mascara_valores_observados = valores.notna()
                    zscore.loc[mascara_valores_observados] = (
                        (valores.loc[mascara_valores_observados] - media) / desviacion
                    ).to_numpy(dtype=np.float64)

            # Todos los NaN originales quedan en z-score 0, aunque algunos hayan
            # contado como cero al calcular media y desviación.
            zscore.loc[mascara_nan] = 0.0
            nombre_zscore = f'{feature}__zscore'
            nuevas[nombre_zscore] = zscore.to_numpy(dtype=np.float32)

            if nombre_zscore not in features_transformadas:
                features_transformadas.append(nombre_zscore)

        grupo = pd.concat([grupo, pd.DataFrame(nuevas, index=grupo.index)], axis=1)
        partes_procesadas.append(grupo)

    # Mantiene el orden original para que etiquetas, features y candidatos sigan alineados.
    df_resultado = pd.concat(partes_procesadas, axis=0).sort_index().reset_index(drop=True)
    return df_resultado, list(dict.fromkeys(features_transformadas))


df_features, FEATURE_COLS = generar_features_normalizadas(df_agrupado)

# ============================================================
# 7. MATRIZ DE CARACTERÍSTICAS CONDICIONADA POR POSICIÓN
# ============================================================

# Cada característica se replica por posición. La fila de un jugador activa
# únicamente el bloque de su posición, pero los pares se generan entre TODOS
# los jugadores de una temporada. Así los scores de posiciones diferentes son
# comparables y el modelo aprende, por ejemplo, pesos distintos para los goles
# de un delantero y los de un centrocampista.
N_FEATURES_BASE = len(FEATURE_COLS)
N_POSICIONES = len(POSICIONES)
FEATURE_COLS_CONDICIONADAS = [
    f'{pos}::{feature}'
    for pos in POSICIONES
    for feature in FEATURE_COLS
]


def obtener_signo_restriccion_condicionada(nombre_feature):
    """Recupera el signo según la estadística, ignorando el prefijo de posición."""
    if '::' in nombre_feature:
        nombre_feature = nombre_feature.split('::', 1)[1]
    return obtener_signo_restriccion(nombre_feature)


def crear_matriz_jugadores_condicionada(df_input, feature_cols=FEATURE_COLS):
    """Devuelve una matriz CSR con un bloque de estadísticas por posición."""
    datos = df_input.reset_index(drop=True)
    n = len(datos)
    f = len(feature_cols)
    if n == 0:
        return sparse.csr_matrix((0, len(POSICIONES) * f), dtype=np.float32)

    # Los NaN ya se imputan como z-score 0 (la media) durante la normalización.
    # Por tanto, la matriz usa directamente los z-scores, sin máscaras adicionales.
    features = datos[feature_cols].to_numpy(dtype=np.float32, copy=True)
    efectivos = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    posiciones_jugador = datos['posicion'].astype(str).to_numpy()
    matriz = np.zeros((n, len(POSICIONES) * f), dtype=np.float32)
    for idx_pos, posicion in enumerate(POSICIONES):
        filas = np.flatnonzero(posiciones_jugador == posicion)
        if len(filas):
            inicio = idx_pos * f
            matriz[filas, inicio:inicio + f] = efectivos[filas]

    return sparse.csr_matrix(matriz)


# ============================================================
# 8. GENERACIÓN DE PARES GLOBALES CON PONDERACIÓN DELTA-NDCG
# ============================================================


def generar_pares_global(df_temporadas, feature_cols=FEATURE_COLS):
    """Genera duelos entre todos los candidatos de cada temporada.

    La ponderación es una aproximación estática inspirada en Delta-NDCG: prioriza
    pares cuyo intercambio de orden real tendría mayor impacto en NDCG. No es una
    implementación dinámica de LambdaRank, porque los pesos no se recalculan por
    iteración a partir del ranking provisional del modelo.
    """
    matrices_diff = []
    arrays_target = []
    arrays_weights = []

    for temporada, grupo_temp in df_temporadas.groupby('temporada', sort=False):
        grupo = grupo_temp.loc[grupo_temp['posicion_balon_oro'].notna()].reset_index(drop=True)
        n = len(grupo)
        if n < 2:
            continue

        ranks = pd.to_numeric(grupo['posicion_balon_oro'], errors='coerce').to_numpy(dtype=float)
        filas_validas = np.isfinite(ranks)
        if filas_validas.sum() < 2:
            continue
        if not filas_validas.all():
            grupo = grupo.loc[filas_validas].reset_index(drop=True)
            ranks = ranks[filas_validas]
            n = len(grupo)
        if n < 2:
            continue

        X = crear_matriz_jugadores_condicionada(grupo, feature_cols)

        i, j = np.triu_indices(n, k=1)
        validos = ranks[i] != ranks[j]
        i, j = i[validos], j[validos]
        if len(i) == 0:
            continue

        diferencias = X[i] - X[j]
        targets = (ranks[i] < ranks[j]).astype(np.float32)

        # Rango relativo dentro de la edición para soportar posibles huecos en
        # las posiciones oficiales sin confundirlos con el número de candidatos.
        orden = np.argsort(ranks, kind='stable')
        ranks_relativos = np.empty(n, dtype=np.float64)
        ranks_relativos[orden] = np.arange(1, n + 1, dtype=np.float64)
        relevancias = np.array([relevancia_ndcg(r, n) for r in ranks_relativos], dtype=np.float64)
        ganancias = np.power(2.0, relevancias) - 1.0
        descuentos = 1.0 / np.log2(ranks_relativos + 1.0)
        posiciones_ideales = np.arange(1, n + 1, dtype=np.float64)
        idcg = np.sum((np.sort(ganancias)[::-1]) / np.log2(posiciones_ideales + 1.0))

        delta_ndcg = np.abs(
            (ganancias[i] - ganancias[j]) * (descuentos[i] - descuentos[j])
        )
        if idcg > 1e-12:
            # Raíz cuadrada para suavizar la diferencia relativa entre pares:
            # los pesos pequeños ganan importancia relativa frente a los mayores.
            weights = np.cbrt(np.clip(delta_ndcg / idcg, 0.0, None))
        else:
            weights = np.ones_like(delta_ndcg)

        # Cada temporada aporta un peso total equivalente, evitando que una
        # edición con más pares domine automáticamente todo el entrenamiento.
        suma_pesos = float(np.sum(weights))
        if suma_pesos > 1e-12:
            weights = weights / suma_pesos
        else:
            weights = np.full_like(weights, 1.0 / len(weights))

        matrices_diff.append(diferencias.tocsr())
        arrays_target.append(targets)
        arrays_weights.append(weights.astype(np.float64))

    n_features_cond = len(POSICIONES) * len(feature_cols)
    if not matrices_diff:
        return (
            sparse.csr_matrix((0, n_features_cond), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float64),
        )

    return (
        sparse.vstack(matrices_diff, format='csr'),
        np.concatenate(arrays_target),
        np.concatenate(arrays_weights),
    )


# ============================================================
# 9. REGRESIÓN LOGÍSTICA PAIRWISE CON RESTRICCIONES MONOTÓNICAS
# ============================================================

class LogisticPairwisePosicionMonotonic:
    def __init__(self, C=1.0, max_iter=MAX_ITER_OPTIMIZACION, ftol=1e-9, gtol=1e-6, epsilon=0.0):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.epsilon = float(epsilon)
        if self.epsilon < 0:
            raise ValueError('epsilon no puede ser negativo.')

    def fit(self, X_diff, y, feature_names, sample_weight=None):
        self.feature_names_ = list(feature_names)
        X = sparse.csr_matrix(X_diff, dtype=np.float64)
        y_all = np.asarray(y, dtype=np.float64).ravel()
        if len(y_all) != X.shape[0]:
            raise ValueError('X_diff e y tienen distinto número de filas.')
        if X.shape[0] == 0:
            raise ValueError('No hay pares para entrenar el modelo.')

        w_all = (
            np.asarray(sample_weight, dtype=np.float64).ravel()
            if sample_weight is not None else np.ones_like(y_all, dtype=np.float64)
        )
        if len(w_all) != len(y_all):
            raise ValueError('sample_weight e y tienen distinto tamaño.')
        w_all = np.nan_to_num(w_all, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(w_all < 0):
            raise ValueError('Los pesos de los pares no pueden ser negativos.')
        suma_pesos = max(float(np.sum(w_all)), 1e-12)

        # Monotonicidad dura: coeficientes positivos >= epsilon y negativos <= -epsilon.
        # Con epsilon=0 se permite que la regularización reduzca un coeficiente a cero.
        signos = [obtener_signo_restriccion_condicionada(f) for f in self.feature_names_]
        bounds = [
            (self.epsilon, None) if s > 0 else (None, -self.epsilon)
            for s in signos
        ]
        X.eliminate_zeros()

        def loss_and_gradient(coefs):
            logits = np.asarray(X @ coefs).ravel()
            bce = np.logaddexp(0.0, logits) - y_all * logits
            loss_bce = float(np.sum(w_all * bce) / suma_pesos)
            loss_l2 = 0.5 * float(np.dot(coefs, coefs)) / self.C
            grad_bce = np.asarray(X.T @ (w_all * (expit(np.clip(logits, -50.0, 50.0)) - y_all))).ravel() / suma_pesos
            grad = grad_bce + coefs / self.C
            return loss_bce + loss_l2, grad

        x0 = np.array(
            [self.epsilon if s > 0 else -self.epsilon for s in signos],
            dtype=np.float64,
        )
        resultado = minimize(
            loss_and_gradient,
            x0=x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': self.max_iter,
                'ftol': self.ftol,
                'gtol': self.gtol,
                'maxls': 100,
            },
        )
        self.optimization_result_ = resultado
        if not resultado.success:
            raise RuntimeError(f'La optimización no convergió: {resultado.message}')

        self.coef_ = resultado.x
        return self


def calcular_logits_jugadores(df_input, modelo, feature_cols=FEATURE_COLS):
    X = crear_matriz_jugadores_condicionada(df_input, feature_cols)
    logits = np.asarray(X @ modelo.coef_).ravel()
    return logits, X


def posiciones_predichas(scores):
    """Devuelve posiciones 1..N, ordenadas de mayor a menor score."""
    return pd.Series(np.asarray(scores, dtype=float)).rank(
        ascending=False, method='first'
    ).to_numpy(dtype=int)



def probabilidades_duelo_ranking(scores_ranking_ordenados):
    """Calcula probabilidades de duelo con los scores logísticos nativos.

    Los scores se ordenan de mayor a menor. Para cada jugador posterior al líder,
    calcula sigmoid(score_actual - score_líder) y sigmoid(score_actual - score_anterior).
    No aplica una proyección Bradley–Terry adicional ni una calibración externa.
    """
    scores = np.asarray(scores_ranking_ordenados, dtype=np.float64).ravel()
    if not np.isfinite(scores).all():
        raise ValueError('Hay scores no finitos al calcular probabilidades del ranking.')

    prob_lider, prob_anterior = [], []
    for idx in range(len(scores)):
        if idx == 0:
            prob_lider.append(np.nan)
            prob_anterior.append(np.nan)
            continue
        p_lider = float(expit(np.clip(scores[idx] - scores[0], -50.0, 50.0)))
        p_anterior = float(expit(np.clip(scores[idx] - scores[idx - 1], -50.0, 50.0)))
        prob_lider.append(f'{p_lider * 100:.4f}%')
        prob_anterior.append(f'{p_anterior * 100:.4f}%')
    return prob_lider, prob_anterior


def evaluar_ranking(scores, datos_validacion):
    datos = datos_validacion.reset_index(drop=True).copy()
    scores = np.asarray(scores, dtype=float).ravel()
    if len(scores) != len(datos):
        raise ValueError('Los scores y las filas de validación no coinciden.')
    datos['score_modelo'] = scores
    datos = datos.loc[datos['posicion_balon_oro'].notna()].copy()
    datos['posicion_real'] = pd.to_numeric(datos['posicion_balon_oro'], errors='coerce')
    datos = datos.dropna(subset=['posicion_real']).reset_index(drop=True)
    n = len(datos)
    if n == 0:
        return {'ndcg@30': 0.0, 'spearman': 0.0, 'mae': np.nan, 'rmse': np.nan, 'numero_jugadores': 0}

    posiciones_reales = datos['posicion_real'].to_numpy(dtype=float)
    scores_limpios = datos['score_modelo'].to_numpy(dtype=float)
    orden_real = np.argsort(posiciones_reales, kind='stable')
    rangos_relativos = np.empty(n, dtype=np.float64)
    rangos_relativos[orden_real] = np.arange(1, n + 1, dtype=np.float64)
    relevancias = np.array([relevancia_ndcg(r, n) for r in rangos_relativos], dtype=float)

    ndcg = ndcg_score([relevancias], [scores_limpios], k=min(30, n)) if n > 1 else 0.0
    rho = spearmanr(-posiciones_reales, scores_limpios).statistic if n > 1 else 0.0
    if not np.isfinite(rho):
        rho = 0.0

    posiciones_predichas = pd.Series(scores_limpios).rank(ascending=False, method='first').to_numpy(dtype=float)
    return {
        'ndcg@30': float(ndcg),
        'spearman': float(rho),
        'mae': float(mean_absolute_error(posiciones_reales, posiciones_predichas)),
        'rmse': float(np.sqrt(mean_squared_error(posiciones_reales, posiciones_predichas))),
        'numero_jugadores': int(n),
    }



def evaluar_resumen_por_temporada(scores, datos):
    """Promedia métricas de ranking calculadas por edición, nunca entre años."""
    scores = np.asarray(scores, dtype=float).ravel()
    datos = datos.reset_index(drop=True).copy()
    if len(scores) != len(datos):
        raise ValueError('Los scores y las filas para evaluar no coinciden.')
    resultados = []
    for _, indices in datos.groupby('temporada', sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        sub = datos.iloc[idx]
        if sub['posicion_balon_oro'].notna().sum() < 2:
            continue
        resultados.append(evaluar_ranking(scores[idx], sub))
    if not resultados:
        return {'ndcg@30': 0.0, 'spearman': 0.0, 'mae': np.nan, 'rmse': np.nan, 'numero_jugadores': 0}
    tabla = pd.DataFrame(resultados)
    return {
        'ndcg@30': float(tabla['ndcg@30'].mean()),
        'spearman': float(tabla['spearman'].mean()),
        'mae': float(tabla['mae'].mean()),
        'rmse': float(tabla['rmse'].mean()),
        'numero_jugadores': int(tabla['numero_jugadores'].sum()),
    }


# ============================================================
# 10. OPTIMIZACIÓN EN DOS FASES: EVALUACIÓN INDEPENDIENTE Y 2025/26
# ============================================================

imprimir_separador('PREPARANDO OPTUNA Y EL BACKTEST CRONOLÓGICO')

ORDEN_TEMPORADAS_FEATURES = df_features['temporada'].astype(str).map(convertir_temporada_a_orden)


def entrenar_modelo(train_rows, C, ftol, gtol, epsilon):
    """Entrena el modelo pairwise con los parámetros devueltos por Optuna."""
    X_diff, y_pairs, w_pairs = generar_pares_global(train_rows, FEATURE_COLS)
    if X_diff.shape[0] == 0:
        raise ValueError('No se pudieron generar pares de entrenamiento.')
    modelo = LogisticPairwisePosicionMonotonic(
        C=C, ftol=ftol, gtol=gtol, epsilon=epsilon
    )
    modelo.fit(X_diff, y_pairs, FEATURE_COLS_CONDICIONADAS, sample_weight=w_pairs)
    return modelo


def obtener_temporadas_optuna_elegibles(candidatas):
    """Conserva folds cronológicos con entrenamiento previo y al menos dos rangos reales distintos."""
    elegibles = []
    orden_temporadas = ORDEN_TEMPORADAS_FEATURES
    etiquetas = pd.to_numeric(df_features['posicion_balon_oro'], errors='coerce')

    for temporada_validacion in sorted(set(candidatas), key=convertir_temporada_a_orden):
        corte = convertir_temporada_a_orden(temporada_validacion)
        mascara_train = (orden_temporadas < corte) & etiquetas.notna()
        mascara_valid = df_features['temporada'].astype(str).eq(temporada_validacion) & etiquetas.notna()
        train = df_features.loc[mascara_train]
        valid = df_features.loc[mascara_valid]

        train_tiene_pares = (
            not train.empty
            and train.groupby('temporada')['posicion_balon_oro'].nunique().ge(2).any()
        )
        valid_tiene_pares = (
            len(valid) >= 2
            and pd.to_numeric(valid['posicion_balon_oro'], errors='coerce').nunique() >= 2
        )
        if train_tiene_pares and valid_tiene_pares:
            elegibles.append(temporada_validacion)
    return elegibles


def crear_objetivo_optuna(temporadas_validacion, nombre_fase):
    """Devuelve una función objetivo con validación temporal walk-forward."""
    temporadas_validacion = list(temporadas_validacion)

    def objetivo_optuna(trial):
        c_val = trial.suggest_float('C', 0.1, 100.0, log=True)
        ftol_val = trial.suggest_categorical(
            'ftol', [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
                     1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        )
        gtol_val = trial.suggest_categorical(
            'gtol', [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
                     1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        )
        epsilon_val = trial.suggest_categorical(
            'epsilon', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        )

        metricas_ranking = []
        folds_evaluados = 0
        for temporada_validacion in temporadas_validacion:
            corte = convertir_temporada_a_orden(temporada_validacion)
            orden_temporadas = ORDEN_TEMPORADAS_FEATURES
            train = df_features[
                (orden_temporadas < corte)
                & df_features['posicion_balon_oro'].notna()
            ].copy()
            valid = df_features[
                df_features['temporada'].astype(str).eq(temporada_validacion)
                & df_features['posicion_balon_oro'].notna()
            ].copy()

            if train.empty or len(valid) < 2:
                continue
            if train.groupby('temporada')['posicion_balon_oro'].nunique().ge(2).sum() == 0:
                continue
            if valid['posicion_balon_oro'].nunique() < 2:
                continue

            try:
                modelo = entrenar_modelo(train, c_val, ftol_val, gtol_val, epsilon_val)
                scores_modelo, _ = calcular_logits_jugadores(valid, modelo)
                scores_ranking = scores_modelo
                metricas_ranking.append(
                    evaluar_ranking(scores_ranking, valid)['ndcg@30']
                )
                folds_evaluados += 1
            except (RuntimeError, ValueError) as exc:
                print(
                    f'⚠ Optuna {nombre_fase}, trial {trial.number}, '
                    f'temporada {temporada_validacion}: {exc}'
                )
                raise optuna.exceptions.TrialPruned()

        if not metricas_ranking:
            raise optuna.exceptions.TrialPruned(
                'No hubo folds temporales válidos en este ensayo.'
            )
        trial.set_user_attr('folds_evaluados', folds_evaluados)
        return float(np.mean(metricas_ranking))

    return objetivo_optuna


def ejecutar_optuna_fase(temporadas_validacion, n_trials, nombre_estudio, trials_path, best_path):
    """Ejecuta una búsqueda, guarda su historial y devuelve sus hiperparámetros."""
    if not temporadas_validacion:
        raise ValueError(f'No hay temporadas elegibles para la búsqueda de Optuna: {nombre_estudio}.')

    print(f'\nFase Optuna: {nombre_estudio}')
    print(f'Temporadas de validación cronológica: {temporadas_validacion}')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        study_name=nombre_estudio,
    )
    study.optimize(
        crear_objetivo_optuna(temporadas_validacion, nombre_estudio),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError(f'Optuna no completó ningún ensayo válido en {nombre_estudio}.')

    study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'duration', 'user_attrs')).to_csv(
        trials_path, index=False, encoding='utf-8-sig'
    )
    best_params = {
        'C': float(study.best_params['C']),
        'ftol': float(study.best_params['ftol']),
        'gtol': float(study.best_params['gtol']),
        'epsilon': float(study.best_params['epsilon']),
    }
    pd.DataFrame([{
        'fase': nombre_estudio,
        'ndcg30_medio_optuna': float(study.best_value),
        'temporadas_validacion': ';'.join(temporadas_validacion),
        'n_trials_solicitados': int(n_trials),
        **best_params,
    }]).to_csv(best_path, index=False, encoding='utf-8-sig')

    print(f'✓ Mejor C: {best_params["C"]:.6g}')
    print(f'✓ Mejor ftol: {best_params["ftol"]:.2e}')
    print(f'✓ Mejor gtol: {best_params["gtol"]:.2e}')
    print(f'✓ Mejor epsilon: {best_params["epsilon"]:.1e}')
    print(f'✓ Mejor NDCG@30 medio: {study.best_value:.4f}')
    print(f'✓ Ensayos guardados en: {trials_path}')
    print(f'✓ Mejores parámetros guardados en: {best_path}')
    return study, best_params


# Cohortes independientes:
# - Fase A selecciona parámetros solo con temporadas anteriores/al límite de 2019/20.
# - Se evalúan 2020/21-2024/25 usando exclusivamente los parámetros de la Fase A.
# - Fase B busca parámetros para 2025/26 con todas las temporadas históricas elegibles
#   anteriores a 2025/26 como folds de validación cronológica.
temporadas_disponibles = set(df_features['temporada'].astype(str).unique())
backtest_efectivo = [t for t in BACKTEST_TEMPORADAS if t in temporadas_disponibles]
orden_limite_evaluacion = convertir_temporada_a_orden(TEMPORADA_LIMITE_OPTUNA_EVALUACION)
temporadas_candidatas_evaluacion = [
    t for t in temporadas_disponibles
    if convertir_temporada_a_orden(t) <= orden_limite_evaluacion
]
temporadas_optuna_evaluacion = obtener_temporadas_optuna_elegibles(temporadas_candidatas_evaluacion)
orden_temporada_prediccion_optuna = convertir_temporada_a_orden(TEMPORADA_PREDICCION)
temporadas_candidatas_optuna_final = [
    t for t in temporadas_disponibles
    if convertir_temporada_a_orden(t) < orden_temporada_prediccion_optuna
]
temporadas_optuna_final = obtener_temporadas_optuna_elegibles(temporadas_candidatas_optuna_final)

if not backtest_efectivo:
    raise ValueError('Ninguna temporada de BACKTEST_TEMPORADAS está presente en el dataset.')
if not temporadas_optuna_evaluacion:
    raise ValueError(
        'No hay temporadas elegibles para la primera búsqueda de Optuna hasta 2019/20. '
        'Comprueba las temporadas etiquetadas y que existan temporadas anteriores para entrenar.'
    )
if not temporadas_optuna_final:
    raise ValueError(
        f'No hay temporadas históricas elegibles para la búsqueda final de Optuna anteriores a {TEMPORADA_PREDICCION}.'
    )

print(f'✓ Folds de Optuna para la evaluación independiente: {temporadas_optuna_evaluacion}')
print(f'✓ Folds de Optuna para la predicción 2025/26: {temporadas_optuna_final}')
print(f'✓ Temporadas reservadas como validación independiente: {backtest_efectivo}')

# FASE A: parámetros seleccionados sin usar los resultados 2020/21-2024/25.
estudio_evaluacion, PARAMS_OPTUNA_EVALUACION = ejecutar_optuna_fase(
    temporadas_optuna_evaluacion,
    N_TRIALS_OPTUNA_EVALUACION,
    'pairwise_solo_temporada_evaluacion_hasta_2019_20',
    OPTUNA_EVALUACION_TRIALS_FILE,
    OPTUNA_EVALUACION_BEST_FILE,
)

# FASE B: segunda búsqueda para 2025/26, utilizando todos los folds históricos elegibles.
estudio_prediccion, PARAMS_OPTUNA_PREDICCION_FINAL = ejecutar_optuna_fase(
    temporadas_optuna_final,
    N_TRIALS_OPTUNA_PREDICCION_FINAL,
    'pairwise_solo_temporada_prediccion_2025_26',
    OPTUNA_PREDICCION_TRIALS_FILE,
    OPTUNA_PREDICCION_BEST_FILE,
)

print('\nParámetros reservados para evaluar el backtest independiente:')
print(PARAMS_OPTUNA_EVALUACION)
print('Parámetros elegidos específicamente para la predicción 2025/26:')
print(PARAMS_OPTUNA_PREDICCION_FINAL)
# ============================================================
# 11. VALIDACIÓN INDEPENDIENTE 2020/21-2024/25
# ============================================================

imprimir_separador('VALIDACIÓN INDEPENDIENTE: 2020/21-2024/25')
print('Estos rankings usan parámetros seleccionados únicamente con validaciones hasta 2019/20.')
resumen_backtest = []
metricas_validacion = []

for temporada_validacion in backtest_efectivo:
    corte = convertir_temporada_a_orden(temporada_validacion)
    orden_temporadas = ORDEN_TEMPORADAS_FEATURES
    train = df_features[
        (orden_temporadas < corte)
        & df_features['posicion_balon_oro'].notna()
    ].copy()
    valid = df_features[
        df_features['temporada'].astype(str).eq(temporada_validacion)
        & df_features['posicion_balon_oro'].notna()
    ].copy()

    if train.empty or len(valid) < 2:
        print(f'⚠ Se omite {temporada_validacion}: train o validación vacíos/insuficientes.')
        continue

    modelo = entrenar_modelo(train, **PARAMS_OPTUNA_EVALUACION)

    scores_train_modelo, _ = calcular_logits_jugadores(train, modelo)
    scores_train_ranking = scores_train_modelo
    scores_valid_modelo, _ = calcular_logits_jugadores(valid, modelo)
    scores_valid_ranking = scores_valid_modelo

    metricas_train = evaluar_resumen_por_temporada(scores_train_ranking, train)
    metricas_valid = evaluar_ranking(scores_valid_ranking, valid)
    metricas_valid_modelo = evaluar_ranking(scores_valid_modelo, valid)
    metricas_validacion.append(metricas_valid)

    ranking = valid[
        ['temporada', 'nombre', 'club', 'posicion', 'posicion_balon_oro']
    ].copy().reset_index(drop=True)
    ranking = ranking.rename(columns={'posicion_balon_oro': 'posicion_real'})
    ranking['score_modelo'] = scores_valid_modelo
    ranking['score_ranking'] = scores_valid_ranking
    ranking = ranking.sort_values('score_ranking', ascending=False).reset_index(drop=True)
    ranking['posicion_predicha'] = np.arange(1, len(ranking) + 1)

    scores_ordenados = ranking['score_ranking'].to_numpy(dtype=float)
    prob_lider, prob_anterior = probabilidades_duelo_ranking(scores_ordenados)
    ranking['prob_duelo_contra_lider'] = prob_lider
    ranking['prob_duelo_contra_anterior'] = prob_anterior

    columnas_ranking = [
        'temporada', 'posicion_predicha', 'posicion_real', 'nombre', 'club', 'posicion',
        'score_modelo', 'score_ranking',
        'prob_duelo_contra_lider', 'prob_duelo_contra_anterior',
    ]
    ranking = ranking[columnas_ranking]
    temp_clean = temporada_validacion.replace('/', '_')
    archivo_ranking = os.path.join(
        OUTPUT_DIR, f'prediccion_solo_temporada_backtest_{temp_clean}.csv'
    )
    ranking.to_csv(archivo_ranking, index=False, encoding='utf-8-sig')

    gap = metricas_train['ndcg@30'] - metricas_valid['ndcg@30']
    resumen_backtest.append({
        'temporada': temporada_validacion,
        'rol_temporada': 'validacion_independiente',
        **{f'train_{k}': v for k, v in metricas_train.items()},
        **{f'valid_ranking_{k}': v for k, v in metricas_valid.items()},
        'valid_modelo_ndcg@30': metricas_valid_modelo['ndcg@30'],
        'gap_ndcg30': gap,
        'C': PARAMS_OPTUNA_EVALUACION['C'],
        'ftol': PARAMS_OPTUNA_EVALUACION['ftol'],
        'gtol': PARAMS_OPTUNA_EVALUACION['gtol'],
        'epsilon': PARAMS_OPTUNA_EVALUACION['epsilon'],
        'archivo_ranking': archivo_ranking,
    })

    print(f'\nTemporada de validación independiente: {temporada_validacion}')
    print(f"  NDCG@30 modelo:  {metricas_valid_modelo['ndcg@30']:.4f}")
    print(f"  NDCG@30 ranking: {metricas_valid['ndcg@30']:.4f} | Spearman: {metricas_valid['spearman']:.4f}")
    print(f'  Ranking CSV: {archivo_ranking}')
    print(ranking.head(15).to_string(index=False))

pd.DataFrame(resumen_backtest).to_csv(BACKTEST_FILE, index=False, encoding='utf-8-sig')
if metricas_validacion:
    met_df = pd.DataFrame(metricas_validacion)
    imprimir_separador('RESUMEN DE VALIDACIÓN INDEPENDIENTE 2020/21-2024/25')
    for metrica in ['ndcg@30', 'spearman', 'mae', 'rmse']:
        print(f'  {metrica}: {met_df[metrica].mean():.4f}')
print(f'✓ Resumen CSV de validación independiente: {BACKTEST_FILE}')


# ============================================================
# 12. ENTRENAMIENTO FINAL Y PREDICCIÓN 2025/26
# ============================================================

imprimir_separador(f'ENTRENAMIENTO FINAL: {TEMPORADA_PREDICCION}')
print(
    f'La predicción usa los scores logísticos nativos del modelo con z-scores por temporada; '
    f'Optuna usa validación walk-forward anterior a {TEMPORADA_PREDICCION}, sin proyección Bradley–Terry.'
)
orden_objetivo = convertir_temporada_a_orden(TEMPORADA_PREDICCION)
orden_temporadas = ORDEN_TEMPORADAS_FEATURES
train_final = df_features[
    (orden_temporadas < orden_objetivo)
    & df_features['posicion_balon_oro'].notna()
].copy()
pred_final = df_features[
    df_features['temporada'].astype(str).eq(TEMPORADA_PREDICCION)
].copy()

if train_final.empty:
    raise ValueError('El entrenamiento final no contiene temporadas históricas etiquetadas.')
if pred_final.empty:
    raise ValueError(f'No hay candidatos de la temporada {TEMPORADA_PREDICCION}.')

modelo_final = entrenar_modelo(train_final, **PARAMS_OPTUNA_PREDICCION_FINAL)
scores_finales_modelo, X_final = calcular_logits_jugadores(pred_final, modelo_final)
scores_finales_ranking = scores_finales_modelo

ranking_final = pred_final[
    ['temporada', 'nombre', 'club', 'posicion', 'posicion_balon_oro']
].copy().reset_index(drop=True)
ranking_final = ranking_final.rename(columns={'posicion_balon_oro': 'posicion_real'})
ranking_final['score_modelo'] = scores_finales_modelo
ranking_final['score_ranking'] = scores_finales_ranking
ranking_final = ranking_final.sort_values('score_ranking', ascending=False).reset_index(drop=True)
ranking_final['posicion_predicha'] = np.arange(1, len(ranking_final) + 1)

scores_ordenados = ranking_final['score_ranking'].to_numpy(dtype=float)
prob_lider, prob_anterior = probabilidades_duelo_ranking(scores_ordenados)
ranking_final['prob_duelo_contra_lider'] = prob_lider
ranking_final['prob_duelo_contra_anterior'] = prob_anterior

columnas_final = [
    'temporada', 'posicion_predicha', 'posicion_real', 'nombre', 'club', 'posicion',
    'score_modelo', 'score_ranking',
    'prob_duelo_contra_lider', 'prob_duelo_contra_anterior',
]
ranking_final = ranking_final[columnas_final]
archivo_prediccion = os.path.join(
    OUTPUT_DIR, f'prediccion_solo_temporada_{TEMPORADA_PREDICCION.replace("/", "_")}.csv'
)
ranking_final.to_csv(archivo_prediccion, index=False, encoding='utf-8-sig')
print(f'\nTOP 30 DEL RANKING {TEMPORADA_PREDICCION}:')
print(ranking_final.head(30).to_string(index=False))
print(f'✓ Ranking final guardado en: {archivo_prediccion}')


# ============================================================
# 13. EXPORTAR COEFICIENTES E IMPORTANCIA DEL ÚNICO MODELO
# ============================================================

coefs_modelo = modelo_final.coef_.reshape(len(POSICIONES), len(FEATURE_COLS))
filas_importancia = []
TOLERANCIA_IMPACTO = 1e-12
TOLERANCIA_COEFICIENTE = 1e-10
resumen_uso_posicion = {}
nombre_modelo = 'zscore_solo_temporada'

for idx_pos, posicion in enumerate(POSICIONES):
    datos_pos = train_final[train_final['posicion'].eq(posicion)].copy()
    if len(datos_pos):
        values = datos_pos[FEATURE_COLS].to_numpy(dtype=np.float64, copy=True)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        impacto = np.mean(np.abs(values), axis=0) * np.abs(coefs_modelo[idx_pos])
    else:
        impacto = np.zeros(len(FEATURE_COLS), dtype=float)

    filas_posicion = []
    for idx_feat, feature in enumerate(FEATURE_COLS):
        fila = {
            'modelo_normalizacion': nombre_modelo,
            'posicion': posicion,
            'caracteristica': feature,
            'estadistica': re.sub(r'__zscore$', '', feature),
            'coeficiente': float(coefs_modelo[idx_pos, idx_feat]),
            'signo_monotonico': 'negativo' if obtener_signo_restriccion(feature) < 0 else 'positivo',
            'impacto_medio_absoluto': float(np.nan_to_num(impacto[idx_feat], nan=0.0)),
        }
        filas_importancia.append(fila)
        filas_posicion.append(fila)

    impacto_vals = np.array([f['impacto_medio_absoluto'] for f in filas_posicion])
    coef_vals = np.array([f['coeficiente'] for f in filas_posicion])
    resumen_uso_posicion[posicion] = {
        'caracteristicas_utilizadas_posicion': int((impacto_vals > TOLERANCIA_IMPACTO).sum()),
        'coeficientes_no_nulos_posicion': int((np.abs(coef_vals) > TOLERANCIA_COEFICIENTE).sum()),
    }

df_importancia = pd.DataFrame(filas_importancia)
df_importancia['caracteristicas_utilizadas_posicion'] = df_importancia['posicion'].map(
    lambda p: resumen_uso_posicion[p]['caracteristicas_utilizadas_posicion']
)
df_importancia['coeficientes_no_nulos_posicion'] = df_importancia['posicion'].map(
    lambda p: resumen_uso_posicion[p]['coeficientes_no_nulos_posicion']
)
print('\nCARACTERÍSTICAS UTILIZADAS POR POSICIÓN:')
for posicion in POSICIONES:
    resumen = resumen_uso_posicion[posicion]
    print(
        f"  {posicion}: {resumen['caracteristicas_utilizadas_posicion']} con impacto efectivo; "
        f"{resumen['coeficientes_no_nulos_posicion']} coeficientes no nulos"
    )

df_importancia['peso_relativo_porcentaje'] = 0.0
for _, indices in df_importancia.groupby('posicion').groups.items():
    suma = df_importancia.loc[indices, 'impacto_medio_absoluto'].sum()
    if suma > 1e-12:
        df_importancia.loc[indices, 'peso_relativo_porcentaje'] = (
            df_importancia.loc[indices, 'impacto_medio_absoluto'] / suma * 100.0
        )
df_importancia = df_importancia.sort_values(
    ['posicion', 'impacto_medio_absoluto'], ascending=[True, False]
).reset_index(drop=True)
df_importancia.to_csv(COEFICIENTES_FILE, index=False, encoding='utf-8-sig')
df_importancia.to_csv(PESO_ESTADISTICAS_FILE, index=False, encoding='utf-8-sig')
print(f'✓ Coeficientes e importancia guardados en: {COEFICIENTES_FILE}')


# ============================================================
# 14. COMPARACIONES INDIVIDUALES CON CONTRIBUCIONES DEL ÚNICO MODELO
# ============================================================


def buscar_jugador(df_candidatos, patron):
    nombres = df_candidatos['nombre'].astype(str).str.casefold().tolist()
    return next((i for i, nombre in enumerate(nombres) if patron in nombre), None)


def exportar_comparacion(idx_a, idx_b, etiqueta_a, etiqueta_b, archivo):
    if idx_a is None or idx_b is None:
        print(f'⚠ No se encontraron ambos jugadores para {archivo}.')
        return

    datos_pred = pred_final.reset_index(drop=True)
    fila_a = datos_pred.iloc[idx_a]
    fila_b = datos_pred.iloc[idx_b]
    pos_a = str(fila_a['posicion'])
    pos_b = str(fila_b['posicion'])
    idx_pos_a = POSICIONES.index(pos_a) if pos_a in POSICIONES else None
    idx_pos_b = POSICIONES.index(pos_b) if pos_b in POSICIONES else None

    vals_a = np.nan_to_num(
        fila_a[FEATURE_COLS].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    vals_b = np.nan_to_num(
        fila_b[FEATURE_COLS].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    coef_a = coefs_modelo[idx_pos_a] if idx_pos_a is not None else np.zeros(len(FEATURE_COLS))
    coef_b = coefs_modelo[idx_pos_b] if idx_pos_b is not None else np.zeros(len(FEATURE_COLS))
    contrib_a = vals_a * coef_a
    contrib_b = vals_b * coef_b

    raw_a, raw_b = [], []
    for feature in FEATURE_COLS:
        col_raw = re.sub(r'__zscore$', '', feature)
        match_a = df_agrupado[
            df_agrupado['temporada'].eq(TEMPORADA_PREDICCION)
            & df_agrupado['nombre'].eq(str(fila_a['nombre']))
        ]
        match_b = df_agrupado[
            df_agrupado['temporada'].eq(TEMPORADA_PREDICCION)
            & df_agrupado['nombre'].eq(str(fila_b['nombre']))
        ]
        raw_a.append(
            match_a[col_raw].iloc[0]
            if col_raw in df_agrupado.columns and not match_a.empty else np.nan
        )
        raw_b.append(
            match_b[col_raw].iloc[0]
            if col_raw in df_agrupado.columns and not match_b.empty else np.nan
        )

    df_comp = pd.DataFrame({
        'caracteristica': FEATURE_COLS,
        'posicion_jugador_a': pos_a,
        'posicion_jugador_b': pos_b,
        f'valor_real_{etiqueta_a}': raw_a,
        f'valor_real_{etiqueta_b}': raw_b,
        f'zscore_temporada_{etiqueta_a}': vals_a,
        f'zscore_temporada_{etiqueta_b}': vals_b,
        f'coeficiente_{etiqueta_a}': coef_a,
        f'coeficiente_{etiqueta_b}': coef_b,
        f'contribucion_{etiqueta_a}': contrib_a,
        f'contribucion_{etiqueta_b}': contrib_b,
        'diferencia_contribucion_a_menos_b': contrib_a - contrib_b,
    })
    df_comp['abs_diferencia_contribucion'] = df_comp[
        'diferencia_contribucion_a_menos_b'
    ].abs()
    df_comp = df_comp.sort_values(
        'abs_diferencia_contribucion', ascending=False
    ).reset_index(drop=True)
    df_comp.to_csv(archivo, index=False, encoding='utf-8-sig')
    print(f'✓ Comparación guardada en: {archivo}')


pred_final_reset = pred_final.reset_index(drop=True)
nombres_prediccion = pred_final_reset['nombre'].astype(str).str.casefold().tolist()
idx_mbappe = buscar_jugador(pred_final_reset, 'mbapp')
idx_yamal = next(
    (i for i, n in enumerate(nombres_prediccion) if 'lamine' in n or 'yamal' in n),
    None,
)
idx_cubarsi = buscar_jugador(pred_final_reset, 'cubars')
exportar_comparacion(idx_mbappe, idx_yamal, 'mbappe', 'lamine', COMPARACION_DUELO_FILE)
exportar_comparacion(idx_cubarsi, idx_yamal, 'cubarsi', 'lamine', COMPARACION_DUELO_CUBARSI_FILE)

imprimir_separador('PROCESO COMPLETADO: MODELO ÚNICO SIN PROYECCIÓN BRADLEY–TERRY')
print(f'✓ Resumen de validación independiente: {BACKTEST_FILE}')
print(f'✓ Mejores parámetros de evaluación: {OPTUNA_EVALUACION_BEST_FILE}')
print(f'✓ Mejores parámetros para 2025/26: {OPTUNA_PREDICCION_BEST_FILE}')
print(f'✓ Ranking final: {archivo_prediccion}')
print(f'✓ Coeficientes e importancia: {COEFICIENTES_FILE}')
