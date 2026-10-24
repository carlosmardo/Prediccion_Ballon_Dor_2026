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
    OUTPUT_DIR, 'resumen_backtest_logistic_posicion_pairwise.csv'
)
OPTUNA_EVALUACION_TRIALS_FILE = os.path.join(
    OUTPUT_DIR, 'optuna_trials_evaluacion_hasta_2019_20.csv'
)
OPTUNA_EVALUACION_BEST_FILE = os.path.join(
    OUTPUT_DIR, 'optuna_best_params_evaluacion_hasta_2019_20.csv'
)
OPTUNA_PREDICCION_TRIALS_FILE = os.path.join(
    OUTPUT_DIR, 'optuna_trials_prediccion_2025_26.csv'
)
OPTUNA_PREDICCION_BEST_FILE = os.path.join(
    OUTPUT_DIR, 'optuna_best_params_prediccion_2025_26.csv'
)
COEFICIENTES_FILE = os.path.join(
    OUTPUT_DIR, 'coeficientes_logistic_posicion_pairwise.csv'
)
PESO_ESTADISTICAS_FILE = os.path.join(
    OUTPUT_DIR, 'importancia_estadisticas_por_posicion.csv'
)
COMPARACION_DUELO_FILE = os.path.join(
    OUTPUT_DIR, 'comparacion_mbappe_lamine_posicion_pairwise.csv'
)
COMPARACION_DUELO_CUBARSI_FILE = os.path.join(
    OUTPUT_DIR, 'comparacion_cubarsi_lamine_posicion_pairwise.csv'
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
# 6. GENERAR DOS VERSIONES DE Z-SCORES Y MÁSCARAS
# ============================================================

imprimir_separador('GENERANDO Z-SCORES: TEMPORADA Y POSICIÓN / SOLO TEMPORADA')

categorias_feature = sorted(
    {limpiar_nombre_columna(c) for c in categorias_unicas},
    key=len,
    reverse=True
)


def generar_features_normalizadas(df_base, normalizar_por_posicion):
    """Crea una versión de características normalizadas según el modo elegido.

    normalizar_por_posicion=True: media y desviación por temporada, posición y feature.
    normalizar_por_posicion=False: media y desviación por temporada y feature.
    Las máscaras y el tratamiento de NaN mantienen la lógica del pipeline actual.
    """
    features_transformadas = []
    partes_procesadas = []

    columnas_grupo = ['temporada', 'posicion'] if normalizar_por_posicion else ['temporada']

    for _, grupo_original in df_base.groupby(columnas_grupo, sort=False):
        grupo = grupo_original.copy()
        temporada = str(grupo['temporada'].iloc[0]).strip()
        nuevas = {}

        for feature in features_base_creadas:
            valores = pd.to_numeric(grupo[feature], errors='coerce').replace([np.inf, -np.inf], np.nan)
            mascara_validos = valores.notna().to_numpy()
            zscore = pd.Series(0.0, index=grupo.index, dtype=float)

            competicion_de_feature = next(
                (cat for cat in categorias_feature if feature.startswith(f'{cat}_')),
                None
            )
            if competicion_de_feature is None:
                raise ValueError(f'No se pudo identificar la categoría de la característica: {feature}')

            jugo_competicion = np.array([
                competicion_de_feature in [
                    limpiar_nombre_columna(c)
                    for c in competiciones_jugador.get((temporada, nombre), [])
                ]
                for nombre in grupo['nombre']
            ])

            stat_base = None
            for stat in list(estadisticas_aditivas) + list(MAPEO_PORCENTAJES.keys()) + [
                'goalsAssistsPerMins', 'resultado_competicion'
            ]:
                if feature.endswith(f'_{stat}') or feature == stat:
                    stat_base = stat
                    break

            es_portero = grupo['posicion'].eq('Goalkeeper').to_numpy()
            if stat_base == 'resultado_competicion' or (
                stat_base and stat_base.endswith('_resultado_competicion')
            ):
                es_compatible_posicion = np.ones(len(grupo), dtype=bool)
            else:
                es_compatible_posicion = np.where(
                    es_portero,
                    np.array([stat_base in ESTADISTICAS_PERMITIDAS_PORTERO for _ in range(len(grupo))]),
                    np.array([stat_base not in ESTADISTICAS_EXCLUSIVAS_PORTERO for _ in range(len(grupo))])
                )

            mascara_imputable = (~mascara_validos) & jugo_competicion & es_compatible_posicion

            if mascara_validos.any():
                valores_validos = valores.loc[mascara_validos]
                media = float(valores_validos.mean())
                desviacion = float(valores_validos.std(ddof=0))
                if desviacion > 1e-12:
                    zscore.loc[mascara_validos] = ((valores_validos - media) / desviacion).to_numpy()
                    zscore.loc[mascara_imputable] = 0.0
                    zscore.loc[(~mascara_validos) & (~mascara_imputable)] = (0.0 - media) / desviacion
                else:
                    zscore.loc[:] = 0.0

            nombre_zscore = f'{feature}__zscore'
            nombre_mask = f'{feature}__mask'
            mascara_efectiva = mascara_validos | mascara_imputable
            nuevas[nombre_zscore] = zscore.to_numpy(dtype=np.float32)
            nuevas[nombre_mask] = mascara_efectiva.astype(np.float32)

            if nombre_zscore not in features_transformadas:
                features_transformadas.append(nombre_zscore)

        grupo = pd.concat([grupo, pd.DataFrame(nuevas, index=grupo.index)], axis=1)
        partes_procesadas.append(grupo)

    # Restaurar exactamente el orden de df_agrupado para alinear los dos modelos.
    df_resultado = pd.concat(partes_procesadas, axis=0).sort_index().reset_index(drop=True)
    return df_resultado, list(dict.fromkeys(features_transformadas))


# Modelo A: compara cada jugador con otros de su misma posición y temporada.
df_features_posicional, FEATURE_COLS = generar_features_normalizadas(
    df_agrupado, normalizar_por_posicion=True
)

# Modelo B: compara cada jugador con todos los jugadores de la temporada.
df_features_global, FEATURE_COLS_GLOBAL = generar_features_normalizadas(
    df_agrupado, normalizar_por_posicion=False
)

if FEATURE_COLS != FEATURE_COLS_GLOBAL:
    raise ValueError('Las versiones posicional y global no generaron las mismas características.')

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

    mask_cols = [col.replace('__zscore', '__mask') for col in feature_cols]
    faltan_masks = [col for col in mask_cols if col not in datos.columns]
    if faltan_masks:
        raise ValueError(f'No se encuentran las máscaras de estas características: {faltan_masks[:10]}')

    features = datos[feature_cols].to_numpy(dtype=np.float32, copy=True)
    masks = datos[mask_cols].to_numpy(dtype=np.float32, copy=True)
    efectivos = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0) * np.nan_to_num(
        masks, nan=0.0, posinf=0.0, neginf=0.0
    )

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
            weights = np.cbrt(np.clip(delta_ndcg / idcg, 0.0, None)) #sqrt cbrt
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
                'maxls': 40,
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


def normalizar_scores_continuos(scores):
    """Estandariza scores continuos dentro de un conjunto de jugadores.

    Se usa z-score para poner ambos modelos en una escala comparable sin
    convertir sus resultados en percentiles discretos. Si el modelo produce
    prácticamente el mismo score para todos, devuelve ceros en vez de amplificar
    diferencias numéricas insignificantes.
    """
    valores = np.asarray(scores, dtype=np.float64).ravel()
    if valores.size == 0:
        return valores
    if not np.isfinite(valores).all():
        raise ValueError('Los scores contienen NaN o infinitos antes de normalizar.')

    media = float(np.mean(valores))
    desviacion = float(np.std(valores, ddof=0))
    if not np.isfinite(desviacion) or desviacion <= 1e-12:
        return np.zeros_like(valores, dtype=np.float64)
    return (valores - media) / desviacion


def calcular_probabilidad_duelo_ensamble(
    score_pos_a, score_pos_b, score_global_a, score_global_b,
    peso_posicional=0.5
):
    """Combina las probabilidades nativas de las dos regresiones logísticas.

    P(A>B) = peso_posicional * sigmoid(score_pos_A - score_pos_B)
             + (1-peso_posicional) * sigmoid(score_global_A - score_global_B).
    No aplica calibración posterior.
    """
    if not 0.0 <= peso_posicional <= 1.0:
        raise ValueError('peso_posicional debe estar entre 0 y 1.')
    p_posicional = expit(np.clip(np.asarray(score_pos_a) - np.asarray(score_pos_b), -50.0, 50.0))
    p_global = expit(np.clip(np.asarray(score_global_a) - np.asarray(score_global_b), -50.0, 50.0))
    return peso_posicional * p_posicional + (1.0 - peso_posicional) * p_global


def matriz_probabilidades_ensamble(scores_posicionales, scores_globales, peso_posicional=0.5):
    """Matriz P[i, j] = probabilidad nativa combinada de que i supere a j."""
    scores_pos = np.asarray(scores_posicionales, dtype=np.float64).ravel()
    scores_global = np.asarray(scores_globales, dtype=np.float64).ravel()
    if len(scores_pos) != len(scores_global):
        raise ValueError('Los scores posicionales y globales no tienen el mismo tamaño.')
    if not np.isfinite(scores_pos).all() or not np.isfinite(scores_global).all():
        raise ValueError('Hay scores no finitos al calcular la matriz de probabilidades.')
    if not 0.0 <= peso_posicional <= 1.0:
        raise ValueError('peso_posicional debe estar entre 0 y 1.')

    diferencias_pos = scores_pos[:, None] - scores_pos[None, :]
    diferencias_global = scores_global[:, None] - scores_global[None, :]
    p_posicional = expit(np.clip(diferencias_pos, -50.0, 50.0))
    p_global = expit(np.clip(diferencias_global, -50.0, 50.0))
    matriz = peso_posicional * p_posicional + (1.0 - peso_posicional) * p_global
    # Una comparación de un jugador consigo mismo no es un duelo; 0.5 permite
    # descontar esa diagonal al calcular las victorias esperadas.
    np.fill_diagonal(matriz, 0.5)
    return matriz


def proyectar_probabilidades_a_scores_transitivos(matriz_probabilidades, regularizacion=1e-6):
    """Ajusta utilidades Bradley-Terry a la matriz de probabilidades combinadas.

    La matriz original es la media de las probabilidades nativas de los dos
    modelos y puede tener ciclos. Esta proyección obtiene una utilidad escalar
    por jugador minimizando la entropía cruzada entre las probabilidades
    combinadas originales y sigmoid(utilidad_i - utilidad_j).

    El resultado produce probabilidades transitivas: si utilidad_i > utilidad_j,
    entonces P(i supera a j) > 0.5. No es calibración supervisada externa; es una
    proyección de las probabilidades combinadas a un modelo probabilístico global.
    """
    p_matrix = np.asarray(matriz_probabilidades, dtype=np.float64)
    if p_matrix.ndim != 2 or p_matrix.shape[0] != p_matrix.shape[1]:
        raise ValueError('La matriz de probabilidades debe ser cuadrada.')
    if not np.isfinite(p_matrix).all():
        raise ValueError('La matriz de probabilidades contiene valores no finitos.')

    n = p_matrix.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    i, j = np.triu_indices(n, k=1)
    targets = np.clip(p_matrix[i, j], 1e-12, 1.0 - 1e-12)
    n_pairs = len(i)

    def loss_and_gradient(utilities):
        differences = utilities[i] - utilities[j]
        probabilities = expit(differences)
        loss = float(
            np.mean(np.logaddexp(0.0, differences) - targets * differences)
            + 0.5 * regularizacion * np.dot(utilities, utilities)
        )
        residuals = (probabilities - targets) / n_pairs
        gradient = np.zeros(n, dtype=np.float64)
        np.add.at(gradient, i, residuals)
        np.add.at(gradient, j, -residuals)
        gradient += regularizacion * utilities
        return loss, gradient

    resultado = minimize(
        loss_and_gradient,
        x0=np.zeros(n, dtype=np.float64),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40},
    )
    if not np.isfinite(resultado.x).all():
        raise RuntimeError('La proyección transitiva produjo utilidades no finitas.')
    if not resultado.success:
        raise RuntimeError(f'No convergió la proyección Bradley-Terry: {resultado.message}')

    # La utilidad solo se identifica hasta una constante aditiva; centramos en 0.
    utilities = resultado.x - np.mean(resultado.x)
    return utilities


def combinar_scores_por_temporada(scores_posicionales, scores_globales, datos, peso_posicional=0.5):
    """Combina probabilidades nativas y las proyecta a un ranking transitivo.

    Primero calcula para cada pareja el promedio ponderado de las probabilidades
    sigmoid nativas del modelo posicional y global. Después ajusta una utilidad
    Bradley-Terry global a esa matriz. El ranking y las probabilidades mostradas
    usan esas mismas utilidades, de modo que la probabilidad de que un jugador
    supere a otro es >50 % exactamente cuando su utilidad/ranking es superior.

    La proyección puede modificar ligeramente las probabilidades originales
    porque no siempre son transitivas; no se aplica calibración posterior basada
    en resultados reales.
    """
    scores_posicionales = np.asarray(scores_posicionales, dtype=np.float64).ravel()
    scores_globales = np.asarray(scores_globales, dtype=np.float64).ravel()
    datos = datos.reset_index(drop=True)
    if not (len(scores_posicionales) == len(scores_globales) == len(datos)):
        raise ValueError('Los scores de ambos modelos y los datos no están alineados.')
    if not 0.0 <= peso_posicional <= 1.0:
        raise ValueError('peso_posicional debe estar entre 0 y 1.')

    ensemble = np.zeros(len(datos), dtype=np.float64)
    for _, indices in datos.groupby('temporada', sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        matriz_p = matriz_probabilidades_ensamble(
            scores_posicionales[idx], scores_globales[idx], peso_posicional
        )
        ensemble[idx] = proyectar_probabilidades_a_scores_transitivos(matriz_p)
    return ensemble

def posiciones_predichas(scores):
    """Devuelve posiciones 1..N, ordenadas de mayor a menor score."""
    return pd.Series(np.asarray(scores, dtype=float)).rank(
        ascending=False, method='first'
    ).to_numpy(dtype=int)



def probabilidades_duelo_ranking(scores_ensamble_ordenados):
    """Probabilidades coherentes con el ranking, usando utilidades transitivas.

    Las utilidades son el ajuste Bradley-Terry a la matriz de probabilidades
    nativas combinadas. Al ordenar los jugadores por estas utilidades, la
    probabilidad de que un jugador supere al líder o al anterior nunca excede
    el 50 % para un jugador situado por debajo.
    """
    scores = np.asarray(scores_ensamble_ordenados, dtype=np.float64).ravel()
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

PESO_MODELO_POSICIONAL = 0.5
ORDEN_TEMPORADAS_FEATURES = df_features_posicional['temporada'].astype(str).map(convertir_temporada_a_orden)


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
    etiquetas = pd.to_numeric(df_features_posicional['posicion_balon_oro'], errors='coerce')

    for temporada_validacion in sorted(set(candidatas), key=convertir_temporada_a_orden):
        corte = convertir_temporada_a_orden(temporada_validacion)
        mascara_train = (orden_temporadas < corte) & etiquetas.notna()
        mascara_valid = df_features_posicional['temporada'].astype(str).eq(temporada_validacion) & etiquetas.notna()
        train = df_features_posicional.loc[mascara_train]
        valid = df_features_posicional.loc[mascara_valid]

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
        c_val = trial.suggest_float('C', 1.0, 75.0, log=True)
        ftol_val = trial.suggest_categorical(
            'ftol', [1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11,
                     1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        )
        gtol_val = trial.suggest_categorical(
            'gtol', [1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11,
                     1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        )
        epsilon_val = trial.suggest_categorical(
            'epsilon', [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        )

        metricas_ensamble = []
        folds_evaluados = 0
        for temporada_validacion in temporadas_validacion:
            corte = convertir_temporada_a_orden(temporada_validacion)
            orden_temporadas = ORDEN_TEMPORADAS_FEATURES
            train_pos = df_features_posicional[
                (orden_temporadas < corte)
                & df_features_posicional['posicion_balon_oro'].notna()
            ].copy()
            valid_pos = df_features_posicional[
                df_features_posicional['temporada'].astype(str).eq(temporada_validacion)
                & df_features_posicional['posicion_balon_oro'].notna()
            ].copy()

            if train_pos.empty or len(valid_pos) < 2:
                continue
            if train_pos.groupby('temporada')['posicion_balon_oro'].nunique().ge(2).sum() == 0:
                continue
            if valid_pos['posicion_balon_oro'].nunique() < 2:
                continue

            train_global = df_features_global.loc[train_pos.index].copy()
            valid_global = df_features_global.loc[valid_pos.index].copy()
            try:
                modelo_pos = entrenar_modelo(train_pos, c_val, ftol_val, gtol_val, epsilon_val)
                modelo_global = entrenar_modelo(train_global, c_val, ftol_val, gtol_val, epsilon_val)
                scores_pos, _ = calcular_logits_jugadores(valid_pos, modelo_pos)
                scores_global, _ = calcular_logits_jugadores(valid_global, modelo_global)
                scores_ensamble = combinar_scores_por_temporada(
                    scores_pos, scores_global, valid_pos, PESO_MODELO_POSICIONAL
                )
                metricas_ensamble.append(evaluar_ranking(scores_ensamble, valid_pos)['ndcg@30'])
                folds_evaluados += 1
            except (RuntimeError, ValueError) as exc:
                print(f'⚠ Optuna {nombre_fase}, trial {trial.number}, temporada {temporada_validacion}: {exc}')
                raise optuna.exceptions.TrialPruned()

        if not metricas_ensamble:
            raise optuna.exceptions.TrialPruned('No hubo folds temporales válidos en este ensayo.')
        trial.set_user_attr('folds_evaluados', folds_evaluados)
        return float(np.mean(metricas_ensamble))

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
        sampler=optuna.samplers.TPESampler(seed=42),
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
temporadas_disponibles = set(df_features_posicional['temporada'].astype(str).unique())
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
    'pairwise_evaluacion_hasta_2019_20',
    OPTUNA_EVALUACION_TRIALS_FILE,
    OPTUNA_EVALUACION_BEST_FILE,
)

# FASE B: segunda búsqueda para 2025/26, utilizando todos los folds históricos elegibles.
estudio_prediccion, PARAMS_OPTUNA_PREDICCION_FINAL = ejecutar_optuna_fase(
    temporadas_optuna_final,
    N_TRIALS_OPTUNA_PREDICCION_FINAL,
    'pairwise_prediccion_2025_26',
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
    train_pos = df_features_posicional[
        (orden_temporadas < corte)
        & df_features_posicional['posicion_balon_oro'].notna()
    ].copy()
    valid_pos = df_features_posicional[
        df_features_posicional['temporada'].astype(str).eq(temporada_validacion)
        & df_features_posicional['posicion_balon_oro'].notna()
    ].copy()
    train_global = df_features_global.loc[train_pos.index].copy()
    valid_global = df_features_global.loc[valid_pos.index].copy()

    if train_pos.empty or len(valid_pos) < 2:
        print(f'⚠ Se omite {temporada_validacion}: train o validación vacíos/insuficientes.')
        continue

    modelo_pos = entrenar_modelo(train_pos, **PARAMS_OPTUNA_EVALUACION)
    modelo_global = entrenar_modelo(train_global, **PARAMS_OPTUNA_EVALUACION)

    scores_train_pos, _ = calcular_logits_jugadores(train_pos, modelo_pos)
    scores_train_global, _ = calcular_logits_jugadores(train_global, modelo_global)
    scores_train_ensamble = combinar_scores_por_temporada(
        scores_train_pos, scores_train_global, train_pos, PESO_MODELO_POSICIONAL
    )

    scores_valid_pos, _ = calcular_logits_jugadores(valid_pos, modelo_pos)
    scores_valid_global, _ = calcular_logits_jugadores(valid_global, modelo_global)
    scores_valid_ensamble = combinar_scores_por_temporada(
        scores_valid_pos, scores_valid_global, valid_pos, PESO_MODELO_POSICIONAL
    )

    metricas_train = evaluar_resumen_por_temporada(scores_train_ensamble, train_pos)
    metricas_valid = evaluar_ranking(scores_valid_ensamble, valid_pos)
    metricas_valid_pos = evaluar_ranking(scores_valid_pos, valid_pos)
    metricas_valid_global = evaluar_ranking(scores_valid_global, valid_pos)
    metricas_validacion.append(metricas_valid)

    ranking = valid_pos[
        ['temporada', 'nombre', 'club', 'posicion', 'posicion_balon_oro']
    ].copy().reset_index(drop=True)
    ranking = ranking.rename(columns={'posicion_balon_oro': 'posicion_real'})
    ranking['score_modelo_posicional'] = scores_valid_pos
    ranking['score_modelo_global'] = scores_valid_global
    ranking['score_ensamble'] = scores_valid_ensamble
    ranking = ranking.sort_values('score_ensamble', ascending=False).reset_index(drop=True)
    ranking['posicion_predicha'] = np.arange(1, len(ranking) + 1)

    scores_ensamble_ordenados = ranking['score_ensamble'].to_numpy(dtype=float)
    prob_lider, prob_anterior = probabilidades_duelo_ranking(scores_ensamble_ordenados)
    ranking['prob_duelo_contra_lider'] = prob_lider
    ranking['prob_duelo_contra_anterior'] = prob_anterior

    columnas_ranking = [
        'temporada', 'posicion_predicha', 'posicion_real', 'nombre', 'club', 'posicion',
        'score_ensamble', 'score_modelo_posicional', 'score_modelo_global',
        'prob_duelo_contra_lider', 'prob_duelo_contra_anterior',
    ]
    ranking = ranking[columnas_ranking]
    temp_clean = temporada_validacion.replace('/', '_')
    archivo_ranking = os.path.join(OUTPUT_DIR, f'prediccion_ensamble_backtest_{temp_clean}.csv')
    ranking.to_csv(archivo_ranking, index=False, encoding='utf-8-sig')

    gap = metricas_train['ndcg@30'] - metricas_valid['ndcg@30']
    resumen_backtest.append({
        'temporada': temporada_validacion,
        'rol_temporada': 'validacion_independiente',
        **{f'train_ensamble_{k}': v for k, v in metricas_train.items()},
        **{f'valid_ensamble_{k}': v for k, v in metricas_valid.items()},
        'valid_posicional_ndcg@30': metricas_valid_pos['ndcg@30'],
        'valid_global_ndcg@30': metricas_valid_global['ndcg@30'],
        'gap_ndcg30_ensamble': gap,
        'C': PARAMS_OPTUNA_EVALUACION['C'],
        'ftol': PARAMS_OPTUNA_EVALUACION['ftol'],
        'gtol': PARAMS_OPTUNA_EVALUACION['gtol'],
        'epsilon': PARAMS_OPTUNA_EVALUACION['epsilon'],
        'peso_modelo_posicional': PESO_MODELO_POSICIONAL,
        'archivo_ranking': archivo_ranking,
    })

    print(f'\nTemporada de validación independiente: {temporada_validacion}')
    print(f"  NDCG@30 posicional: {metricas_valid_pos['ndcg@30']:.4f}")
    print(f"  NDCG@30 global:     {metricas_valid_global['ndcg@30']:.4f}")
    print(f"  NDCG@30 ensamble:   {metricas_valid['ndcg@30']:.4f} | Spearman: {metricas_valid['spearman']:.4f}")
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

imprimir_separador(f'ENTRENAMIENTO FINAL Y ENSAMBLE {TEMPORADA_PREDICCION}')
print(f'Esta predicción utiliza la segunda búsqueda de Optuna con validación walk-forward de todo el histórico anterior a {TEMPORADA_PREDICCION}.')
orden_objetivo = convertir_temporada_a_orden(TEMPORADA_PREDICCION)
orden_temporadas = ORDEN_TEMPORADAS_FEATURES
train_final_posicional = df_features_posicional[
    (orden_temporadas < orden_objetivo)
    & df_features_posicional['posicion_balon_oro'].notna()
].copy()
pred_final_posicional = df_features_posicional[
    df_features_posicional['temporada'].astype(str).eq(TEMPORADA_PREDICCION)
].copy()
train_final_global = df_features_global.loc[train_final_posicional.index].copy()
pred_final_global = df_features_global.loc[pred_final_posicional.index].copy()

if train_final_posicional.empty:
    raise ValueError('El entrenamiento final no contiene temporadas históricas etiquetadas.')
if pred_final_posicional.empty:
    raise ValueError(f'No hay candidatos de la temporada {TEMPORADA_PREDICCION}.')

modelo_final_posicional = entrenar_modelo(
    train_final_posicional, **PARAMS_OPTUNA_PREDICCION_FINAL
)
modelo_final_global = entrenar_modelo(
    train_final_global, **PARAMS_OPTUNA_PREDICCION_FINAL
)

scores_finales_posicional, X_final_posicional = calcular_logits_jugadores(
    pred_final_posicional, modelo_final_posicional
)
scores_finales_global, X_final_global = calcular_logits_jugadores(
    pred_final_global, modelo_final_global
)
scores_finales_ensamble = combinar_scores_por_temporada(
    scores_finales_posicional, scores_finales_global,
    pred_final_posicional, PESO_MODELO_POSICIONAL
)

ranking_final = pred_final_posicional[
    ['temporada', 'nombre', 'club', 'posicion', 'posicion_balon_oro']
].copy().reset_index(drop=True)
ranking_final = ranking_final.rename(columns={'posicion_balon_oro': 'posicion_real'})
ranking_final['score_modelo_posicional'] = scores_finales_posicional
ranking_final['score_modelo_global'] = scores_finales_global
ranking_final['score_ensamble'] = scores_finales_ensamble
ranking_final = ranking_final.sort_values('score_ensamble', ascending=False).reset_index(drop=True)
ranking_final['posicion_predicha'] = np.arange(1, len(ranking_final) + 1)

scores_ensamble_ordenados = ranking_final['score_ensamble'].to_numpy(dtype=float)
prob_lider, prob_anterior = probabilidades_duelo_ranking(scores_ensamble_ordenados)
ranking_final['prob_duelo_contra_lider'] = prob_lider
ranking_final['prob_duelo_contra_anterior'] = prob_anterior

columnas_final = [
    'temporada', 'posicion_predicha', 'posicion_real', 'nombre', 'club', 'posicion',
    'score_ensamble', 'score_modelo_posicional', 'score_modelo_global',
    'prob_duelo_contra_lider', 'prob_duelo_contra_anterior',
]
ranking_final = ranking_final[columnas_final]
archivo_prediccion = os.path.join(
    OUTPUT_DIR, f'prediccion_ensamble_{TEMPORADA_PREDICCION.replace("/", "_")}.csv'
)
ranking_final.to_csv(archivo_prediccion, index=False, encoding='utf-8-sig')
print(f'\nTOP 30 DEL RANKING UNIFICADO {TEMPORADA_PREDICCION}:')
print(ranking_final.head(30).to_string(index=False))
print(f'✓ Ranking final guardado en: {archivo_prediccion}')


# ============================================================
# 13. EXPORTAR COEFICIENTES E IMPORTANCIA DE LOS DOS MODELOS
# ============================================================

coefs_posicional = modelo_final_posicional.coef_.reshape(len(POSICIONES), len(FEATURE_COLS))
coefs_global = modelo_final_global.coef_.reshape(len(POSICIONES), len(FEATURE_COLS))
filas_importancia = []
TOLERANCIA_IMPACTO = 1e-12
TOLERANCIA_COEFICIENTE = 1e-10
resumen_uso_posicion = {}

for nombre_modelo, datos_train, coefs_modelo in [
    ('zscore_temporada_posicion', train_final_posicional, coefs_posicional),
    ('zscore_solo_temporada', train_final_global, coefs_global),
]:
    for idx_pos, posicion in enumerate(POSICIONES):
        datos_pos = datos_train[datos_train['posicion'].eq(posicion)].copy()
        if len(datos_pos):
            values = datos_pos[FEATURE_COLS].to_numpy(dtype=np.float64, copy=True)
            mask_cols = [col.replace('__zscore', '__mask') for col in FEATURE_COLS]
            masks = datos_pos[mask_cols].to_numpy(dtype=np.float64, copy=True)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) * np.nan_to_num(
                masks, nan=0.0, posinf=0.0, neginf=0.0
            )
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
        resumen_uso_posicion[(nombre_modelo, posicion)] = {
            'caracteristicas_utilizadas_posicion': int((impacto_vals > TOLERANCIA_IMPACTO).sum()),
            'coeficientes_no_nulos_posicion': int((np.abs(coef_vals) > TOLERANCIA_COEFICIENTE).sum()),
        }

df_importancia = pd.DataFrame(filas_importancia)
df_importancia['caracteristicas_utilizadas_posicion'] = df_importancia.apply(
    lambda r: resumen_uso_posicion[(r['modelo_normalizacion'], r['posicion'])]['caracteristicas_utilizadas_posicion'],
    axis=1,
)
df_importancia['coeficientes_no_nulos_posicion'] = df_importancia.apply(
    lambda r: resumen_uso_posicion[(r['modelo_normalizacion'], r['posicion'])]['coeficientes_no_nulos_posicion'],
    axis=1,
)
print('\nCARACTERÍSTICAS UTILIZADAS POR MODELO Y POSICIÓN:')
for nombre_modelo in ['zscore_temporada_posicion', 'zscore_solo_temporada']:
    print(f'  Modelo: {nombre_modelo}')
    for posicion in POSICIONES:
        resumen = resumen_uso_posicion[(nombre_modelo, posicion)]
        print(
            f"    {posicion}: {resumen['caracteristicas_utilizadas_posicion']} con impacto efectivo; "
            f"{resumen['coeficientes_no_nulos_posicion']} coeficientes no nulos"
        )

df_importancia['peso_relativo_porcentaje'] = 0.0
for _, indices in df_importancia.groupby(['modelo_normalizacion', 'posicion']).groups.items():
    suma = df_importancia.loc[indices, 'impacto_medio_absoluto'].sum()
    if suma > 1e-12:
        df_importancia.loc[indices, 'peso_relativo_porcentaje'] = (
            df_importancia.loc[indices, 'impacto_medio_absoluto'] / suma * 100.0
        )
df_importancia = df_importancia.sort_values(
    ['modelo_normalizacion', 'posicion', 'impacto_medio_absoluto'],
    ascending=[True, True, False]
).reset_index(drop=True)
df_importancia.to_csv(COEFICIENTES_FILE, index=False, encoding='utf-8-sig')
df_importancia.to_csv(PESO_ESTADISTICAS_FILE, index=False, encoding='utf-8-sig')
print(f'✓ Coeficientes e importancia guardados en: {COEFICIENTES_FILE}')


# ============================================================
# 14. COMPARACIONES INDIVIDUALES CON CONTRIBUCIONES DE AMBOS MODELOS
# ============================================================


def buscar_jugador(df_candidatos, patron):
    nombres = df_candidatos['nombre'].astype(str).str.casefold().tolist()
    return next((i for i, nombre in enumerate(nombres) if patron in nombre), None)


def exportar_comparacion(idx_a, idx_b, etiqueta_a, etiqueta_b, archivo):
    if idx_a is None or idx_b is None:
        print(f'⚠ No se encontraron ambos jugadores para {archivo}.')
        return

    fila_a_pos = pred_final_posicional.reset_index(drop=True).iloc[idx_a]
    fila_b_pos = pred_final_posicional.reset_index(drop=True).iloc[idx_b]
    fila_a_glob = pred_final_global.reset_index(drop=True).iloc[idx_a]
    fila_b_glob = pred_final_global.reset_index(drop=True).iloc[idx_b]
    pos_a = str(fila_a_pos['posicion'])
    pos_b = str(fila_b_pos['posicion'])
    idx_pos_a = POSICIONES.index(pos_a) if pos_a in POSICIONES else None
    idx_pos_b = POSICIONES.index(pos_b) if pos_b in POSICIONES else None

    cols_mask = [c.replace('__zscore', '__mask') for c in FEATURE_COLS]
    vals_a_pos = np.nan_to_num(fila_a_pos[FEATURE_COLS].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0) * np.nan_to_num(fila_a_pos[cols_mask].to_numpy(dtype=float), nan=0.0)
    vals_b_pos = np.nan_to_num(fila_b_pos[FEATURE_COLS].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0) * np.nan_to_num(fila_b_pos[cols_mask].to_numpy(dtype=float), nan=0.0)
    vals_a_glob = np.nan_to_num(fila_a_glob[FEATURE_COLS].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0) * np.nan_to_num(fila_a_glob[cols_mask].to_numpy(dtype=float), nan=0.0)
    vals_b_glob = np.nan_to_num(fila_b_glob[FEATURE_COLS].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0) * np.nan_to_num(fila_b_glob[cols_mask].to_numpy(dtype=float), nan=0.0)

    coef_a_pos = coefs_posicional[idx_pos_a] if idx_pos_a is not None else np.zeros(len(FEATURE_COLS))
    coef_b_pos = coefs_posicional[idx_pos_b] if idx_pos_b is not None else np.zeros(len(FEATURE_COLS))
    coef_a_glob = coefs_global[idx_pos_a] if idx_pos_a is not None else np.zeros(len(FEATURE_COLS))
    coef_b_glob = coefs_global[idx_pos_b] if idx_pos_b is not None else np.zeros(len(FEATURE_COLS))
    contrib_a_pos, contrib_b_pos = vals_a_pos * coef_a_pos, vals_b_pos * coef_b_pos
    contrib_a_glob, contrib_b_glob = vals_a_glob * coef_a_glob, vals_b_glob * coef_b_glob

    raw_a, raw_b = [], []
    for feature in FEATURE_COLS:
        col_raw = re.sub(r'__zscore$', '', feature)
        match_a = df_agrupado[
            df_agrupado['temporada'].eq(TEMPORADA_PREDICCION)
            & df_agrupado['nombre'].eq(str(fila_a_pos['nombre']))
        ]
        match_b = df_agrupado[
            df_agrupado['temporada'].eq(TEMPORADA_PREDICCION)
            & df_agrupado['nombre'].eq(str(fila_b_pos['nombre']))
        ]
        raw_a.append(match_a[col_raw].iloc[0] if col_raw in df_agrupado.columns and not match_a.empty else np.nan)
        raw_b.append(match_b[col_raw].iloc[0] if col_raw in df_agrupado.columns and not match_b.empty else np.nan)

    df_comp = pd.DataFrame({
        'caracteristica': FEATURE_COLS,
        'posicion_jugador_a': pos_a,
        'posicion_jugador_b': pos_b,
        f'valor_real_{etiqueta_a}': raw_a,
        f'valor_real_{etiqueta_b}': raw_b,
        f'zscore_posicional_{etiqueta_a}': vals_a_pos,
        f'zscore_posicional_{etiqueta_b}': vals_b_pos,
        f'zscore_global_{etiqueta_a}': vals_a_glob,
        f'zscore_global_{etiqueta_b}': vals_b_glob,
        f'coeficiente_posicional_{etiqueta_a}': coef_a_pos,
        f'coeficiente_posicional_{etiqueta_b}': coef_b_pos,
        f'coeficiente_global_{etiqueta_a}': coef_a_glob,
        f'coeficiente_global_{etiqueta_b}': coef_b_glob,
        f'contribucion_posicional_{etiqueta_a}': contrib_a_pos,
        f'contribucion_posicional_{etiqueta_b}': contrib_b_pos,
        f'contribucion_global_{etiqueta_a}': contrib_a_glob,
        f'contribucion_global_{etiqueta_b}': contrib_b_glob,
        'diferencia_contribucion_posicional_a_menos_b': contrib_a_pos - contrib_b_pos,
        'diferencia_contribucion_global_a_menos_b': contrib_a_glob - contrib_b_glob,
    })
    df_comp['max_abs_diferencia_contribucion'] = np.maximum(
        df_comp['diferencia_contribucion_posicional_a_menos_b'].abs(),
        df_comp['diferencia_contribucion_global_a_menos_b'].abs(),
    )
    df_comp = df_comp.sort_values('max_abs_diferencia_contribucion', ascending=False).reset_index(drop=True)
    df_comp.to_csv(archivo, index=False, encoding='utf-8-sig')
    print(f'✓ Comparación guardada en: {archivo}')


pred_final_reset = pred_final_posicional.reset_index(drop=True)
nombres_prediccion = pred_final_reset['nombre'].astype(str).str.casefold().tolist()
idx_mbappe = buscar_jugador(pred_final_reset, 'mbapp')
idx_yamal = next((i for i, n in enumerate(nombres_prediccion) if 'lamine' in n or 'yamal' in n), None)
idx_cubarsi = buscar_jugador(pred_final_reset, 'cubars')
exportar_comparacion(idx_mbappe, idx_yamal, 'mbappe', 'lamine', COMPARACION_DUELO_FILE)
exportar_comparacion(idx_cubarsi, idx_yamal, 'cubarsi', 'lamine', COMPARACION_DUELO_CUBARSI_FILE)

imprimir_separador('PROCESO COMPLETADO: VALIDACIÓN INDEPENDIENTE Y PREDICCIÓN FINAL')
print(f'✓ Resumen de validación independiente: {BACKTEST_FILE}')
print(f'✓ Mejores parámetros de evaluación: {OPTUNA_EVALUACION_BEST_FILE}')
print(f'✓ Mejores parámetros para 2025/26: {OPTUNA_PREDICCION_BEST_FILE}')
print(f'✓ Ranking final: {archivo_prediccion}')
print(f'✓ Coeficientes e importancia: {COEFICIENTES_FILE}')
