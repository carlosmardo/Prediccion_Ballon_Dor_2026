import re
from pathlib import Path
import pandas as pd
import numpy as np


# ===============================================================
# CONFIGURACIÓN
# ===============================================================

DATA_DIR = Path("data")

archivos = {
    "stats_historico_ballon_dor": DATA_DIR / "stats_historico_ballon_dor.csv",
    "nominados_ballon_dor": DATA_DIR / "nominados_ballon_dor_stats.csv",
    "clubes_competiciones": DATA_DIR / "clubes_competiciones.csv",
    "clubes_ligas": DATA_DIR / "clubes_liga.csv",
    "selecciones_competiciones": DATA_DIR / "selecciones_competiciones.csv",

    # Archivos finales enriquecidos
    "stats_historico_ballon_dor_limpio": DATA_DIR / "stats_historico_ballon_dor_limpio.csv",
    "nominados_ballon_dor_limpio": DATA_DIR / "nominados_ballon_dor_limpio.csv",
}

# Cargar datasets iniciales en memoria para conservar los CSV originales intactos
dataframes = {
    nombre: pd.read_csv(archivo, low_memory=False)
    for nombre, archivo in archivos.items()
    if not nombre.endswith("_limpio")
}


# ===============================================================
# 1. INSPECCIÓN INICIAL DE TODOS LOS DATASETS
# ===============================================================

print("\n" + "=" * 80 + "\nINSPECCIÓN INICIAL DE LOS DATASETS\n" + "=" * 80)

for nombre, df in dataframes.items():
    print(f"\n" + "=" * 80 + f"\n{nombre}\n" + "=" * 80)
    print(f"Filas: {len(df):,}\nColumnas: {len(df.columns):,}\n\nColumnas:")

    for i, columna in enumerate(df.columns, start=1):
        print(f"{i:3}. {columna}")


# ===============================================================
# 2. ELIMINAR COLUMNAS INNECESARIAS
# ===============================================================

columnas_eliminar = ["countRating", "id", "rating", "statisticsType", "substitutionsIn", "substitutionsOut", "totalRating", "type"]
archivos_stats = ["stats_historico_ballon_dor", "nominados_ballon_dor"]

print("\n" + "=" * 80 + "\nELIMINACIÓN DE COLUMNAS\n" + "=" * 80)

for nombre in archivos_stats:
    df = dataframes[nombre]
    columnas_antes = len(df.columns)
    df = df.drop(columns=columnas_eliminar, errors="ignore")
    dataframes[nombre] = df
    print(f"{nombre}: {columnas_antes - len(df.columns)} columnas eliminadas | {len(df.columns)} restantes")


# ===============================================================
# 3. NORMALIZAR CLUBES MÚLTIPLES
# ===============================================================

df = dataframes["stats_historico_ballon_dor"]
clubes_multiples_antes = df["club"].dropna().astype(str).str.contains("/", regex=False).sum()
df["club"] = df["club"].str.split("/").str[0].str.strip()
dataframes["stats_historico_ballon_dor"] = df

print("\n" + "=" * 80 + "\nCLUBES CON MÚLTIPLES EQUIPOS\n" + "=" * 80)
print(f"Filas con múltiples clubes normalizadas: {clubes_multiples_antes:,}")


# ===============================================================
# 4. JUGADORES SIN NINGUNA ESTADÍSTICA
# ===============================================================

df = dataframes["stats_historico_ballon_dor"]
columnas_identificacion = ["temporada", "posicion_balon_oro", "club", "posicion", "nacionalidad", "nombre", "id_sofascore"]
columnas_estadisticas = [col for col in df.columns if col not in columnas_identificacion]

estadisticas = df[columnas_estadisticas].replace(["", "**"], pd.NA)
sin_estadisticas = estadisticas.isna().all(axis=1)
df_sin_estadisticas = df[sin_estadisticas]

print("\n" + "=" * 80 + "\nJUGADORES SIN NINGUNA ESTADÍSTICA\n" + "=" * 80)
print(f"Filas sin estadísticas: {len(df_sin_estadisticas):,}\nFilas totales: {len(df):,}")

if len(df) > 0: print(f"Porcentaje: {len(df_sin_estadisticas) / len(df) * 100:.2f}%")

print("\nJugadores:\n" + df_sin_estadisticas[["temporada", "posicion_balon_oro", "nombre", "club", "id_sofascore"]].to_string(index=False))


# ===============================================================
# 5. VER EQUIPOS, SELECCIONES Y NACIONALIDADES
# ===============================================================

print("\n" + "=" * 80 + "\nEQUIPOS, SELECCIONES Y NACIONALIDADES POR DATASET\n" + "=" * 80)

for nombre, df in dataframes.items():
    print(f"\n" + "=" * 80 + f"\n{nombre}\n" + "=" * 80)

    if "club" in df.columns:
        clubes = sorted(df["club"].dropna().astype(str).str.strip().unique())
        print(f"\nEQUIPOS/SELECCIONES ÚNICOS: {len(clubes)}")

        for i, club in enumerate(clubes, start=1): print(f"{i:4}. {club}")

    if "nacionalidad" in df.columns:
        nacionalidades = sorted(df["nacionalidad"].dropna().astype(str).str.strip().unique())
        print(f"\nNACIONALIDADES ÚNICAS: {len(nacionalidades)}")

        for i, nac in enumerate(nacionalidades, start=1): print(f"{i:4}. {nac}")


# ===============================================================
# 6. NORMALIZACIÓN DE CLUBES DENTRO DE stats_historico_ballon_dor
# ===============================================================

normalizacion_clubes_historico = {
    "Ajax Amsterdam": "Ajax", "Atlético Madrid": "Atletico Madrid", "FC Barcelona": "Barcelona",
    "FC Porto": "Porto", "Sevilla": "Sevilla FC", "Tottenham Hotspur": "Tottenham", "VfL Wolfsburg": "Wolfsburg"
}

df = dataframes["stats_historico_ballon_dor"]
df["club"] = df["club"].replace(normalizacion_clubes_historico)
dataframes["stats_historico_ballon_dor"] = df

print("\n" + "=" * 80 + "\nNORMALIZACIÓN INTERNA DE CLUBES\n" + "=" * 80)

for origen, destino in normalizacion_clubes_historico.items(): print(f"{origen} -> {destino}")


# ===============================================================
# 7. TODOS LOS CLUBES ÚNICOS DE TODOS LOS DATASETS
# ===============================================================

todos_clubes = set()
for nombre, df in dataframes.items():
    if "club" in df.columns:
        todos_clubes.update(df["club"].dropna().astype(str).str.strip().unique())

todos_clubes = sorted(todos_clubes)

print("\n" + "=" * 80 + f"\nCLUBES ÚNICOS EN TODOS LOS DATASETS\n" + "=" * 80 + f"\nTotal: {len(todos_clubes)}")

for i, club in enumerate(todos_clubes, start=1): print(f"{i:4}. {club}")


# ===============================================================
# 8. TODAS LAS SELECCIONES / NACIONALIDADES ÚNICAS
# ===============================================================

todos_selecciones = set()

for key in ["stats_historico_ballon_dor", "nominados_ballon_dor"]:
    df = dataframes[key]
    todos_selecciones.update(df["nacionalidad"].dropna().astype(str).str.strip().unique())

df = dataframes["selecciones_competiciones"]
todos_selecciones.update(df["club"].dropna().astype(str).str.strip().unique())
todos_selecciones = sorted(todos_selecciones)

print("\n" + "=" * 80 + f"\nSELECCIONES / NACIONALIDADES ÚNICAS EN TODOS LOS DATASETS\n" + "=" * 80 + f"\nTotal: {len(todos_selecciones)}")

for i, seleccion in enumerate(todos_selecciones, start=1): print(f"{i:4}. {seleccion}")


# ===============================================================
# 9. NORMALIZACIÓN GLOBAL DE CLUBES
# ===============================================================

normalizacion_clubes_global = {"FC Barcelona": "Barcelona"}

print("\n" + "=" * 80 + "\nNORMALIZACIÓN GLOBAL DE CLUBES\n" + "=" * 80)

for origen, destino in normalizacion_clubes_global.items(): print(f"{origen} -> {destino}")

for nombre in ["stats_historico_ballon_dor", "nominados_ballon_dor", "clubes_competiciones", "clubes_ligas"]:
    df = dataframes[nombre]
    df["club"] = df["club"].replace(normalizacion_clubes_global)
    dataframes[nombre] = df
    print(f"[OK] {nombre}")


# ===============================================================
# 10. NORMALIZACIÓN GLOBAL DE SELECCIONES / NACIONALIDADES
# ===============================================================

normalizacion_selecciones = {
    "Bosnia Herzegovina": "Bosnia & Herzegovina", "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire", "Turkey": "Türkiye"
}

print("\n" + "=" * 80 + "\nNORMALIZACIÓN GLOBAL DE SELECCIONES / NACIONALIDADES\n" + "=" * 80)

for origen, destino in normalizacion_selecciones.items(): print(f"{origen} -> {destino}")

for nombre, col in [("stats_historico_ballon_dor", "nacionalidad"), ("nominados_ballon_dor", "nacionalidad"), ("selecciones_competiciones", "club")]:
    df = dataframes[nombre]
    df[col] = df[col].replace(normalizacion_selecciones)
    dataframes[nombre] = df
    print(f"[OK] {nombre}")


# ===============================================================
# 11. VER COMPETICIONES Y LIGAS ÚNICAS POR DATASET
# ===============================================================

print("\n" + "=" * 80 + "\nCOMPETICIONES Y LIGAS ÚNICAS POR DATASET\n" + "=" * 80)

columnas_competicion = {
    "stats_historico_ballon_dor": "competicion", "nominados_ballon_dor": "competicion",
    "clubes_competiciones": "competicion", "clubes_ligas": "liga", "selecciones_competiciones": "competicion"
}

for nombre, columna in columnas_competicion.items():
    df = dataframes[nombre]
    valores = sorted(df[columna].dropna().astype(str).str.strip().unique())
    print(f"\n" + "=" * 80 + f"\n{nombre}\n" + "=" * 80 + f"\nColumna: {columna}\nTotal de nombres únicos: {len(valores)}")
    for i, valor in enumerate(valores, start=1): print(f"{i:4}. {valor}")


# ===============================================================
# 12. ELIMINAR FILA INCORRECTA DE clubes_competiciones
# ===============================================================

df = dataframes["clubes_competiciones"]
filas_antes = len(df)
df = df[~df["competicion"].astype(str).str.strip().eq("competicion")].copy()
dataframes["clubes_competiciones"] = df
print("\n" + "=" * 80 + f"\nELIMINACIÓN DE FILA INCORRECTA\n" + "=" * 80 + f"\nclubes_competiciones: {filas_antes - len(df)} fila(s) eliminada(s)")


# ===============================================================
# 13. NORMALIZACIÓN DE COMPETICIONES DE CLUBES
# ===============================================================

normalizacion_competiciones_clubes = {
    "Copa de Campeones de Europa": "UEFA Champions League",
    "Copa de la UEFA (- 2009)": "UEFA Europa League",
    "Copa Italia": "Coppa Italia",
    "Supercopa de Italia": "Supercoppa Italiana",
    "Copa de Alemania": "DFB Pokal",
    "Eurojackpot KNVB Beker": "KNVB Beker",
    "Mundial de Clubes FIFA": "FIFA Club World Cup",
    "Supercopa de Europa": "UEFA Super Cup",
    "Supertaça Cândido de Oliveira": "Supertaça",
    "Franz-Beckenbauer-Supercup": "Franz Beckenbauer Supercup",
    "Copa de Campeones Concacaf": "CONCACAF Champions Cup",
    "Emirates Cup": "FA Cup",
}

print("\n" + "=" * 80 + "\nNORMALIZACIÓN DE COMPETICIONES DE CLUBES\n" + "=" * 80)

for origen, destino in normalizacion_competiciones_clubes.items(): print(f"{origen} -> {destino}")

for nombre in ["stats_historico_ballon_dor", "nominados_ballon_dor", "clubes_competiciones"]:
    df = dataframes[nombre]
    df["competicion"] = df["competicion"].replace(normalizacion_competiciones_clubes)
    dataframes[nombre] = df
    print(f"[OK] {nombre}")


# ===============================================================
# 14. NORMALIZACIÓN DE COMPETICIONES DE SELECCIONES
# ===============================================================

normalizacion_competiciones_selecciones = {
    "Copa del Mundo": "FIFA World Cup", "Eurocopa": "EURO", "Clasificación Eurocopa": "EURO, Qualification",
    "Play-off de la Clasificación para la Eurocopa": "EURO, Qualification", "Copa América Centenario": "Copa América",
    "Campeonato Africano de Naciones": "Africa Cup of Nations", "Clasificación Copa África": "Africa Cup of Nations Qual.",
    "Copa Confederaciones": "Confederations Cup", "Copa Rey Fahd 1992": "Confederations Cup",
    "Copa Rey Fahd 1995": "Confederations Cup", "Eliminatorias Sudamericanas": "World Cup Qual. CONMEBOL",
    "Clasificación Mundial Europa": "World Cup Qual. UEFA", "Clasificación Mundial Asia": "World Cup Qual. AFC",
    "Clasificación Mundial África": "World Cup Qual. CAF", "Fase final UEFA Nations League": "UEFA Nations League",
    "UEFA Nations League Play-off": "UEFA Nations League", "Juegos Olímpicos": "Olympic Games"
}

print("\n" + "=" * 80 + "\nNORMALIZACIÓN DE COMPETICIONES DE SELECCIONES\n" + "=" * 80)

for origen, destino in normalizacion_competiciones_selecciones.items(): print(f"{origen} -> {destino}")

for nombre in ["stats_historico_ballon_dor", "nominados_ballon_dor", "selecciones_competiciones"]:
    df = dataframes[nombre]
    df["competicion"] = df["competicion"].replace(normalizacion_competiciones_selecciones)
    dataframes[nombre] = df
    print(f"[OK] {nombre}")


# ===============================================================
# 15. ELIMINAR COMPETICIONES QUE NO SE UTILIZARÁN
# ===============================================================

COMPETICIONES_EXCLUIR = ["MLS All Star Game"]
print("\n" + "=" * 80 + "\nELIMINACIÓN DE COMPETICIONES EXCLUIDAS\n" + "=" * 80)

for nombre in ["stats_historico_ballon_dor", "nominados_ballon_dor"]:
    df = dataframes[nombre]
    filas_antes = len(df)
    df["competicion"] = df["competicion"].astype("string").str.strip()
    df = df[~df["competicion"].isin(COMPETICIONES_EXCLUIR)].copy()
    dataframes[nombre] = df
    print(f"{nombre}: {filas_antes - len(df)} fila(s) eliminada(s)")

# COMPROBACIÓN DE COMPETICIONES EXCLUIDAS
print("\n" + "=" * 80 + "\nCOMPROBACIÓN DE COMPETICIONES EXCLUIDAS\n" + "=" * 80)

for nombre in ["stats_historico_ballon_dor", "nominados_ballon_dor"]:
    df = dataframes[nombre]
    encontradas = df[df["competicion"].astype("string").str.strip().isin(COMPETICIONES_EXCLUIR)]
    if len(encontradas) > 0: print(f"[ERROR] {nombre}: todavía quedan {len(encontradas)} fila(s) de competiciones excluidas.")
    else: print(f"[OK] {nombre}: ninguna competición excluida.")


# ===============================================================
# NUEVA FUNCIÓN: ELIMINAR ICC Y PREMIER LEAGUE ASIA TROPHY
# ===============================================================

def eliminar_icc_y_asia_trophy():
    """
    Elimina todas las filas pertenecientes a:
    - International Champions Cup
    - Premier League Asia Trophy
    """
    print("\n" + "=" * 80 + "\nELIMINACIÓN DE INTERNATIONAL CHAMPIONS CUP Y PREMIER LEAGUE ASIA TROPHY\n" + "=" * 80)
    comps_eliminar = ["International Champions Cup", "Premier League Asia Trophy"]

    for nombre in ["stats_historico_ballon_dor", "nominados_ballon_dor"]:
        df = dataframes[nombre]
        filas_antes = len(df)
        dataframes[nombre] = df[~df["competicion"].astype("string").str.strip().isin(comps_eliminar)].copy()
        print(f"{nombre}: {filas_antes - len(dataframes[nombre])} fila(s) eliminada(s) ({', '.join(comps_eliminar)})")

# Ejecutar la eliminación de ICC y Premier League Asia Trophy
eliminar_icc_y_asia_trophy()


# ===============================================================
# 16. NORMALIZACIÓN DE LIGAS
# ===============================================================

normalizacion_ligas = {
    "Brasileirão Betano": "Campeonato Brasileiro Série A", "Major League Soccer": "MLS",
    "russian premier league": "Premier Liga", "Liga Portugal Betclic": "Liga Portugal",
    "VriendenLoterij Eredivisie": "Eredivisie"
}

print("\n" + "=" * 80 + "\nNORMALIZACIÓN DE LIGAS\n" + "=" * 80)
for origen, destino in normalizacion_ligas.items(): print(f"{origen} -> {destino}")

for nombre, col in [("clubes_ligas", "liga"), ("stats_historico_ballon_dor", "competicion"), ("nominados_ballon_dor", "competicion")]:
    df = dataframes[nombre]
    df[col] = df[col].replace(normalizacion_ligas)
    dataframes[nombre] = df
    print(f"[OK] {nombre}")


# ===============================================================
# 17. NORMALIZAR TODAS LAS TEMPORADAS
# ===============================================================

def normalizar_temporada(valor):
    if pd.isna(valor): return valor
    valor = str(valor).strip()
    if re.fullmatch(r"\d{2}/\d{2}", valor):
        inicio = int(valor.split("/")[0])
        año_inicio = 2000 + inicio if inicio <= 26 else 1900 + inicio
        return f"{año_inicio}/{str(año_inicio + 1)[-2:]}"
    if re.fullmatch(r"\d{4}/\d{2}", valor): return valor
    if re.fullmatch(r"\d{4}", valor):
        año_fin = int(valor)
        return f"{año_fin - 1}/{str(año_fin)[-2:]}"
    return valor

print("\n" + "=" * 80 + "\nPRUEBAS DE NORMALIZACIÓN DE TEMPORADAS\n" + "=" * 80)

temporadas_prueba = ["00/01", "01/02", "18/19", "25/26", "97/98", "98/99", "99/00", "2024", "2025", "2026", "1879/80", "1889/90"]

for temporada in temporadas_prueba:
    print(f"{temporada:>10} -> {normalizar_temporada(temporada)}")

print("\n" + "=" * 80 + "\nNORMALIZACIÓN DE TODAS LAS TEMPORADAS\n" + "=" * 80)

for nombre, df in dataframes.items():
    print(f"\n" + "-" * 80 + f"\n{nombre}\n" + "-" * 80)
    if "temporada" not in df.columns:
        print("[AVISO] No existe la columna 'temporada'.")
        continue

    temporadas_antes = df["temporada"].dropna().astype(str).str.strip().unique()
    df["temporada"] = df["temporada"].apply(normalizar_temporada)
    temporadas_despues = df["temporada"].dropna().astype(str).str.strip().unique()
    dataframes[nombre] = df

    print(f"Temporadas únicas antes: {len(temporadas_antes)}\nTemporadas únicas después: {len(temporadas_despues)}\n\nConversiones realizadas:")

    conversiones = {antes: normalizar_temporada(antes) for antes in sorted(temporadas_antes) if antes != normalizar_temporada(antes)}

    if conversiones:
        for antes, despues in conversiones.items(): print(f"{antes:>10} -> {despues}")
    else: print("Ninguna.")


# ===============================================================
# NUEVA FUNCIÓN: ELIMINAR COMPETICIONES ESPECÍFICAS
# ===============================================================

def eliminar_competiciones_especificas():
    """
    Elimina:
    - De stats_historico_ballon_dor: filas de la temporada 2022/23 con MLS, Leagues Cup y US Open Cup.
    - De nominados_ballon_dor: filas con la competición Leagues Cup.
    """
    print("\n" + "=" * 80 + "\nELIMINACIÓN DE COMPETICIONES ESPECÍFICAS (MLS / LEAGUES CUP / US OPEN CUP)\n" + "=" * 80)

    # 1. stats_historico_ballon_dor
    df_stats = dataframes["stats_historico_ballon_dor"]
    filas_antes_stats = len(df_stats)
    comps_excluir_2022_23 = ["MLS", "Leagues Cup", "US Open Cup"]
    mascara_stats = (
        df_stats["temporada"].astype("string").str.strip().eq("2022/23") &
        df_stats["competicion"].astype("string").str.strip().isin(comps_excluir_2022_23)
    )
    df_stats = df_stats[~mascara_stats].copy()
    dataframes["stats_historico_ballon_dor"] = df_stats

    print(f"stats_historico_ballon_dor: {filas_antes_stats - len(df_stats)} fila(s) eliminada(s) (2022/23 - MLS, Leagues Cup, US Open Cup)")

    # 2. nominados_ballon_dor
    df_nom = dataframes["nominados_ballon_dor"]
    filas_antes_nom = len(df_nom)
    mascara_nom = df_nom["competicion"].astype("string").str.strip().eq("Leagues Cup")
    df_nom = df_nom[~mascara_nom].copy()
    dataframes["nominados_ballon_dor"] = df_nom

    print(f"nominados_ballon_dor: {filas_antes_nom - len(df_nom)} fila(s) eliminada(s) (Leagues Cup)")

# Ejecutar la filtración de competiciones específicas tras normalizar temporadas y ligas
eliminar_competiciones_especificas()


# ===============================================================
# 17.3. ELIMINAR FIFA WORLD CUP 2022 DE 2021/22 Y PREPARAR
# ===============================================================

print("\n" + "=" * 80 + "\nELIMINACIÓN FIFA WORLD CUP 2022 DE 2021/22\n" + "=" * 80)

for nombre in ["stats_historico_ballon_dor", "nominados_ballon_dor"]:
    df = dataframes[nombre]
    mascara_eliminar = (
        df["temporada"].astype("string").str.strip().eq("2021/22") &
        df["competicion"].astype("string").str.strip().eq("FIFA World Cup")
    )
    filas_eliminadas = int(mascara_eliminar.sum())
    df = df[~mascara_eliminar].copy()
    dataframes[nombre] = df
    print(f"{nombre}: {filas_eliminadas} fila(s) de FIFA World Cup eliminada(s) de 2021/22")

print("\n" + "=" * 80 + "\nPREPARACIÓN TEMPORAL DE FIFA WORLD CUP 2022\n" + "=" * 80)

for nombre in ["stats_historico_ballon_dor", "nominados_ballon_dor"]:
    df = dataframes[nombre]
    mascara_mundial_2022 = (
        df["temporada"].astype("string").str.strip().eq("2022/23") &
        df["competicion"].astype("string").str.strip().eq("FIFA World Cup")
    )
    filas_modificadas = int(mascara_mundial_2022.sum())
    df.loc[mascara_mundial_2022, "temporada"] = "2021/22"
    dataframes[nombre] = df
    print(f"{nombre}: {filas_modificadas} fila(s) cambiada(s) temporalmente de 2022/23 -> 2021/22")

# COMPROBACIÓN FIFA WORLD CUP 2022
print("\n" + "=" * 80 + "\nCOMPROBACIÓN FIFA WORLD CUP 2022\n" + "=" * 80)

for nombre in ["stats_historico_ballon_dor", "nominados_ballon_dor"]:
    df = dataframes[nombre]
    encontradas = df[df["temporada"].astype("string").str.strip().eq("2021/22") & df["competicion"].astype("string").str.strip().eq("FIFA World Cup")]

    if len(encontradas) > 0: print(f"[OK] {nombre}: {len(encontradas)} fila(s) de FIFA World Cup temporalmente preparadas en 2021/22.")
    else: print(f"[AVISO] {nombre}: no se encontraron filas FIFA World Cup en 2021/22.")


# ===============================================================
# 18. COMPROBAR FORMATO DE TEMPORADAS
# ===============================================================

print("\n" + "=" * 80 + "\nCOMPROBACIÓN FINAL DE TEMPORADAS\n" + "=" * 80)

for nombre, df in dataframes.items():
    print(f"\n" + "-" * 80 + f"\n{nombre}\n" + "-" * 80)

    if "temporada" not in df.columns:
        print("[AVISO] No existe la columna 'temporada'.")
        continue

    temporadas = sorted(df["temporada"].dropna().astype(str).str.strip().unique())
    formatos_incorrectos = [t for t in temporadas if not re.fullmatch(r"\d{4}/\d{2}", t)]

    if formatos_incorrectos:
        print("[ERROR] Se encontraron temporadas con formato incorrecto:")
        for t in formatos_incorrectos: print(f"    {t}")
    else: print("[OK] Todas las temporadas tienen el formato YYYY/YY.")
    print(f"Total de temporadas únicas: {len(temporadas)}")


# ===============================================================
# 19. MOSTRAR TODAS LAS TEMPORADAS FINALES
# ===============================================================

print("\n" + "=" * 80 + "\nTODAS LAS TEMPORADAS DESPUÉS DE NORMALIZAR\n" + "=" * 80)

for nombre, df in dataframes.items():
    print(f"\n" + "=" * 80 + f"\n{nombre}\n" + "=" * 80)
    if "temporada" not in df.columns:
        print("[AVISO] No existe la columna 'temporada'.")
        continue
    temporadas = sorted(df["temporada"].dropna().astype(str).str.strip().unique())
    print(f"Total: {len(temporadas)}")
    for i, t in enumerate(temporadas, start=1): print(f"{i:4}. {t}")


# ===============================================================
# 20. MOSTRAR COMPETICIONES Y LIGAS
# ===============================================================

print("\n" + "=" * 80 + "\nCOMPETICIONES Y LIGAS DESPUÉS DE LA NORMALIZACIÓN\n" + "=" * 80)

for nombre, columna in columnas_competicion.items():
    df = dataframes[nombre]
    valores = sorted(df[columna].dropna().astype(str).str.strip().unique())
    print(f"\n" + "=" * 80 + f"\n{nombre}\n" + "=" * 80 + f"\nColumna: {columna}\nTotal de nombres únicos: {len(valores)}")
    for i, valor in enumerate(valores, start=1): print(f"{i:4}. {valor}")


# ===============================================================
# 21. COMPROBAR COMPETICIONES COMUNES
# ===============================================================

print("\n" + "=" * 80 + "\nCOMPETICIONES COMUNES ENTRE DATASETS\n" + "=" * 80)

def obtener_competiciones(df):
    return set(df["competicion"].dropna().astype(str).str.strip().unique())

comp_stats = obtener_competiciones(dataframes["stats_historico_ballon_dor"])
comp_nominados = obtener_competiciones(dataframes["nominados_ballon_dor"])
comp_clubes = obtener_competiciones(dataframes["clubes_competiciones"])
comp_selecciones = obtener_competiciones(dataframes["selecciones_competiciones"])

for titulo, set1, set2 in [
    ("stats_historico_ballon_dor ∩ clubes_competiciones", comp_stats, comp_clubes),
    ("stats_historico_ballon_dor ∩ selecciones_competiciones", comp_stats, comp_selecciones),
    ("nominados_ballon_dor ∩ clubes_competiciones", comp_nominados, comp_clubes),
    ("nominados_ballon_dor ∩ selecciones_competiciones", comp_nominados, comp_selecciones)
]:
    print(f"\n" + "-" * 80 + f"\n{titulo}\n" + "-" * 80)
    comunes = sorted(set1 & set2)
    print(f"Total: {len(comunes)}")
    for i, comp in enumerate(comunes, start=1): print(f"{i:4}. {comp}")


# ===============================================================
# NUEVA SECCIÓN: ACTUALIZAR CLUBES ESPECÍFICOS Y MOVER FIFA INTERCONTINENTAL CUP A 2025/26
# ===============================================================

print("\n" + "=" * 80 + "\nACTUALIZACIÓN DE CLUBES ESPECÍFICOS Y MOVIMIENTO DE FIFA INTERCONTINENTAL CUP\n" + "=" * 80)

cambios_clubes_jugadores = [
    ("Thierry Henry", "2006/07", "Arsenal"),
    ("Cristiano Ronaldo", "2008/09", "Manchester United"),
    ("Samuel Eto'o", "2008/09", "Barcelona"),
    ("Sergio Agüero", "2010/11", "Atletico Madrid"),
    ("Radamel Falcao", "2011/12", "Atletico Madrid"),
    ("Zlatan Ibrahimović", "2011/12", "AC Milan"),
    ("Mesut Özil", "2011/12", "Real Madrid"),
    ("Gareth Bale", "2012/13", "Tottenham"),
    ("Luis Suárez", "2012/13", "Liverpool"),
    ("Neymar", "2016/17", "Barcelona"),
    ("Cristiano Ronaldo", "2017/18", "Real Madrid"),
    ("Thibaut Courtois", "2017/18", "Chelsea"),
    ("Eden Hazard", "2018/19", "Chelsea"),
    ("Matthijs de Ligt", "2018/19", "Ajax"),
    ("Lionel Messi", "2020/21", "Barcelona"),
    ("Cristiano Ronaldo", "2020/21", "Juventus"),
    ("Gianluigi Donnarumma", "2020/21", "AC Milan"),
    ("Romelu Lukaku", "2020/21", "Inter Milan"),
    ("Sébastien Haller", "2021/22", "Ajax"),
    ("Dušan Vlahović", "2021/22", "Juventus"),
    ("Luis Díaz", "2021/22", "Liverpool"),
    ("Gianluigi Donnarumma", "2024/25", "Paris Saint-Germain"),
    ("Viktor Gyökeres", "2024/25", "Sporting CP"),
    ("Florian Wirtz", "2024/25", "Bayer Leverkusen"),
]

for nombre_df in ["stats_historico_ballon_dor", "nominados_ballon_dor"]:
    if nombre_df in dataframes:
        df = dataframes[nombre_df]
        if "nombre" in df.columns and "temporada" in df.columns and "club" in df.columns:
            modificaciones_club = 0
            for jugador, temporada, nuevo_club in cambios_clubes_jugadores:
                mascara = (
                    df["nombre"].astype(str).str.strip().eq(jugador) &
                    df["temporada"].astype(str).str.strip().eq(temporada)
                )
                cant = mascara.sum()
                if cant > 0:
                    df.loc[mascara, "club"] = nuevo_club
                    modificaciones_club += cant
            dataframes[nombre_df] = df
            print(f"{nombre_df}: {modificaciones_club} registro(s) con club actualizado.")

# Mover la FIFA Intercontinental Cup directamente de 2024/25 a 2025/26
for nombre_df in ["clubes_competiciones", "stats_historico_ballon_dor", "nominados_ballon_dor"]:
    if nombre_df in dataframes:
        df = dataframes[nombre_df]
        if "competicion" in df.columns and "temporada" in df.columns:
            mascara = (
                df["competicion"].astype(str).str.strip().eq("FIFA Intercontinental Cup") &
                df["temporada"].astype(str).str.strip().eq("2024/25")
            )
            cant = mascara.sum()
            if cant > 0:
                df.loc[mascara, "temporada"] = "2025/26"
                dataframes[nombre_df] = df
                print(f"{nombre_df}: {cant} registro(s) de FIFA Intercontinental Cup cambiados de 2024/25 a 2025/26.")


# ===============================================================
# 22. UNIR DATOS DE CLUBES Y SELECCIONES
# ===============================================================

print("\n" + "=" * 80 + "\nCARGANDO DATASETS PARA LA UNIÓN\n" + "=" * 80)

stats_historico_ballon_dor = dataframes["stats_historico_ballon_dor"].copy()
nominados_ballon_dor = dataframes["nominados_ballon_dor"].copy()
clubes_competiciones = dataframes["clubes_competiciones"].copy()
clubes_ligas = dataframes["clubes_ligas"].copy()  # <--- CORREGIDO AQUÍ (clubes_ligas en lugar de clubes_liga)
selecciones_competiciones = dataframes["selecciones_competiciones"].copy()

print(f"\nstats_historico_ballon_dor: {len(stats_historico_ballon_dor)} filas\nnominados_ballon_dor: {len(nominados_ballon_dor)} filas")
print(f"clubes_competiciones: {len(clubes_competiciones)} filas\nclubes_ligas: {len(clubes_ligas)} filas\nselecciones_competiciones: {len(selecciones_competiciones)} filas")

# LIMPIAR CLAVES DE UNIÓN
print("\n" + "=" * 80 + "\nLIMPIANDO CLAVES DE UNIÓN\n" + "=" * 80)

def limpiar_claves(df, columnas):
    for col in columnas:
        if col in df.columns: df[col] = df[col].astype("string").str.strip()
    return df

stats_historico_ballon_dor = limpiar_claves(stats_historico_ballon_dor, ["temporada", "club", "nacionalidad", "competicion"])
nominados_ballon_dor = limpiar_claves(nominados_ballon_dor, ["temporada", "club", "nacionalidad", "competicion"])
clubes_competiciones = limpiar_claves(clubes_competiciones, ["temporada", "club", "competicion", "ronda"])
clubes_ligas = limpiar_claves(clubes_ligas, ["temporada", "club", "liga"])
selecciones_competiciones = limpiar_claves(selecciones_competiciones, ["temporada", "club", "competicion", "ronda"])

# COMPLETAR GOALSASSISTSSUM
def completar_goals_assists_sum(df, nombre_dataset):
    if not all(col in df.columns for col in ["goals", "assists", "goalsAssistsSum"]): return df
    mascara = df["goalsAssistsSum"].isna() & (df["goals"].notna() | df["assists"].notna())
    df.loc[mascara, "goalsAssistsSum"] = df.loc[mascara, "goals"].fillna(0) + df.loc[mascara, "assists"].fillna(0)
    print(f"{nombre_dataset}: {int(mascara.sum())} fila(s) con goalsAssistsSum completado")
    return df

stats_historico_ballon_dor = completar_goals_assists_sum(stats_historico_ballon_dor, "stats_historico_ballon_dor")
nominados_ballon_dor = completar_goals_assists_sum(nominados_ballon_dor, "nominados_ballon_dor")

# COMBINAR CENTROS EN PASES
print("\n" + "=" * 80 + "\nCOMBINANDO CENTROS EN PASES\n" + "=" * 80)

def combinar_pases_y_centros(df, nombre_dataset):
    columnas_req = ["accuratePasses", "accuratePassesPercentage", "accurateCrosses", "accurateCrossesPercentage"]
    if not all(col in df.columns for col in columnas_req):
        print(f"[AVISO] {nombre_dataset}: no contiene todas las columnas requeridas para combinar pases y centros.")
        return df

    p_acc = pd.to_numeric(df["accuratePasses"], errors="coerce")
    p_pct = pd.to_numeric(df["accuratePassesPercentage"], errors="coerce")
    c_acc = pd.to_numeric(df["accurateCrosses"], errors="coerce")
    c_pct = pd.to_numeric(df["accurateCrossesPercentage"], errors="coerce")

    p_tot = np.where((p_pct > 0) & p_acc.notna(), (p_acc / (p_pct / 100.0)), np.where((p_pct == 0) & (p_acc == 0), 0.0, np.nan))
    c_tot = np.where((c_pct > 0) & c_acc.notna(), (c_acc / (c_pct / 100.0)), np.where((c_pct == 0) & (c_acc == 0), 0.0, np.nan))

    p_acc_num = np.nan_to_num(p_acc, nan=0.0)
    c_acc_num = np.nan_to_num(c_acc, nan=0.0)
    p_tot_num = np.nan_to_num(p_tot, nan=0.0)
    c_tot_num = np.nan_to_num(c_tot, nan=0.0)

    nuevos_p_acc = p_acc_num + c_acc_num
    nuevos_p_tot = p_tot_num + c_tot_num

    nuevos_p_pct = np.where(nuevos_p_tot > 0, (nuevos_p_acc / nuevos_p_tot) * 100.0, np.nan)

    mascara = p_acc.notna() | c_acc.notna()

    df.loc[mascara, "accuratePasses"] = np.round(nuevos_p_acc[mascara], 2)
    df.loc[mascara, "accuratePassesPercentage"] = np.round(nuevos_p_pct[mascara], 2)

    print(f"{nombre_dataset}: {int(mascara.sum())} fila(s) actualizadas combinando centros ('accurateCrosses') en pases ('accuratePasses')")
    return df

stats_historico_ballon_dor = combinar_pases_y_centros(stats_historico_ballon_dor, "stats_historico_ballon_dor")
nominados_ballon_dor = combinar_pases_y_centros(nominados_ballon_dor, "nominados_ballon_dor")


# ===============================================================
# ACTUALIZANDO POSICIONES DE JUGADORES ESPECÍFICOS
# ===============================================================

print("\n" + "=" * 80 + "\nACTUALIZANDO POSICIONES DE JUGADORES ESPECÍFICOS\n" + "=" * 80)

CAMBIOS_POSICION = {
    "Sadio Mané": "Forward",
    "Lamine Yamal": "Forward",
    "Luis Díaz": "Forward",
    "Denzel Dumfries": "Defender",
    "Raphinha": "Forward",
    "Mohamed Salah": "Forward",
    "Alejandro Grimaldo": "Defender",
    "Nico Williams": "Forward",
    "Riyad Mahrez": "Forward",
    "Arjen Robben": "Forward",
    "Dani Alves": "Defender",
    "Franck Ribéry": "Forward",
    "Michael Olise": "Forward"
}

def actualizar_posiciones_jugadores(df, nombre_dataset):
    if "nombre" not in df.columns or "posicion" not in df.columns:
        print(f"[AVISO] {nombre_dataset}: faltan las columnas 'nombre' o 'posicion'.")
        return df

    modificaciones = 0
    for jugador, nueva_posicion in CAMBIOS_POSICION.items():
        mascara = df["nombre"].astype("string").str.strip().eq(jugador)
        cant = mascara.sum()
        if cant > 0:
            df.loc[mascara, "posicion"] = nueva_posicion
            modificaciones += cant

    print(f"{nombre_dataset}: {modificaciones} fila(s) actualizadas con nuevas posiciones.")
    return df

stats_historico_ballon_dor = actualizar_posiciones_jugadores(stats_historico_ballon_dor, "stats_historico_ballon_dor")
nominados_ballon_dor = actualizar_posiciones_jugadores(nominados_ballon_dor, "nominados_ballon_dor")


# LIMPIAR ESTADÍSTICAS SEGÚN POSICIÓN
print("\n" + "=" * 80 + "\nLIMPIANDO ESTADÍSTICAS SEGÚN LA POSICIÓN DEL JUGADOR\n" + "=" * 80)

estadisticas_solo_porteros = [
    "saves", "savesCaught", "savesParried", "savedShotsFromInsideTheBox", "savedShotsFromOutsideTheBox",
    "penaltySave", "goalsPrevented", "goalsConcededInsideTheBox", "goalsConcededOutsideTheBox",
    "highClaims", "punches", "runsOut", "successfulRunsOut", "crossesNotClaimed", "goalKicks", "penaltyFaced"
]

def limpiar_estadisticas_porteros(df, nombre_dataset):
    if "posicion" not in df.columns:
        print(f"[AVISO] {nombre_dataset}: no existe la columna 'posicion'.")
        return df
    
    posicion_normalizada = df["posicion"].astype("string").str.strip().str.casefold()
    mascara_no_portero = posicion_normalizada != "goalkeeper"
    mascara_no_defensivo = ~posicion_normalizada.isin(["goalkeeper", "defender"])

    print(f"\n{nombre_dataset}:")

    # 1. Limpieza de estadísticas exclusivas de porteros
    for est in estadisticas_solo_porteros:
        if est not in df.columns: continue
        antes = df[est].notna().sum()
        df.loc[mascara_no_portero, est] = pd.NA
        despues = df[est].notna().sum()
        if df.loc[mascara_no_portero, est].notna().sum() > 0:
            raise ValueError(f"{nombre_dataset}: todavía existen valores de {est} en no porteros.")
        print(f"    {est}: {antes:,} -> {despues:,} ({antes - despues:,} eliminados)")

    # 2. Limpieza de cleanSheet y goalsConceded (se conservan para Goalkeeper y Defender)
    for col in ["cleanSheet", "goalsConceded"]:
        if col in df.columns:
            antes = df[col].notna().sum()
            df.loc[mascara_no_defensivo, col] = pd.NA
            despues = df[col].notna().sum()
            if df.loc[mascara_no_defensivo, col].notna().sum() > 0:
                raise ValueError(f"{nombre_dataset}: todavía existen valores de {col} fuera de Goalkeeper y Defender.")
            print(f"    {col}: {antes:,} -> {despues:,} ({antes - despues:,} eliminados)")

    print("[OK] Las estadísticas defensivas y de portero se limpiaron correctamente según la posición.")
    return df

stats_historico_ballon_dor = limpiar_estadisticas_porteros(stats_historico_ballon_dor, "stats_historico_ballon_dor")
nominados_ballon_dor = limpiar_estadisticas_porteros(nominados_ballon_dor, "nominados_ballon_dor")

# PREPARAR CLAVE DE LIGA Y ELIMINAR DUPLICADOS EXACTOS
clubes_ligas["competicion"] = clubes_ligas["liga"]

print("\n" + "=" * 80 + "\nELIMINANDO DUPLICADOS EXACTOS DE TABLAS DE REFERENCIA\n" + "=" * 80)

for df_ref, cols, nombre in [
    (clubes_competiciones, ["temporada", "club", "competicion", "ronda"], "clubes_competiciones"),
    (clubes_ligas, ["temporada", "club", "liga", "posicion"], "clubes_ligas"),
    (selecciones_competiciones, ["temporada", "club", "competicion", "ronda"], "selecciones_competiciones")
]:
    antes = len(df_ref)
    df_ref.drop_duplicates(subset=cols, inplace=True)
    print(f"{nombre}: {antes - len(df_ref)} duplicado(s) exacto(s) eliminado(s)")

# COMPROBACIONES DE CONFLICTOS Y UNICIDAD
print("\n" + "=" * 80 + "\nCOMPROBACIÓN DE CLAVES\n" + "=" * 80)

for df_ref, clave, col_val, nombre in [
    (clubes_competiciones, ["temporada", "club", "competicion"], "ronda", "clubes_competiciones"),
    (clubes_ligas, ["temporada", "club", "competicion"], "posicion", "clubes_ligas"),
    (selecciones_competiciones, ["temporada", "club", "competicion"], "ronda", "selecciones_competiciones")
]:
    conflictos = df_ref.groupby(clave, dropna=False)[col_val].nunique(dropna=False)
    if (conflictos > 1).any(): raise ValueError(f"{nombre} contiene claves duplicadas con diferentes valores.")
    if df_ref.duplicated(subset=clave).any(): raise ValueError(f"La clave de {nombre} sigue teniendo duplicados.")
    print(f"[OK] {nombre}: clave única y sin conflictos de {col_val}.")

# TABLAS DE REFERENCIA PARA EL MERGE
clubes_competiciones_ref = clubes_competiciones[["temporada", "club", "competicion", "ronda"]].copy().rename(columns={"ronda": "resultado_club"})
clubes_competiciones_ref["resultado_club"] = clubes_competiciones_ref["resultado_club"].astype("string").str.strip()

clubes_ligas_ref = clubes_ligas[["temporada", "club", "competicion", "posicion"]].copy().rename(columns={"posicion": "resultado_liga"})
clubes_ligas_ref["resultado_liga"] = clubes_ligas_ref["resultado_liga"].astype("string").str.strip()

selecciones_competiciones_ref = selecciones_competiciones[["temporada", "club", "competicion", "ronda"]].copy().rename(columns={"club": "nacionalidad", "ronda": "resultado_seleccion"})
selecciones_competiciones_ref["resultado_seleccion"] = selecciones_competiciones_ref["resultado_seleccion"].astype("string").str.strip()

# FUNCIÓN PARA ENRIQUECER DATASETS
def enriquecer_dataset(df, nombre_dataset):
    print("\n" + "#" * 80 + f"\nENRIQUECIENDO: {nombre_dataset}\n" + "#" * 80)
    filas_orig, cols_orig = len(df), list(df.columns)

    if "resultado_competicion" in df.columns: df = df.drop(columns=["resultado_competicion"])
    cols_antiguas = [c for c in ["ronda", "posicion_liga"] if c in df.columns]
    if cols_antiguas: df = df.drop(columns=cols_antiguas)

    print("\n[1/3] Buscando resultados de competiciones de clubes...")
    df = df.merge(clubes_competiciones_ref, how="left", on=["temporada", "club", "competicion"], validate="many_to_one")

    print("[2/3] Buscando posiciones de liga...")
    df = df.merge(clubes_ligas_ref, how="left", on=["temporada", "club", "competicion"], validate="many_to_one")

    print("[3/3] Buscando resultados de selecciones...")
    df = df.merge(selecciones_competiciones_ref, how="left", on=["temporada", "nacionalidad", "competicion"], validate="many_to_one")

    df["resultado_competicion"] = df["resultado_club"].combine_first(df["resultado_liga"]).combine_first(df["resultado_seleccion"]).astype("string").str.strip()
    df = df.drop(columns=["resultado_club", "resultado_liga", "resultado_seleccion"])

    if len(df) != filas_orig: raise ValueError(f"La unión de {nombre_dataset} cambió el número de filas.")
    if any(c not in df.columns for c in cols_orig if c not in cols_antiguas): raise ValueError(f"Se perdieron columnas en {nombre_dataset}.")

    print(f"\n[OK] Filas totales: {len(df)} | Con resultado: {df['resultado_competicion'].notna().sum()} | Sin resultado: {df['resultado_competicion'].isna().sum()}")
    print("\nValores de resultado_competicion:\n" + df["resultado_competicion"].value_counts(dropna=False).head(30).to_string())
    return df

# ENRIQUECER DATASETS
stats_historico_ballon_dor = enriquecer_dataset(stats_historico_ballon_dor, "stats_historico_ballon_dor")
nominados_ballon_dor = enriquecer_dataset(nominados_ballon_dor, "nominados_ballon_dor")

# RESTAURAR FIFA WORLD CUP 2022 A 2022/23
print("\n" + "=" * 80 + "\nRESTAURANDO FIFA WORLD CUP 2022 A LA TEMPORADA 2022/23\n" + "=" * 80)

for nombre, df in [("stats_historico_ballon_dor", stats_historico_ballon_dor), ("nominados_ballon_dor", nominados_ballon_dor)]:
    mascara_mundial_2022 = df["temporada"].astype("string").str.strip().eq("2021/22") & df["competicion"].astype("string").str.strip().eq("FIFA World Cup")
    df.loc[mascara_mundial_2022, "temporada"] = "2022/23"
    print(f"{nombre}: {int(mascara_mundial_2022.sum())} fila(s) restaurada(s) de 2021/22 -> 2022/23")


# ===============================================================
# MOVER FIFA INTERCONTINENTAL CUP A NOMINADOS (FILTRANDO NO NOMINADOS)
# ===============================================================

print("\n" + "=" * 80 + "\nMOVIENDO FIFA INTERCONTINENTAL CUP A NOMINADOS (FILTRANDO NO NOMINADOS)\n" + "=" * 80)

# Obtener los nombres de los jugadores que SÍ están nominados en 2025/26
nombres_nominados_2025_26 = set(
    nominados_ballon_dor[
        nominados_ballon_dor["temporada"].astype(str).str.strip().eq("2025/26")
    ]["nombre"].astype(str).str.strip()
)

# Localizar filas de FIFA Intercontinental Cup en 2025/26 dentro de stats_historico_ballon_dor
mascara_inter = (
    stats_historico_ballon_dor["competicion"].astype(str).str.strip().eq("FIFA Intercontinental Cup") &
    stats_historico_ballon_dor["temporada"].astype(str).str.strip().eq("2025/26")
)

filas_inter = stats_historico_ballon_dor[mascara_inter].copy()

if len(filas_inter) > 0:
    # Filtrar solo aquellos jugadores que realmente están nominados en 2025/26
    mascara_validos = filas_inter["nombre"].astype(str).str.strip().isin(nombres_nominados_2025_26)
    
    filas_a_mover = filas_inter[mascara_validos].copy()
    filas_a_eliminar = filas_inter[~mascara_validos].copy()

    if len(filas_a_mover) > 0:
        if "posicion_balon_oro" in filas_a_mover.columns:
            filas_a_mover["posicion_balon_oro"] = 0
        nominados_ballon_dor = pd.concat([nominados_ballon_dor, filas_a_mover], ignore_index=True)
        print(f"[OK] Se movieron {len(filas_a_mover)} registro(s) de FIFA Intercontinental Cup a nominados_ballon_dor (con posicion_balon_oro = 0).")

    if len(filas_a_eliminar) > 0:
        print(f"[OK] Se eliminaron {len(filas_a_eliminar)} registro(s) de jugadores no nominados (ej. Désiré Doué) para que no se cuelen:")
        for nom in filas_a_eliminar["nombre"].unique():
            print(f"    - {nom}")

    # Retirar todas las filas de intercontinental cup de stats_historico_ballon_dor
    stats_historico_ballon_dor = stats_historico_ballon_dor[~mascara_inter].reset_index(drop=True)
else:
    print("[AVISO] No se encontraron registros de FIFA Intercontinental Cup en 2025/26 dentro de stats_historico_ballon_dor.")


# ORDENAR DATASETS FINALES
def ordenar_por_temporada_y_ranking(df, nombre_dataset):
    df = df.copy()
    df["__orden_temporada"] = pd.to_numeric(df["temporada"].astype("string").str.extract(r"^(\d{4})", expand=False), errors="coerce")
    df["__orden_balon_dor"] = pd.to_numeric(df["posicion_balon_oro"], errors="coerce")
    df = df.sort_values(["__orden_temporada", "__orden_balon_dor"], ascending=[True, True], kind="stable").drop(columns=["__orden_temporada", "__orden_balon_dor"]).reset_index(drop=True)
    print(f"[OK] {nombre_dataset}: ordenado por temporada y posición del Balón de Oro.")
    return df

stats_historico_ballon_dor = ordenar_por_temporada_y_ranking(stats_historico_ballon_dor, "stats_historico_ballon_dor")
nominados_ballon_dor = ordenar_por_temporada_y_ranking(nominados_ballon_dor, "nominados_ballon_dor")

# GUARDAR RESULTADOS ÚNICAMENTE EN LOS ARCHIVOS LIMPIOS
print("\n" + "=" * 80 + "\nGUARDANDO DATASETS ACTUALIZADOS\n" + "=" * 80)

stats_historico_ballon_dor.to_csv(archivos["stats_historico_ballon_dor_limpio"], index=False, encoding="utf-8")
nominados_ballon_dor.to_csv(archivos["nominados_ballon_dor_limpio"], index=False, encoding="utf-8")

print(f"\n[OK] {archivos['stats_historico_ballon_dor_limpio']}\n[OK] {archivos['nominados_ballon_dor_limpio']}")

# COMPROBACIÓN FINAL
print("\n" + "=" * 80 + "\nCOMPROBACIÓN FINAL\n" + "=" * 80)

for nombre, df in [("STATS HISTÓRICOS", stats_historico_ballon_dor), ("NOMINADOS", nominados_ballon_dor)]:
    print(f"\n{nombre}:\nFilas: {len(df)} | Columnas: {len(df.columns)}")
    print(f"Últimas columnas: {df.columns[-5:].tolist()}")
    print(f"    resultado_competicion: {'OK' if 'resultado_competicion' in df.columns else 'ERROR'}")

# COMPROBAR ARCHIVOS GENERADOS
print("\n" + "=" * 80 + "\nARCHIVOS GENERADOS\n" + "=" * 80)

for nombre in ["stats_historico_ballon_dor_limpio", "nominados_ballon_dor_limpio"]:
    archivo = archivos[nombre]
    if archivo.exists(): print(f"[OK] {archivo} ({archivo.stat().st_size / (1024 * 1024):.2f} MB)")
    else: print(f"[ERROR] No se ha encontrado: {archivo}")

print("\n" + "=" * 80 + "\nLIMPIEZA, NORMALIZACIÓN Y UNIÓN TERMINADAS\n" + "=" * 80)