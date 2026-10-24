import pandas as pd
from pathlib import Path


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

DATA_DIR = Path("data")

archivos = {
    "historico": DATA_DIR / "stats_historico_ballon_dor_limpio.csv",
    "nominados": DATA_DIR / "nominados_ballon_dor_limpio.csv",
}


# ============================================================
# 2. CARGAR DATASETS
# ============================================================

print("=" * 70)
print("PREPARACIÓN DE DATOS PARA MACHINE LEARNING")
print("=" * 70)

historico = pd.read_csv(archivos["historico"])
nominados = pd.read_csv(archivos["nominados"])

print("\nDATASETS CARGADOS")
print("-" * 70)

print(f"Histórico:  {historico.shape[0]} filas × {historico.shape[1]} columnas")
print(f"Nominados:  {nominados.shape[0]} filas × {nominados.shape[1]} columnas")


# ============================================================
# 3. INFORMACIÓN GENERAL
# ============================================================

print("\n" + "=" * 70)
print("3. INFORMACIÓN GENERAL")
print("=" * 70)

print("\nColumnas histórico:")
print(list(historico.columns))

print("\nColumnas nominados:")
print(list(nominados.columns))


# ============================================================
# 4. COMPROBAR COLUMNAS
# ============================================================

print("\n" + "=" * 70)
print("4. COMPROBACIÓN DE COLUMNAS")
print("=" * 70)

columnas_historico = set(historico.columns)
columnas_nominados = set(nominados.columns)

solo_historico = columnas_historico - columnas_nominados
solo_nominados = columnas_nominados - columnas_historico

print("\nColumnas solamente en histórico:")
print(sorted(solo_historico))

print("\nColumnas solamente en nominados:")
print(sorted(solo_nominados))

columnas_comunes = columnas_historico & columnas_nominados

print(f"\nColumnas comunes: {len(columnas_comunes)}")


# ============================================================
# 5. COMBINAR LOS DATASETS
# ============================================================
#
# No estamos agrupando ni eliminando competiciones.
#
# Cada fila sigue siendo:
#
# jugador + temporada + competición
#
# Añadimos una columna indicando si la fila pertenece al
# histórico o a los nominados de 2026.
# ============================================================

historico["dataset"] = "historico"
nominados["dataset"] = "nominados_2026"

df = pd.concat(
    [historico, nominados],
    ignore_index=True,
    sort=False
)

print("\n" + "=" * 70)
print("5. DATASET COMPLETO")
print("=" * 70)

print(f"\nFilas totales:    {len(df)}")
print(f"Columnas totales: {len(df.columns)}")

print("\nFilas por dataset:")
print(df["dataset"].value_counts())


# ============================================================
# 6. COMPROBAR TEMPORADAS
# ============================================================

print("\n" + "=" * 70)
print("6. TEMPORADAS")
print("=" * 70)

print("\nTemporadas del histórico:")
print(sorted(historico["temporada"].dropna().unique()))

print("\nTemporadas de nominados:")
print(sorted(nominados["temporada"].dropna().unique()))


# ============================================================
# 7. COMPROBAR POSICIÓN DEL BALÓN DE ORO
# ============================================================

print("\n" + "=" * 70)
print("7. POSICIÓN DEL BALÓN DE ORO")
print("=" * 70)

print("\nHistórico:")
print(
    historico["posicion_balon_oro"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nNominados 2026:")
print(
    nominados["posicion_balon_oro"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nIMPORTANTE:")
print("El valor 0 de los nominados significa que todavía no conocemos")
print("la posición real del Balón de Oro 2026.")
print("NO se utilizará como objetivo durante el entrenamiento.")


# ============================================================
# 8. COMPROBAR JUGADORES Y COMPETICIONES
# ============================================================

print("\n" + "=" * 70)
print("8. JUGADORES Y COMPETICIONES")
print("=" * 70)

print(f"\nJugadores únicos histórico: {historico['nombre'].nunique()}")
print(f"Jugadores únicos 2026:     {nominados['nombre'].nunique()}")

print(
    f"\nCompeticiones únicas histórico: "
    f"{historico['competicion'].nunique()}"
)

print(
    f"Competiciones únicas 2026:     "
    f"{nominados['competicion'].nunique()}"
)

print("\nCompeticiones del histórico:")
for competicion in sorted(historico["competicion"].dropna().unique()):
    print(f"  - {competicion}")

print("\nCompeticiones de 2026:")
for competicion in sorted(nominados["competicion"].dropna().unique()):
    print(f"  - {competicion}")


# ============================================================
# 9. NÚMERO DE COMPETICIONES POR JUGADOR Y TEMPORADA
# ============================================================

print("\n" + "=" * 70)
print("9. COMPETICIONES POR JUGADOR Y TEMPORADA")
print("=" * 70)

competiciones_por_jugador = (
    df.groupby(
        ["temporada", "nombre"],
        dropna=False
    )["competicion"]
    .nunique()
    .reset_index(name="numero_competiciones")
)

print("\nPrimeras filas:")
print(competiciones_por_jugador.head(20).to_string(index=False))

print("\nDistribución del número de competiciones:")
print(
    competiciones_por_jugador["numero_competiciones"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. COMPROBAR FILAS DUPLICADAS
# ============================================================

print("\n" + "=" * 70)
print("10. DUPLICADOS")
print("=" * 70)

columnas_clave = [
    "temporada",
    "nombre",
    "competicion"
]

duplicados = df.duplicated(
    subset=columnas_clave,
    keep=False
)

print(
    f"\nFilas duplicadas según "
    f"{columnas_clave}: {duplicados.sum()}"
)

if duplicados.any():

    print("\nEjemplos de duplicados:")

    print(
        df.loc[
            duplicados,
            columnas_clave
        ]
        .sort_values(columnas_clave)
        .head(30)
        .to_string(index=False)
    )

else:
    print("\nNo hay duplicados en jugador + temporada + competición.")


# ============================================================
# 11. TIPOS DE DATOS
# ============================================================

print("\n" + "=" * 70)
print("11. TIPOS DE DATOS")
print("=" * 70)

print(
    df.dtypes.to_string()
)


# ============================================================
# 12. VARIABLES NUMÉRICAS
# ============================================================

print("\n" + "=" * 70)
print("12. VARIABLES NUMÉRICAS")
print("=" * 70)

columnas_numericas = df.select_dtypes(
    include=["number"]
).columns.tolist()

print(f"\nNúmero de columnas numéricas: {len(columnas_numericas)}")

for columna in columnas_numericas:
    print(f"  - {columna}")


# ============================================================
# 13. VARIABLES CATEGÓRICAS
# ============================================================

print("\n" + "=" * 70)
print("13. VARIABLES CATEGÓRICAS")
print("=" * 70)

columnas_categoricas = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print(f"\nNúmero de columnas categóricas: {len(columnas_categoricas)}")

for columna in columnas_categoricas:
    print(f"  - {columna}")


# ============================================================
# 14. VALORES NULOS
# ============================================================

print("\n" + "=" * 70)
print("14. VALORES NULOS")
print("=" * 70)

nulos = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

nulos_relevantes = nulos[nulos > 0]

print("\nColumnas con valores nulos:")

if len(nulos_relevantes) == 0:
    print("No hay valores nulos.")
else:
    print(nulos_relevantes.to_string())


# ============================================================
# 15. RESULTADO DE LA COMPETICIÓN
# ============================================================

print("\n" + "=" * 70)
print("15. RESULTADO DE LA COMPETICIÓN")
print("=" * 70)

print("\nValores más frecuentes:")

resultado_competicion = (
    df["resultado_competicion"]
    .dropna()
    .astype(str)
    .str.strip()
    .value_counts()
)

print(resultado_competicion.to_string())


# ============================================================
# 16. COMPROBAR RESULTADOS DE COMPETICIÓN
# ============================================================

print("\n" + "=" * 70)
print("16. EJEMPLOS DE RESULTADO DE COMPETICIÓN")
print("=" * 70)

print("\nValores distintos:")

print(
    sorted(
        df["resultado_competicion"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )
)


# ============================================================
# 17. ESTADÍSTICAS CON MAYOR CANTIDAD DE VALORES NULOS
# ============================================================

print("\n" + "=" * 70)
print("17. ESTADÍSTICAS CON MÁS VALORES NULOS")
print("=" * 70)

print(
    "\nTop 30:"
)

print(
    nulos
    .head(30)
    .to_string()
)


# ============================================================
# 18. COMPROBAR VARIABILIDAD DE LAS VARIABLES NUMÉRICAS
# ============================================================

print("\n" + "=" * 70)
print("18. VARIABLES NUMÉRICAS SIN VARIACIÓN")
print("=" * 70)

variables_sin_variacion = []

for columna in columnas_numericas:

    if df[columna].nunique(dropna=False) <= 1:
        variables_sin_variacion.append(columna)

if variables_sin_variacion:

    print("\nVariables sin variación:")
    for columna in variables_sin_variacion:
        print(f"  - {columna}")

else:

    print("\nNo hay variables numéricas sin variación.")


# ============================================================
# 19. COMPROBAR VARIABLES CON MUY POCOS VALORES DISTINTOS
# ============================================================

print("\n" + "=" * 70)
print("19. VARIABLES NUMÉRICAS CON POCOS VALORES DISTINTOS")
print("=" * 70)

for columna in columnas_numericas:

    numero_valores = df[columna].nunique(dropna=True)

    if numero_valores <= 5:

        print(
            f"\n{columna} → "
            f"{numero_valores} valores distintos"
        )

        print(
            df[columna]
            .value_counts(dropna=False)
            .to_string()
        )


# ============================================================
# 20. COMPROBAR DISTRIBUCIÓN DEL TARGET
# ============================================================

print("\n" + "=" * 70)
print("20. DISTRIBUCIÓN DEL TARGET")
print("=" * 70)

target_historico = (
    historico[
        historico["posicion_balon_oro"] > 0
    ]
    .groupby("temporada")
    ["posicion_balon_oro"]
    .nunique()
)

print("\nNúmero de posiciones distintas registradas por temporada:")

print(target_historico.to_string())


# ============================================================
# 21. COMPROBAR QUE 2026 NO ENTRE EN EL ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("21. SEPARACIÓN TRAIN / PREDICCIÓN")
print("=" * 70)

train = df[
    df["dataset"] == "historico"
].copy()

prediccion_2026 = df[
    df["dataset"] == "nominados_2026"
].copy()

print(f"\nFilas para entrenamiento: {len(train)}")
print(f"Filas para predicción 2026: {len(prediccion_2026)}")

print(
    "\nTemporadas de entrenamiento:"
)

print(
    sorted(
        train["temporada"]
        .dropna()
        .unique()
    )
)

print(
    "\nTemporadas para predicción:"
)

print(
    sorted(
        prediccion_2026["temporada"]
        .dropna()
        .unique()
    )
)


# ============================================================
# 22. COMPROBAR QUE NO HAY TARGET DESCONOCIDO EN HISTÓRICO
# ============================================================

print("\n" + "=" * 70)
print("22. COMPROBACIÓN DEL TARGET HISTÓRICO")
print("=" * 70)

target_nulo = train["posicion_balon_oro"].isna().sum()
target_cero = (train["posicion_balon_oro"] == 0).sum()

print(f"\nTarget NaN en histórico: {target_nulo}")
print(f"Target 0 en histórico:   {target_cero}")

if target_nulo > 0 or target_cero > 0:

    print(
        "\nADVERTENCIA: hay filas del histórico sin posición "
        "válida del Balón de Oro."
    )

else:

    print(
        "\nOK: todas las filas del histórico tienen "
        "una posición válida."
    )


# ============================================================
# 23. GUARDAR DATASET COMPLETO PREPARADO
# ============================================================

salida = DATA_DIR / "dataset_ml_base.csv"

df.to_csv(
    salida,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("23. DATASET GUARDADO")
print("=" * 70)

print(f"\nArchivo: {salida}")
print(f"Filas: {len(df)}")
print(f"Columnas: {len(df.columns)}")


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("PREPARACIÓN TERMINADA")
print("=" * 70)