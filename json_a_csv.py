import json
import csv
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

JSON_FILE = Path("json_stats_jugadores_temporadas_sofascore/historico_2000_2025_sofascore_final.json")
CSV_FILE = Path("csv_stats_jugadores_temporada/stats_historico_ballon_dor.csv")


# ============================================================
# LEER JSON
# ============================================================

print(f"Leyendo: {JSON_FILE}")

with open(JSON_FILE, "r", encoding="utf-8") as f:
    datos = json.load(f)

print(f"Registros encontrados: {len(datos)}")


# ============================================================
# OBTENER TODAS LAS ESTADÍSTICAS EXISTENTES
# ============================================================

estadisticas = set()

for registro in datos:
    stats = registro.get("statistics")

    if isinstance(stats, dict):
        estadisticas.update(stats.keys())

# Ordenarlas para que el CSV sea siempre reproducible
estadisticas = sorted(estadisticas)

print(f"Estadísticas diferentes encontradas: {len(estadisticas)}")


# ============================================================
# COLUMNAS DEL CSV
# ============================================================

columnas_base = [
    "temporada",
    "posicion_balon_oro",
    "club",
    "posicion",
    "nacionalidad",
    "nombre",
    "id_sofascore",
    "competicion",
]

columnas = columnas_base + estadisticas


# ============================================================
# CREAR CSV
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columnas,
        extrasaction="ignore"
    )

    writer.writeheader()

    for registro in datos:

        fila = {
            "temporada": registro.get("season"),
            "posicion_balon_oro": registro.get("ballon_dor_position"),
            "club": registro.get("ballon_dor_club"),
            "posicion": registro.get("position"),
            "nacionalidad": registro.get("nationality"),
            "nombre": registro.get("player_name"),
            "id_sofascore": registro.get("player_id"),
            "competicion": registro.get("competition"),
        }

        # Estadísticas
        stats = registro.get("statistics")

        if isinstance(stats, dict):

            for estadistica in estadisticas:
                fila[estadistica] = stats.get(estadistica)

        writer.writerow(fila)


# ============================================================
# RESULTADO
# ============================================================

print()
print("=" * 60)
print("CSV GENERADO CORRECTAMENTE")
print("=" * 60)

print(f"Archivo: {CSV_FILE}")
print(f"Filas: {len(datos)}")
print(f"Columnas: {len(columnas)}")
print(f"Estadísticas: {len(estadisticas)}")