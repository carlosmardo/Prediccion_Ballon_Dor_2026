import json
import csv


# ============================================================
# CONFIGURACIÓN
# ============================================================

JSON_INPUT = "data/raw/sofascore/historico_2000_2025_sofascore.json"
CSV_OUTPUT = "data/raw/sofascore/reintentos.csv"


# Estados que queremos sacar para volver a procesar
ESTADOS_FALLIDOS = {
    "no_season_found",
    "no_sofascore_id",
    "profile_load_timeout",
    "profile_load_failed",
    "sofascore_profile_not_found",
    "stats_not_found",
}


# ============================================================
# LEER JSON
# ============================================================

with open(JSON_INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)


# Tu JSON es una lista de objetos
if not isinstance(data, list):
    raise ValueError(
        "El JSON no contiene una lista de registros."
    )


# ============================================================
# EXTRAER REGISTROS FALLIDOS
# ============================================================

fallidos = []

for registro in data:

    status = registro.get("status")

    if status not in ESTADOS_FALLIDOS:
        continue

    fallidos.append({
        "Año": registro.get("ballon_dor_year", ""),
        "Posición": registro.get("ballon_dor_position", ""),
        "Nombre": registro.get("player_id", ""),
        "Club": registro.get("ballon_dor_club", ""),
        "Nacionalidad": registro.get("csv_nationality", ""),
    })


# ============================================================
# GUARDAR CSV
# ============================================================

with open(
    CSV_OUTPUT,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Año",
            "Posición",
            "Nombre",
            "Club",
            "Nacionalidad",
        ]
    )

    writer.writeheader()
    writer.writerows(fallidos)


# ============================================================
# RESULTADO
# ============================================================

print("=" * 70)
print("REGISTROS FALLIDOS")
print("=" * 70)

print(f"Total registros JSON: {len(data)}")
print(f"Registros fallidos:   {len(fallidos)}")

print()
print(f"✓ CSV guardado en:")
print(CSV_OUTPUT)