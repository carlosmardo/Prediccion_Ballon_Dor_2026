import json


JSON_COMPLETO = "json_stats_jugadores_temporadas_sofascore/historico_2000_2025_sofascore.json"
JSON_REINTENTOS = "json_stats_jugadores_temporadas_sofascore/historico_2000_2018_sofascore_reintentos.json"
JSON_FIFA_WORLD_CUP = "json_stats_jugadores_temporadas_sofascore/fifa_world_cup_2022_sofascore.json"
JSON_SALIDA = "json_stats_jugadores_temporadas_sofascore/historico_2000_2025_sofascore_final.json"


# =========================================================
# 1. CARGAR JSON COMPLETO
# =========================================================

with open(JSON_COMPLETO, "r", encoding="utf-8") as f:
    registros_completos = json.load(f)


# =========================================================
# 2. CARGAR JSON DE REINTENTOS
# =========================================================

with open(JSON_REINTENTOS, "r", encoding="utf-8") as f:
    registros_reintentos = json.load(f)


# =========================================================
# 3. CARGAR JSON FIFA WORLD CUP 2022
# =========================================================

with open(JSON_FIFA_WORLD_CUP, "r", encoding="utf-8") as f:
    registros_fifa_world_cup = json.load(f)


# =========================================================
# 4. CAMBIAR SEASON 2022 → 2023
# =========================================================

registros_season_modificados = 0

for registro in registros_fifa_world_cup:

    if registro.get("season") == "2022":

        registro["season"] = "2023"
        registros_season_modificados += 1


print(
    f"Registros FIFA World Cup con season 2022 → 2023: "
    f"{registros_season_modificados}"
)


# =========================================================
# 5. FILTRAR SOLO OK DEL FIFA WORLD CUP
# =========================================================

registros_fifa_world_cup_ok = [
    registro
    for registro in registros_fifa_world_cup
    if registro.get("status") == "ok"
]


print(f"Registros JSON completo:       {len(registros_completos)}")
print(f"Registros reintentos:          {len(registros_reintentos)}")
print(f"Registros FIFA World Cup:      {len(registros_fifa_world_cup)}")
print(f"Registros FIFA World Cup OK:   {len(registros_fifa_world_cup_ok)}")


# =========================================================
# 6. ELIMINAR ERRORES DEL JSON COMPLETO
# =========================================================

estados_a_eliminar = {
    "no_season_found",
    "sofascore_profile_not_found",
    "profile_load_timeout",
}


registros_sin_errores = [
    registro
    for registro in registros_completos
    if registro.get("status") not in estados_a_eliminar
]


errores_eliminados = (
    len(registros_completos)
    - len(registros_sin_errores)
)


# =========================================================
# 7. AÑADIR TODOS LOS REGISTROS DE REINTENTOS
#    + FIFA WORLD CUP 2022 CON STATUS OK
# =========================================================

resultado_final = (
    registros_sin_errores
    + registros_reintentos
    + registros_fifa_world_cup_ok
)


# =========================================================
# 8. GUARDAR JSON FINAL
# =========================================================

with open(JSON_SALIDA, "w", encoding="utf-8") as f:
    json.dump(
        resultado_final,
        f,
        ensure_ascii=False,
        indent=4
    )


# =========================================================
# 9. RESUMEN
# =========================================================

print()
print("=" * 60)
print("MERGE COMPLETADO")
print("=" * 60)

print(f"Registros originales:          {len(registros_completos)}")
print(f"Errores eliminados:            {errores_eliminados}")
print(f"Registros válidos:             {len(registros_sin_errores)}")
print(f"Registros de reintentos:       {len(registros_reintentos)}")
print(f"Registros FIFA World Cup OK:   {len(registros_fifa_world_cup_ok)}")
print(f"Season 2022 → 2023:            {registros_season_modificados}")
print(f"Registros finales:              {len(resultado_final)}")

print()
print(f"✓ Archivo generado:")
print(JSON_SALIDA)