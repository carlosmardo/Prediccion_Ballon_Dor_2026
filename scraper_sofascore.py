import csv
import json
import os
from botasaurus.browser import browser


# ============================================================
# CONFIGURACIÓN
# ============================================================

CSV_FILE = "nominados_ballon_dor.csv"
OUTPUT_DIR = "data/raw"
MAX_JUGADORES = None  # None -> procesa TODO el CSV.
GUARDAR_CHECKPOINT = True  # Guarda el progreso después de cada jugador


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_temporada(balon_de_oro_year):
    inicio = str(balon_de_oro_year - 1)[-2:]
    final = str(balon_de_oro_year)[-2:]
    return f"{inicio}/{final}"


def obtener_temporadas_validas(balon_de_oro_year):
    temporada_clubes = calcular_temporada(balon_de_oro_year)
    temporada_selecciones = str(balon_de_oro_year)
    return [temporada_clubes, temporada_selecciones]


def encontrar_temporada(seasons, balon_de_oro_year):
    temporadas_validas = obtener_temporadas_validas(balon_de_oro_year)
    for season in seasons:
        season_name = str(season).strip()
        if season_name in temporadas_validas:
            return season_name
    return None


def normalizar_posicion(position):
    if not position:
        return None
    position = str(position).strip().upper()
    posiciones = {
        "F": "Forward",
        "M": "Midfielder",
        "D": "Defender",
        "G": "Goalkeeper",
    }
    return posiciones.get(position, position)


def leer_csv():
    jugadores = []
    with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not any(str(value).strip() for value in row.values() if value is not None):
                continue

            # AÑO
            try:
                año = int(str(row["Año"]).strip())
            except (ValueError, TypeError, KeyError):
                print("\n⚠ Fila ignorada por año inválido:")
                print(row)
                continue

            # POSICIÓN
            try:
                posicion = int(str(row["Posición"]).strip())
            except (ValueError, TypeError, KeyError):
                print("\n⚠ Fila ignorada por posición inválida:")
                print(row)
                continue

            # NOMBRE / ID
            nombre_csv = str(row.get("Nombre", "")).strip()
            if not nombre_csv:
                print("\n⚠ Fila ignorada porque Nombre está vacío:")
                print(row)
                continue

            # INTENTAR OBTENER ID
            try:
                jugador_id = int(nombre_csv)
                tiene_sofascore_id = True
            except (ValueError, TypeError):
                jugador_id = None
                tiene_sofascore_id = False

            jugadores.append(
                {
                    "año": año,
                    "posicion": posicion,
                    "player_id": jugador_id,
                    "nombre_csv": nombre_csv,
                    "tiene_sofascore_id": tiene_sofascore_id,
                    "club": str(row.get("Club", "")).strip(),
                    "nacionalidad": str(row.get("Nacionalidad", "")).strip(),
                }
            )
    return jugadores


# ============================================================
# SELECTORES DE SOFASCORE
# ============================================================

def abrir_selector_season(driver):
    driver.run_js(
        """
        const boton = Array.from(document.querySelectorAll('button')).find(
            el => el.innerText.trim() === 'Season'
        );
        if (!boton) {
            throw new Error('No se encontró el botón Season');
        }
        boton.click();
        """
    )
    driver.sleep(1)


def abrir_selector_competicion(driver):
    driver.run_js(
        """
        const comboboxes = document.querySelectorAll('[role="combobox"]');
        if (comboboxes.length < 1) {
            throw new Error('No se encontró el selector de competición');
        }
        comboboxes[0].click();
        """
    )
    driver.sleep(1)


def obtener_opciones(driver):
    return driver.run_js(
        """
        return Array.from(document.querySelectorAll('[role="option"]'))
            .map(el => el.innerText.trim())
            .filter(Boolean);
        """
    )


def seleccionar_opcion(driver, nombre):
    nombre_js = json.dumps(nombre, ensure_ascii=False)
    return driver.run_js(
        """
        const nombre = __NOMBRE__;
        const opcion = Array.from(document.querySelectorAll('[role="option"]')).find(
            el => el.innerText.trim() === nombre
        );
        if (!opcion) {
            return false;
        }
        opcion.click();
        return true;
        """.replace("__NOMBRE__", nombre_js)
    )


def volver_a_competiciones(driver):
    abrir_selector_season(driver)
    abrir_selector_competicion(driver)


# ============================================================
# INFORMACIÓN DEL JUGADOR
# ============================================================

def obtener_datos_jugador(driver, player_id):
    resultado = driver.run_js(
        """
        const playerId = __PLAYER_ID__;
        const url = `/api/v1/player/${playerId}`;
        return fetch(
            url,
            {
                method: "GET",
                credentials: "include",
                headers: {
                    "X-Requested-With": "XMLHttpRequest"
                }
            }
        )
        .then(
            async response => {
                const text = await response.text();
                return {
                    status: response.status,
                    ok: response.ok,
                    body: text
                };
            }
        );
        """.replace("__PLAYER_ID__", str(player_id))
    )

    if not resultado["ok"]:
        raise RuntimeError(
            f"No se pudo obtener la información del jugador {player_id}. HTTP {resultado['status']}"
        )

    try:
        data = json.loads(resultado["body"])
    except json.JSONDecodeError:
        raise RuntimeError("La respuesta de información del jugador no es JSON válido.")

    return data


# ============================================================
# CREAR REGISTROS DE CONTROL
# ============================================================

def crear_registro_no_sofascore_id(jugador):
    temporada = calcular_temporada(jugador["año"])
    return {
        "status": "no_sofascore_id",
        "ballon_dor_year": jugador["año"],
        "ballon_dor_position": jugador["posicion"],
        "ballon_dor_club": jugador["club"],
        "player_id": None,
        "player_name": jugador["nombre_csv"],
        "position": None,
        "position_sofascore": None,
        "nationality": None,
        "country_alpha2": None,
        "csv_nationality": jugador["nacionalidad"],
        "season": temporada,
        "competition": None,
        "tournament_id": None,
        "season_id": None,
        "team": None,
        "statistics": None,
    }


def crear_registro_profile_not_found(jugador):
    temporada = calcular_temporada(jugador["año"])
    return {
        "status": "sofascore_profile_not_found",
        "ballon_dor_year": jugador["año"],
        "ballon_dor_position": jugador["posicion"],
        "ballon_dor_club": jugador["club"],
        "player_id": jugador["player_id"],
        "player_name": jugador["nombre_csv"],
        "position": None,
        "position_sofascore": None,
        "nationality": None,
        "country_alpha2": None,
        "csv_nationality": jugador["nacionalidad"],
        "season": temporada,
        "competition": None,
        "tournament_id": None,
        "season_id": None,
        "team": None,
        "statistics": None,
    }


def crear_registro_no_season_found(
    jugador, player_name, position, position_sofascore, nationality, country_alpha2
):
    temporada = calcular_temporada(jugador["año"])
    return {
        "status": "no_season_found",
        "ballon_dor_year": jugador["año"],
        "ballon_dor_position": jugador["posicion"],
        "ballon_dor_club": jugador["club"],
        "player_id": jugador["player_id"],
        "player_name": player_name,
        "position": position,
        "position_sofascore": position_sofascore,
        "nationality": nationality,
        "country_alpha2": country_alpha2,
        "csv_nationality": jugador["nacionalidad"],
        "season": temporada,
        "competition": None,
        "tournament_id": None,
        "season_id": None,
        "team": None,
        "statistics": None,
    }


def crear_registro_stats_not_found(
    jugador,
    player_name,
    position,
    position_sofascore,
    nationality,
    country_alpha2,
    competition,
    tournament_id,
    season_id,
    season=None
):
    if season is None:
        season = calcular_temporada(jugador["año"])
    return {
        "status": "stats_not_found",
        "ballon_dor_year": jugador["año"],
        "ballon_dor_position": jugador["posicion"],
        "ballon_dor_club": jugador["club"],
        "player_id": jugador["player_id"],
        "player_name": player_name,
        "position": position,
        "position_sofascore": position_sofascore,
        "nationality": nationality,
        "country_alpha2": country_alpha2,
        "csv_nationality": jugador["nacionalidad"],
        "season": season,
        "competition": competition,
        "tournament_id": tournament_id,
        "season_id": season_id,
        "team": None,
        "statistics": None,
    }


def construir_output_file(jugadores):
    años = sorted({jugador["año"] for jugador in jugadores})
    if len(años) == 1:
        output_name = f"historico_{años[0]}_sofascore_reintentos.json"
    elif años:
        output_name = f"historico_{años[0]}_{años[-1]}_sofascore_reintentos.json"
    else:
        output_name = "historico_sofascore_reintentos.json"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, output_name)


def guardar_resultados_json(resultados, jugadores, output_file=None):
    if output_file is None:
        output_file = construir_output_file(jugadores)

    temporal_file = output_file + ".tmp"
    with open(temporal_file, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    os.replace(temporal_file, output_file)
    return output_file


def crear_registro_profile_load_failed(jugador, status, error=None):
    temporada = calcular_temporada(jugador["año"])
    return {
        "status": status,
        "ballon_dor_year": jugador["año"],
        "ballon_dor_position": jugador["posicion"],
        "ballon_dor_club": jugador["club"],
        "player_id": jugador["player_id"],
        "player_name": jugador["nombre_csv"],
        "position": None,
        "position_sofascore": None,
        "nationality": None,
        "country_alpha2": None,
        "csv_nationality": jugador["nacionalidad"],
        "season": temporada,
        "competition": None,
        "tournament_id": None,
        "season_id": None,
        "team": None,
        "statistics": None,
        "error": str(error) if error is not None else None,
    }


def es_error_timeout(error):
    texto = f"{type(error).__name__}: {error}".lower()
    indicadores = (
        "timeout",
        "timed out",
        "document did not become ready",
        "response not received",
    )
    return any(indicador in texto for indicador in indicadores)


# ============================================================
# SCRAPER PRINCIPAL
# ============================================================

@browser(headless=False, output=None)
def scrape_historico(driver, data):
    print("=" * 70)
    print("SCRAPER HISTÓRICO - SOFASCORE")
    print("=" * 70)
    print(f"\nCSV: {CSV_FILE}")

    jugadores = leer_csv()
    if not jugadores:
        print("\n❌ No se encontraron jugadores en el CSV.")
        return

    if MAX_JUGADORES is not None:
        jugadores = jugadores[:MAX_JUGADORES]

    print(f"\nJugadores a procesar: {len(jugadores)}\n")

    for i, jugador in enumerate(jugadores, start=1):
        if jugador["tiene_sofascore_id"]:
            identificacion = f"SofaScore ID {jugador['player_id']}"
        else:
            identificacion = f"Sin ID SofaScore ({jugador['nombre_csv']})"
        print(f"{i}. Balón de Oro {jugador['año']} | Posición {jugador['posicion']} | {identificacion}")

    print("\nABRIENDO SOFASCORE")
    print("-" * 70)
    driver.get("https://www.sofascore.com")
    driver.sleep(3)
    print("✓ SofaScore abierto")

    resultados = []
    output_file = construir_output_file(jugadores)
    current_player_id = None
    pending_season = None

    # Guardamos TODAS las peticiones de estadísticas detectadas durante el jugador actual.
    # Clave: (tournament_id, season_id)
    stats_requests = {}

    def after_response(request_id, response, event):
        nonlocal pending_season
        try:
            if current_player_id is None:
                return

            url = getattr(response, "url", None)
            if not url:
                return

            marker_stats = f"/player/{current_player_id}/unique-tournament/"
            if marker_stats not in url or "/statistics/overall" not in url:
                return

            partes = url.split("/")
            try:
                indice_tournament = partes.index("unique-tournament")
                tournament_id = int(partes[indice_tournament + 1])
                indice_season = partes.index("season")
                season_id = int(partes[indice_season + 1])
            except (ValueError, IndexError):
                print("\n⚠ No se pudieron extraer tournament_id/season_id de la petición.")
                return

            request_data = {
                "player_id": current_player_id,
                "tournament_id": tournament_id,
                "season_id": season_id,
                "url": url,
            }
            stats_requests[(tournament_id, season_id)] = request_data
            pending_season = request_data
        except Exception as e:
            print(f"\n⚠ Error en listener:\n{e}")

    driver.after_response_received(after_response)

    for numero_jugador, jugador in enumerate(jugadores, start=1):
        player_id = jugador["player_id"]
        ballon_dor_year = jugador["año"]
        posicion_balon_dor = jugador["posicion"]
        club_ballon_dor = jugador["club"]
        nacionalidad_csv = jugador["nacionalidad"]

        temporada_objetivo = calcular_temporada(ballon_dor_year)
        temporadas_validas = obtener_temporadas_validas(ballon_dor_year)

        current_player_id = player_id
        pending_season = None
        stats_requests.clear()

        print(f"\n\n{'#' * 70}")
        print(f"JUGADOR {numero_jugador}/{len(jugadores)}")
        print(f"{'#' * 70}\n")
        print(f"Nombre CSV: {jugador['nombre_csv']}")
        print(f"Balón de Oro: {ballon_dor_year}")
        print(f"Posición: {posicion_balon_dor}")
        print(f"Club en ranking: {club_ballon_dor}")
        print(f"Nacionalidad CSV: {nacionalidad_csv}")
        print(f"Temporadas objetivo: {' OR '.join(temporadas_validas)}")

        if not jugador["tiene_sofascore_id"]:
            print("\n⚠ Este jugador no tiene un ID de SofaScore.")
            print(f"  Nombre: {jugador['nombre_csv']}")
            print("  Se conservará en el JSON.")
            print("  El scraper continuará con el siguiente jugador.")
            resultados.append(crear_registro_no_sofascore_id(jugador))
            continue

        print(f"\nSofaScore ID: {player_id}")

        print("\n1. OBTENIENDO INFORMACIÓN DEL JUGADOR")
        print("-" * 70)
        try:
            player_data = obtener_datos_jugador(driver, player_id)
        except Exception as e:
            print(f"\n⚠ No se pudo obtener el perfil de SofaScore.\n{e}\n")
            print("El jugador se conservará con status = sofascore_profile_not_found.")
            resultados.append(crear_registro_profile_not_found(jugador))
            continue

        player = player_data.get("player", player_data)
        player_name = player.get("name", jugador["nombre_csv"])
        player_slug = player.get("slug", "")
        position_sofascore = player.get("position")
        position = normalizar_posicion(position_sofascore)
        country = player.get("country", {}) or {}
        nationality = country.get("name")
        country_alpha2 = country.get("alpha2")

        print(f"\n✓ Jugador: {player_name}")
        print(f"✓ Slug: {player_slug}")
        print(f"✓ Posición SofaScore: {position_sofascore}")
        print(f"✓ Posición normalizada: {position}")
        print(f"✓ Nacionalidad SofaScore: {nationality}")

        if player_slug:
            player_url = f"https://www.sofascore.com/football/player/{player_slug}/{player_id}"
        else:
            player_url = f"https://www.sofascore.com/football/player/{player_id}"

        print("\n2. ABRIENDO PERFIL DE SOFASCORE")
        print("-" * 70)
        print(f"URL: {player_url}")

        try:
            driver.get(player_url)
            driver.sleep(3)
            print("✓ Perfil abierto")
        except Exception as e:
            status = "profile_load_timeout" if es_error_timeout(e) else "profile_load_failed"
            print(f"\n⚠ No se pudo abrir el perfil de SofaScore.\nTipo: {type(e).__name__}\nError: {e}\n")
            print(f"Se conserva el jugador con status = {status}.")
            resultados.append(crear_registro_profile_load_failed(jugador, status, e))
            if GUARDAR_CHECKPOINT:
                try:
                    guardar_resultados_json(resultados, jugadores, output_file)
                    print(f"✓ Checkpoint guardado: {output_file}")
                except Exception as checkpoint_error:
                    print(f"⚠ No se pudo guardar el checkpoint:\n{checkpoint_error}")
            continue

        print("\n3. OBTENIENDO COMPETICIONES")
        print("-" * 70)
        try:
            abrir_selector_season(driver)
            abrir_selector_competicion(driver)
            competitions = obtener_opciones(driver)
        except Exception as e:
            print(f"\n⚠ Error obteniendo competiciones:\n{e}")
            print("El jugador se conservará con status = no_season_found.")
            resultados.append(
                crear_registro_no_season_found(
                    jugador, player_name, position, position_sofascore, nationality, country_alpha2
                )
            )
            continue

        print()
        for competition in competitions:
            print(f"- {competition}")
        print(f"\nTotal de competiciones: {len(competitions)}")

        temporada_encontrada = False
        temporada_encontrada_nombre = None
        estadisticas_encontradas = False
        ultimo_stats_fallido = None

        for competition in competitions:
            print(f"\n{'=' * 70}\nCOMPETICIÓN: {competition}\n{'=' * 70}")

            stats_keys_before_competition = set(stats_requests.keys())
            competition_stats_request = None
            pending_season = None

            seleccionada = seleccionar_opcion(driver, competition)
            if not seleccionada:
                print(f"\n⚠ No se pudo seleccionar {competition}")
                try:
                    volver_a_competiciones(driver)
                except Exception:
                    pass
                continue

            driver.sleep(3)
            print(f"\n✓ {competition} seleccionada")

            nuevas_stats_competition = [
                stats_requests[key] for key in stats_requests if key not in stats_keys_before_competition
            ]
            if nuevas_stats_competition:
                competition_stats_request = nuevas_stats_competition[-1]

            seasons = obtener_opciones(driver)
            if not seasons:
                print("ℹ Selector de temporada cerrado.")
                try:
                    driver.run_js(
                        """
                        const comboboxes = document.querySelectorAll('[role="combobox"]');
                        if (comboboxes.length < 2) {
                            throw new Error('No se encontró el selector de temporadas');
                        }
                        comboboxes[1].click();
                        """
                    )
                    driver.sleep(1)
                    seasons = obtener_opciones(driver)
                except Exception as e:
                    print(f"\n⚠ No se pudo abrir el selector de temporadas:\n{e}")
                    try:
                        volver_a_competiciones(driver)
                    except Exception:
                        pass
                    continue

            print("\nTemporadas disponibles:")
            if seasons:
                print(", ".join(seasons))
            else:
                print("NINGUNA")

            temporada_encontrada_nombre = encontrar_temporada(seasons, ballon_dor_year)
            if temporada_encontrada_nombre is None:
                print(f"\n❌ Ninguna de las temporadas {' OR '.join(temporadas_validas)} está disponible")
                try:
                    volver_a_competiciones(driver)
                except Exception:
                    pass
                continue

            temporada_encontrada = True
            print(f"\n✓ Temporada encontrada en SofaScore: {temporada_encontrada_nombre}")
            if temporada_encontrada_nombre == temporada_objetivo:
                print(f"  Formato competición de clubes: {temporada_encontrada_nombre}")
            elif temporada_encontrada_nombre == str(ballon_dor_year):
                print(f"  Formato competición de selecciones: {temporada_encontrada_nombre}")

            pending_season = None
            stats_keys_before_season = set(stats_requests.keys())

            print("\nSeleccionando temporada...")
            seleccionada = seleccionar_opcion(driver, temporada_encontrada_nombre)
            if not seleccionada:
                print(f"\n⚠ No se pudo seleccionar {temporada_encontrada_nombre}")
                try:
                    volver_a_competiciones(driver)
                except Exception:
                    pass
                continue

            driver.sleep(3)

            nuevas_stats_season = [
                stats_requests[key] for key in stats_requests if key not in stats_keys_before_season
            ]
            if nuevas_stats_season:
                pending_season = nuevas_stats_season[-1]
            elif competition_stats_request is not None:
                pending_season = competition_stats_request
            else:
                pending_season = None
                print("\n⚠ La temporada existe, pero no se detectó ninguna petición de estadísticas.")

            if pending_season is None:
                ultimo_stats_fallido = {
                    "competition": competition,
                    "tournament_id": None,
                    "season_id": None,
                    "season": temporada_encontrada_nombre,
                }
                try:
                    volver_a_competiciones(driver)
                except Exception:
                    pass
                continue

            tournament_id = pending_season["tournament_id"]
            season_id = pending_season["season_id"]

            print(f"\n🎯 TEMPORADA ENCONTRADA\nTemporada: {temporada_encontrada_nombre}\nTournament ID: {tournament_id}\nSeason ID: {season_id}")

            print("\n4. OBTENIENDO ESTADÍSTICAS")
            print("-" * 70)
            print("Realizando petición desde SofaScore...")

            stats_result = driver.run_js(
                """
                const playerId = __PLAYER_ID__;
                const tournamentId = __TOURNAMENT_ID__;
                const seasonId = __SEASON_ID__;
                const url = `/api/v1/player/${playerId}/unique-tournament/${tournamentId}/season/${seasonId}/statistics/overall`;
                return fetch(
                    url,
                    {
                        method: "GET",
                        credentials: "include",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest"
                        }
                    }
                )
                .then(
                    async response => {
                        const text = await response.text();
                        return {
                            status: response.status,
                            ok: response.ok,
                            url: url,
                            body: text
                        };
                    }
                );
                """
                .replace("__PLAYER_ID__", str(player_id))
                .replace("__TOURNAMENT_ID__", str(tournament_id))
                .replace("__SEASON_ID__", str(season_id))
            )

            print(f"\nHTTP Status: {stats_result['status']}")

            if not stats_result["ok"]:
                print(f"\n⚠ Error obteniendo estadísticas.\n{stats_result['body']}")
                ultimo_stats_fallido = {
                    "competition": competition,
                    "tournament_id": tournament_id,
                    "season_id": season_id,
                    "season": temporada_encontrada_nombre,
                }
                try:
                    volver_a_competiciones(driver)
                except Exception:
                    pass
                continue

            try:
                statistics_data = json.loads(stats_result["body"])
            except json.JSONDecodeError:
                print(f"\n⚠ La respuesta no es JSON válido.\n{stats_result['body']}")
                ultimo_stats_fallido = {
                    "competition": competition,
                    "tournament_id": tournament_id,
                    "season_id": season_id,
                    "season": temporada_encontrada_nombre,
                }
                try:
                    volver_a_competiciones(driver)
                except Exception:
                    pass
                continue

            statistics = statistics_data.get("statistics", {})
            team = statistics_data.get("team", {}) or {}

            print("\nESTADÍSTICAS ENCONTRADAS")
            print("-" * 70)
            print(f"Equipo: {team.get('name', 'N/A')}")
            if statistics:
                print(f"\nApariciones: {statistics.get('appearances', 'N/A')}")
                print(f"Titularidades: {statistics.get('matchesStarted', 'N/A')}")
                print(f"Minutos: {statistics.get('minutesPlayed', 'N/A')}")
                print(f"Goles: {statistics.get('goals', 'N/A')}")
                print(f"Asistencias: {statistics.get('assists', 'N/A')}")
            else:
                print("\n⚠ No se encontraron estadísticas.")

            resultados.append(
                {
                    "status": "ok",
                    "ballon_dor_year": ballon_dor_year,
                    "ballon_dor_position": posicion_balon_dor,
                    "ballon_dor_club": club_ballon_dor,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "position_sofascore": position_sofascore,
                    "nationality": nationality,
                    "country_alpha2": country_alpha2,
                    "csv_nationality": nacionalidad_csv,
                    "season": temporada_encontrada_nombre,
                    "competition": competition,
                    "tournament_id": tournament_id,
                    "season_id": season_id,
                    "team": team,
                    "statistics": statistics,
                }
            )
            estadisticas_encontradas = True

            try:
                volver_a_competiciones(driver)
            except Exception as e:
                print(f"\n⚠ Error volviendo al selector:\n{e}")

        if not estadisticas_encontradas:
            if not temporada_encontrada:
                print(
                    "\n⚠ RESULTADO:\nNo existe ninguna de las temporadas "
                    f"{' OR '.join(temporadas_validas)} en ninguna competición encontrada."
                )
                print("Se conserva el jugador con:\nstatus = no_season_found")
                resultados.append(
                    crear_registro_no_season_found(
                        jugador, player_name, position, position_sofascore, nationality, country_alpha2
                    )
                )
            else:
                print(
                    f"\n⚠ La temporada {temporada_encontrada_nombre} sí existe, pero no "
                    "se pudieron obtener estadísticas."
                )
                print("Se conserva el jugador con status = stats_not_found.")
                if ultimo_stats_fallido:
                    resultados.append(
                        crear_registro_stats_not_found(
                            jugador,
                            player_name,
                            position,
                            position_sofascore,
                            nationality,
                            country_alpha2,
                            ultimo_stats_fallido["competition"],
                            ultimo_stats_fallido["tournament_id"],
                            ultimo_stats_fallido["season_id"],
                            ultimo_stats_fallido.get("season"),
                        )
                    )
                else:
                    resultados.append(
                        crear_registro_stats_not_found(
                            jugador,
                            player_name,
                            position,
                            position_sofascore,
                            nationality,
                            country_alpha2,
                            None,
                            None,
                            None,
                            temporada_encontrada_nombre,
                        )
                    )

        print(f"\n{'-' * 70}\nRESULTADO: {player_name}\n{'-' * 70}")
        resultados_jugador = [
            r for r in resultados
            if r.get("player_id") == player_id and r.get("ballon_dor_year") == ballon_dor_year
        ]
        for info in resultados_jugador:
            status = info.get("status")
            print(f"\nStatus: {status}\nCompetición: {info.get('competition')}\nTemporada: {info.get('season')}")
            if status == "ok":
                statistics = info.get("statistics", {}) or {}
                team = info.get("team", {}) or {}
                print(f"Equipo: {team.get('name', 'N/A')}")
                print(f"Goles: {statistics.get('goals', 'N/A')}")
                print(f"Asistencias: {statistics.get('assists', 'N/A')}")
                print(f"Apariciones: {statistics.get('appearances', 'N/A')}")

        print(f"\n✓ Jugador terminado: {player_name}")

        if GUARDAR_CHECKPOINT:
            try:
                guardar_resultados_json(resultados, jugadores, output_file)
                print(f"\n✓ Checkpoint guardado: {output_file}")
            except Exception as e:
                print(f"\n⚠ No se pudo guardar el checkpoint:\n{e}")

    print(f"\n\n{'=' * 70}\nRESULTADO FINAL DEL SCRAPER\n{'=' * 70}\n")
    print(f"Jugadores procesados: {len(jugadores)}")
    print(f"Registros generados: {len(resultados)}")

    estados = {}
    for resultado in resultados:
        status = resultado.get("status", "unknown")
        estados[status] = estados.get(status, 0) + 1

    print("\nEstados:")
    for status, cantidad in estados.items():
        print(f"  - {status}: {cantidad}")

    output_file = guardar_resultados_json(resultados, jugadores, output_file)
    print(f"\n✓ Resultado guardado:\n{output_file}\n\n{'=' * 70}\n✅ SCRAPING TERMINADO\n{'=' * 70}")


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":
    scrape_historico()