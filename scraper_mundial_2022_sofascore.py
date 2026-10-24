import csv
import json
import os

from botasaurus.browser import browser


# ============================================================
# CONFIGURACIÓN
# ============================================================

CSV_FILE = "csv_fuentes/ballon_dor_ranking.csv"
OUTPUT_DIR = "json_stats_jugadores_temporadas_sofascore"

# ============================================================
# SOLO FIFA WORLD CUP 2022
# ============================================================

TARGET_BALON_DOR_YEAR = 2023
TARGET_COMPETITION = "FIFA World Cup"
TARGET_SEASON = "2022"

# IDs SofaScore de los 30 jugadores del Balón de Oro 2023
TARGET_PLAYER_IDS = {
    12994,
    839956,
    826643,
    70996,
    827606,
    868812,
    944656,
    822471,
    331209,
    15466,
    159665,
    41789,
    360938,
    45853,
    158263,
    3306,
    889259,
    991011,
    108579,
    823984,
    85859,
    896569,
    787498,
    934235,
    964994,
    1010231,
    363856,
    871706,
    547410,
    318941,
}

# None -> procesa todos los jugadores objetivo.
MAX_JUGADORES = None

# Guarda el progreso después de cada jugador.
GUARDAR_CHECKPOINT = True


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

            # ------------------------------------------------
            # AÑO
            # ------------------------------------------------

            try:
                año = int(str(row["Año"]).strip())
            except (ValueError, TypeError, KeyError):
                print()
                print("⚠ Fila ignorada por año inválido:")
                print(row)
                continue

            # ------------------------------------------------
            # SOLO BALÓN DE ORO 2023
            # ------------------------------------------------

            if año != TARGET_BALON_DOR_YEAR:
                continue

            # ------------------------------------------------
            # POSICIÓN
            # ------------------------------------------------

            try:
                posicion = int(str(row["Posición"]).strip())
            except (ValueError, TypeError, KeyError):
                print()
                print("⚠ Fila ignorada por posición inválida:")
                print(row)
                continue

            # ------------------------------------------------
            # NOMBRE / ID
            # ------------------------------------------------

            nombre_csv = str(row.get("Nombre", "")).strip()

            if not nombre_csv:
                print()
                print("⚠ Fila ignorada porque Nombre está vacío:")
                print(row)
                continue

            # ------------------------------------------------
            # INTENTAR OBTENER ID
            # ------------------------------------------------

            try:
                jugador_id = int(nombre_csv)
                tiene_sofascore_id = True
            except (ValueError, TypeError):
                jugador_id = None
                tiene_sofascore_id = False

            # ------------------------------------------------
            # SOLO LOS 30 JUGADORES INDICADOS
            # ------------------------------------------------

            if jugador_id not in TARGET_PLAYER_IDS:
                continue

            # ------------------------------------------------
            # GUARDAR FILA
            # ------------------------------------------------

            jugadores.append({
                "año": año,
                "posicion": posicion,
                "player_id": jugador_id,
                "nombre_csv": nombre_csv,
                "tiene_sofascore_id": tiene_sofascore_id,
                "club": str(row.get("Club", "")).strip(),
                "nacionalidad": str(row.get("Nacionalidad", "")).strip(),
            })

    # Ordenar por posición del Balón de Oro.
    jugadores.sort(key=lambda jugador: jugador["posicion"])

    return jugadores


# ============================================================
# SELECTORES DE SOFASCORE
# ============================================================

def abrir_selector_season(driver):
    driver.run_js("""
        const boton = Array.from(document.querySelectorAll('button')).find(el => el.innerText.trim() === 'Season');

        if (!boton) {
            throw new Error('No se encontró el botón Season');
        }

        boton.click();
    """)

    driver.sleep(1)


def abrir_selector_competicion(driver):
    driver.run_js("""
        const comboboxes = document.querySelectorAll('[role="combobox"]');

        if (comboboxes.length < 1) {
            throw new Error('No se encontró el selector de competición');
        }

        comboboxes[0].click();
    """)

    driver.sleep(1)


def obtener_opciones(driver):
    return driver.run_js("""
        return Array.from(document.querySelectorAll('[role="option"]'))
            .map(el => el.innerText.trim())
            .filter(Boolean);
    """)


def seleccionar_opcion(driver, nombre):
    nombre_js = json.dumps(nombre, ensure_ascii=False)

    return driver.run_js("""
        const nombre = __NOMBRE__;

        const opcion = Array.from(document.querySelectorAll('[role="option"]')).find(
            el => el.innerText.trim() === nombre
        );

        if (!opcion) {
            return false;
        }

        opcion.click();
        return true;
    """.replace("__NOMBRE__", nombre_js))


def volver_a_competiciones(driver):
    abrir_selector_season(driver)
    abrir_selector_competicion(driver)


# ============================================================
# INFORMACIÓN DEL JUGADOR
# ============================================================

def obtener_datos_jugador(driver, player_id):
    resultado = driver.run_js("""
        const playerId = __PLAYER_ID__;
        const url = `/api/v1/player/${playerId}`;

        return fetch(url, {
            method: "GET",
            credentials: "include",
            headers: {
                "X-Requested-With": "XMLHttpRequest"
            }
        }).then(async response => {
            const text = await response.text();

            return {
                status: response.status,
                ok: response.ok,
                body: text
            };
        });
    """.replace("__PLAYER_ID__", str(player_id)))

    if not resultado["ok"]:
        raise RuntimeError(
            "No se pudo obtener la información "
            f"del jugador {player_id}. "
            f"HTTP {resultado['status']}"
        )

    try:
        data = json.loads(resultado["body"])
    except json.JSONDecodeError:
        raise RuntimeError(
            "La respuesta de información del jugador no es JSON válido."
        )

    return data


# ============================================================
# CREAR REGISTROS DE CONTROL
# ============================================================

def crear_registro_no_sofascore_id(jugador):
    temporada = TARGET_SEASON

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
        "competition": TARGET_COMPETITION,
        "tournament_id": None,
        "season_id": None,
        "team": None,
        "statistics": None,
    }


def crear_registro_profile_not_found(jugador):
    temporada = TARGET_SEASON

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
        "competition": TARGET_COMPETITION,
        "tournament_id": None,
        "season_id": None,
        "team": None,
        "statistics": None,
    }


def crear_registro_no_season_found(
    jugador,
    player_name,
    position,
    position_sofascore,
    nationality,
    country_alpha2
):
    temporada = TARGET_SEASON

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
        "competition": TARGET_COMPETITION,
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
        season = TARGET_SEASON

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
    if jugadores:
        output_name = (
            f"fifa_world_cup_2022_sofascore.json"
        )
    else:
        output_name = "fifa_world_cup_2022_sofascore.json"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return os.path.join(OUTPUT_DIR, output_name)


def guardar_resultados_json(resultados, jugadores, output_file=None):
    if output_file is None:
        output_file = construir_output_file(jugadores)

    temporal_file = output_file + ".tmp"

    with open(temporal_file, "w", encoding="utf-8") as f:
        json.dump(
            resultados,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(temporal_file, output_file)

    return output_file


def crear_registro_profile_load_failed(jugador, status, error=None):
    temporada = TARGET_SEASON

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
        "competition": TARGET_COMPETITION,
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
    # ========================================================
    # 1. LEER CSV
    # ========================================================

    print("=" * 70)
    print("SCRAPER FIFA WORLD CUP 2022 - SOFASCORE")
    print("=" * 70)
    print()
    print(f"CSV: {CSV_FILE}")
    print(f"Competición: {TARGET_COMPETITION}")
    print(f"Temporada SofaScore: {TARGET_SEASON}")
    print(f"Balón de Oro: {TARGET_BALON_DOR_YEAR}")

    jugadores = leer_csv()

    if not jugadores:
        print()
        print("❌ No se encontraron los jugadores objetivo en el CSV.")
        return

    if MAX_JUGADORES is not None:
        jugadores = jugadores[:MAX_JUGADORES]

    print()
    print(f"Jugadores a procesar: {len(jugadores)}")
    print()

    for i, jugador in enumerate(jugadores, start=1):
        if jugador["tiene_sofascore_id"]:
            identificacion = f"SofaScore ID {jugador['player_id']}"
        else:
            identificacion = f"Sin ID SofaScore ({jugador['nombre_csv']})"

        print(
            f"{i}. "
            f"Balón de Oro {jugador['año']} | "
            f"Posición {jugador['posicion']} | "
            f"{identificacion}"
        )

    # ========================================================
    # 2. ABRIR SOFASCORE
    # ========================================================

    print()
    print("ABRIENDO SOFASCORE")
    print("-" * 70)

    driver.get("https://www.sofascore.com")
    driver.sleep(3)

    print("✓ SofaScore abierto")

    # ========================================================
    # 3. VARIABLES
    # ========================================================

    resultados = []
    output_file = construir_output_file(jugadores)

    current_player_id = None
    pending_season = None
    stats_requests = {}

    # ========================================================
    # 4. LISTENER
    # ========================================================

    def after_response(request_id, response, event):
        nonlocal pending_season

        try:
            if current_player_id is None:
                return

            url = getattr(response, "url", None)

            if not url:
                return

            marker_stats = f"/player/{current_player_id}/unique-tournament/"

            if marker_stats not in url:
                return

            if "/statistics/overall" not in url:
                return

            partes = url.split("/")

            try:
                indice_tournament = partes.index("unique-tournament")
                tournament_id = int(partes[indice_tournament + 1])

                indice_season = partes.index("season")
                season_id = int(partes[indice_season + 1])
            except (ValueError, IndexError):
                print()
                print("⚠ No se pudieron extraer tournament_id/season_id de la petición.")
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
            print()
            print("⚠ Error en listener:")
            print(e)

    driver.after_response_received(after_response)

    # ========================================================
    # 5. PROCESAR JUGADORES
    # ========================================================

    for numero_jugador, jugador in enumerate(jugadores, start=1):
        player_id = jugador["player_id"]
        ballon_dor_year = jugador["año"]
        posicion_balon_dor = jugador["posicion"]
        club_ballon_dor = jugador["club"]
        nacionalidad_csv = jugador["nacionalidad"]

        temporada_objetivo = TARGET_SEASON

        # ====================================================
        # RESET
        # ====================================================

        current_player_id = player_id
        pending_season = None
        stats_requests.clear()

        # ====================================================
        # CABECERA
        # ====================================================

        print()
        print()
        print("#" * 70)
        print(f"JUGADOR {numero_jugador}/{len(jugadores)}")
        print("#" * 70)
        print()

        print(f"Nombre CSV: {jugador['nombre_csv']}")
        print(f"Balón de Oro: {ballon_dor_year}")
        print(f"Posición: {posicion_balon_dor}")
        print(f"Club en ranking: {club_ballon_dor}")
        print(f"Nacionalidad CSV: {nacionalidad_csv}")
        print(f"Competición objetivo: {TARGET_COMPETITION}")
        print(f"Temporada objetivo: {TARGET_SEASON}")

        # ====================================================
        # CASO 1: SIN ID
        # ====================================================

        if not jugador["tiene_sofascore_id"]:
            print()
            print("⚠ Este jugador no tiene un ID de SofaScore.")
            print("  Se conservará en el JSON.")

            registro = crear_registro_no_sofascore_id(jugador)
            resultados.append(registro)
            continue

        # ====================================================
        # CASO 2: TIENE ID
        # ====================================================

        print()
        print(f"SofaScore ID: {player_id}")

        # ====================================================
        # 5.1 INFORMACIÓN
        # ====================================================

        print()
        print("1. OBTENIENDO INFORMACIÓN DEL JUGADOR")
        print("-" * 70)

        try:
            player_data = obtener_datos_jugador(driver, player_id)
        except Exception as e:
            print()
            print("⚠ No se pudo obtener el perfil de SofaScore.")
            print(e)
            print()
            print("El jugador se conservará con status = sofascore_profile_not_found.")

            registro = crear_registro_profile_not_found(jugador)
            resultados.append(registro)
            continue

        # ====================================================
        # ESTRUCTURA
        # ====================================================

        player = player_data.get("player", player_data)

        player_name = player.get("name", jugador["nombre_csv"])
        player_slug = player.get("slug", "")
        position_sofascore = player.get("position")
        position = normalizar_posicion(position_sofascore)

        country = player.get("country", {}) or {}
        nationality = country.get("name")
        country_alpha2 = country.get("alpha2")

        print()
        print(f"✓ Jugador: {player_name}")
        print(f"✓ Slug: {player_slug}")
        print(f"✓ Posición SofaScore: {position_sofascore}")
        print(f"✓ Posición normalizada: {position}")
        print(f"✓ Nacionalidad SofaScore: {nationality}")

        # ====================================================
        # 5.2 URL
        # ====================================================

        if player_slug:
            player_url = (
                "https://www.sofascore.com/"
                "football/player/"
                f"{player_slug}/"
                f"{player_id}"
            )
        else:
            player_url = (
                "https://www.sofascore.com/"
                "football/player/"
                f"{player_id}"
            )

        print()
        print("2. ABRIENDO PERFIL DE SOFASCORE")
        print("-" * 70)
        print(f"URL: {player_url}")

        try:
            driver.get(player_url)
            driver.sleep(3)
            print("✓ Perfil abierto")
        except Exception as e:
            status = (
                "profile_load_timeout"
                if es_error_timeout(e)
                else "profile_load_failed"
            )

            print()
            print("⚠ No se pudo abrir el perfil de SofaScore.")
            print(f"Tipo: {type(e).__name__}")
            print(f"Error: {e}")
            print()
            print(f"Se conserva el jugador con status = {status}.")

            resultados.append(
                crear_registro_profile_load_failed(
                    jugador,
                    status,
                    e
                )
            )

            if GUARDAR_CHECKPOINT:
                try:
                    guardar_resultados_json(
                        resultados,
                        jugadores,
                        output_file
                    )
                    print(f"✓ Checkpoint guardado: {output_file}")
                except Exception as checkpoint_error:
                    print("⚠ No se pudo guardar el checkpoint:")
                    print(checkpoint_error)

            continue

        # ====================================================
        # 5.3 ABRIR SELECTOR DE COMPETICIÓN
        # ====================================================

        print()
        print("3. SELECCIONANDO FIFA WORLD CUP")
        print("-" * 70)

        try:
            abrir_selector_season(driver)
            abrir_selector_competicion(driver)
            competitions = obtener_opciones(driver)
        except Exception as e:
            print()
            print("⚠ Error obteniendo competiciones:")
            print(e)

            registro = crear_registro_no_season_found(
                jugador,
                player_name,
                position,
                position_sofascore,
                nationality,
                country_alpha2
            )

            resultados.append(registro)
            continue

        print()
        print(f"Competiciones encontradas: {len(competitions)}")

        # ====================================================
        # SOLO FIFA WORLD CUP
        # ====================================================

        if TARGET_COMPETITION not in competitions:
            print()
            print(
                f"❌ No se encontró '{TARGET_COMPETITION}' "
                "entre las competiciones disponibles."
            )

            registro = crear_registro_no_season_found(
                jugador,
                player_name,
                position,
                position_sofascore,
                nationality,
                country_alpha2
            )

            resultados.append(registro)
            continue

        print(f"✓ '{TARGET_COMPETITION}' disponible")

        # ====================================================
        # SELECCIONAR FIFA WORLD CUP
        # ====================================================

        stats_keys_before_competition = set(stats_requests.keys())
        competition_stats_request = None
        pending_season = None

        seleccionada = seleccionar_opcion(
            driver,
            TARGET_COMPETITION
        )

        if not seleccionada:
            print()
            print(f"⚠ No se pudo seleccionar {TARGET_COMPETITION}")

            registro = crear_registro_stats_not_found(
                jugador,
                player_name,
                position,
                position_sofascore,
                nationality,
                country_alpha2,
                TARGET_COMPETITION,
                None,
                None,
                TARGET_SEASON
            )

            resultados.append(registro)
            continue

        driver.sleep(3)

        print()
        print(f"✓ {TARGET_COMPETITION} seleccionada")

        # ====================================================
        # COMPROBAR PETICIONES GENERADAS
        # ====================================================

        nuevas_stats_competition = [
            stats_requests[key]
            for key in stats_requests
            if key not in stats_keys_before_competition
        ]

        if nuevas_stats_competition:
            competition_stats_request = nuevas_stats_competition[-1]

        # ====================================================
        # OBTENER TEMPORADAS
        # ====================================================

        seasons = obtener_opciones(driver)

        if not seasons:
            print("ℹ Selector de temporada cerrado.")

            try:
                driver.run_js("""
                    const comboboxes = document.querySelectorAll('[role="combobox"]');

                    if (comboboxes.length < 2) {
                        throw new Error('No se encontró el selector de temporadas');
                    }

                    comboboxes[1].click();
                """)

                driver.sleep(1)
                seasons = obtener_opciones(driver)
            except Exception as e:
                print()
                print("⚠ No se pudo abrir el selector de temporadas:")
                print(e)

                registro = crear_registro_stats_not_found(
                    jugador,
                    player_name,
                    position,
                    position_sofascore,
                    nationality,
                    country_alpha2,
                    TARGET_COMPETITION,
                    None,
                    None,
                    TARGET_SEASON
                )

                resultados.append(registro)
                continue

        print()
        print("Temporadas disponibles:")

        if seasons:
            print(", ".join(seasons))
        else:
            print("NINGUNA")

        # ====================================================
        # COMPROBAR 2022
        # ====================================================

        if TARGET_SEASON not in seasons:
            print()
            print(
                f"❌ La temporada {TARGET_SEASON} "
                "no está disponible para FIFA World Cup."
            )

            registro = crear_registro_stats_not_found(
                jugador,
                player_name,
                position,
                position_sofascore,
                nationality,
                country_alpha2,
                TARGET_COMPETITION,
                None,
                None,
                TARGET_SEASON
            )

            resultados.append(registro)
            continue

        print()
        print(f"✓ Temporada encontrada: {TARGET_SEASON}")

        # ====================================================
        # 5.4 SELECCIONAR TEMPORADA 2022
        # ====================================================

        pending_season = None

        stats_keys_before_season = set(stats_requests.keys())

        print()
        print("Seleccionando temporada 2022...")

        seleccionada = seleccionar_opcion(
            driver,
            TARGET_SEASON
        )

        if not seleccionada:
            print()
            print(f"⚠ No se pudo seleccionar {TARGET_SEASON}")

            registro = crear_registro_stats_not_found(
                jugador,
                player_name,
                position,
                position_sofascore,
                nationality,
                country_alpha2,
                TARGET_COMPETITION,
                None,
                None,
                TARGET_SEASON
            )

            resultados.append(registro)
            continue

        driver.sleep(3)

        # ====================================================
        # COMPROBAR PETICIÓN NUEVA
        # ====================================================

        nuevas_stats_season = [
            stats_requests[key]
            for key in stats_requests
            if key not in stats_keys_before_season
        ]

        if nuevas_stats_season:
            pending_season = nuevas_stats_season[-1]
        elif competition_stats_request is not None:
            pending_season = competition_stats_request

        # ====================================================
        # SI NO HAY PETICIÓN, BUSCARLA DIRECTAMENTE
        # ====================================================

        if pending_season is None:
            print()
            print(
                "⚠ No se detectó ninguna petición automática "
                "de estadísticas."
            )

            registro = crear_registro_stats_not_found(
                jugador,
                player_name,
                position,
                position_sofascore,
                nationality,
                country_alpha2,
                TARGET_COMPETITION,
                None,
                None,
                TARGET_SEASON
            )

            resultados.append(registro)
            continue

        tournament_id = pending_season["tournament_id"]
        season_id = pending_season["season_id"]

        print()
        print("🎯 TEMPORADA ENCONTRADA")
        print(f"Competición: {TARGET_COMPETITION}")
        print(f"Temporada: {TARGET_SEASON}")
        print(f"Tournament ID: {tournament_id}")
        print(f"Season ID: {season_id}")

        # ====================================================
        # 5.5 ESTADÍSTICAS
        # ====================================================

        print()
        print("4. OBTENIENDO ESTADÍSTICAS")
        print("-" * 70)
        print("Realizando petición desde SofaScore...")

        stats_result = driver.run_js("""
            const playerId = __PLAYER_ID__;
            const tournamentId = __TOURNAMENT_ID__;
            const seasonId = __SEASON_ID__;

            const url =
                `/api/v1/player/${playerId}` +
                `/unique-tournament/${tournamentId}` +
                `/season/${seasonId}` +
                `/statistics/overall`;

            return fetch(url, {
                method: "GET",
                credentials: "include",
                headers: {
                    "X-Requested-With": "XMLHttpRequest"
                }
            }).then(async response => {
                const text = await response.text();

                return {
                    status: response.status,
                    ok: response.ok,
                    url: url,
                    body: text
                };
            });
        """.replace(
            "__PLAYER_ID__",
            str(player_id)
        ).replace(
            "__TOURNAMENT_ID__",
            str(tournament_id)
        ).replace(
            "__SEASON_ID__",
            str(season_id)
        ))

        print()
        print(f"HTTP Status: {stats_result['status']}")

        # ====================================================
        # HTTP
        # ====================================================

        if not stats_result["ok"]:
            print()
            print("⚠ Error obteniendo estadísticas.")
            print(stats_result["body"])

            registro = crear_registro_stats_not_found(
                jugador,
                player_name,
                position,
                position_sofascore,
                nationality,
                country_alpha2,
                TARGET_COMPETITION,
                tournament_id,
                season_id,
                TARGET_SEASON
            )

            resultados.append(registro)
            continue

        # ====================================================
        # JSON
        # ====================================================

        try:
            statistics_data = json.loads(stats_result["body"])
        except json.JSONDecodeError:
            print()
            print("⚠ La respuesta no es JSON válido.")
            print(stats_result["body"])

            registro = crear_registro_stats_not_found(
                jugador,
                player_name,
                position,
                position_sofascore,
                nationality,
                country_alpha2,
                TARGET_COMPETITION,
                tournament_id,
                season_id,
                TARGET_SEASON
            )

            resultados.append(registro)
            continue

        statistics = statistics_data.get("statistics", {})
        team = statistics_data.get("team", {}) or {}

        # ====================================================
        # 5.6 MOSTRAR ESTADÍSTICAS
        # ====================================================

        print()
        print("ESTADÍSTICAS ENCONTRADAS")
        print("-" * 70)
        print(f"Equipo: {team.get('name', 'N/A')}")

        if statistics:
            print()
            print(f"Apariciones: {statistics.get('appearances', 'N/A')}")
            print(f"Titularidades: {statistics.get('matchesStarted', 'N/A')}")
            print(f"Minutos: {statistics.get('minutesPlayed', 'N/A')}")
            print(f"Goles: {statistics.get('goals', 'N/A')}")
            print(f"Asistencias: {statistics.get('assists', 'N/A')}")
        else:
            print()
            print("⚠ No se encontraron estadísticas.")

        # ====================================================
        # 5.7 GUARDAR RESULTADO
        # ====================================================

        resultados.append({
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
            "season": TARGET_SEASON,
            "competition": TARGET_COMPETITION,
            "tournament_id": tournament_id,
            "season_id": season_id,
            "team": team,
            "statistics": statistics,
        })

        print()
        print(f"✓ Estadísticas guardadas para {player_name}")

        # ====================================================
        # RESULTADO DEL JUGADOR
        # ====================================================

        print()
        print("-" * 70)
        print(f"RESULTADO: {player_name}")
        print("-" * 70)
        print()
        print("Status: ok")
        print(f"Competición: {TARGET_COMPETITION}")
        print(f"Temporada: {TARGET_SEASON}")
        print(f"Equipo: {team.get('name', 'N/A')}")
        print(f"Goles: {statistics.get('goals', 'N/A')}")
        print(f"Asistencias: {statistics.get('assists', 'N/A')}")
        print(f"Apariciones: {statistics.get('appearances', 'N/A')}")

        print()
        print(f"✓ Jugador terminado: {player_name}")

        # ====================================================
        # CHECKPOINT
        # ====================================================

        if GUARDAR_CHECKPOINT:
            try:
                guardar_resultados_json(
                    resultados,
                    jugadores,
                    output_file
                )

                print()
                print(f"✓ Checkpoint guardado: {output_file}")

            except Exception as e:
                print()
                print("⚠ No se pudo guardar el checkpoint:")
                print(e)

    # ========================================================
    # 6. RESULTADO FINAL
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("RESULTADO FINAL DEL SCRAPER")
    print("=" * 70)
    print()
    print(f"Jugadores procesados: {len(jugadores)}")
    print(f"Registros generados: {len(resultados)}")

    # ========================================================
    # CONTADORES
    # ========================================================

    estados = {}

    for resultado in resultados:
        status = resultado.get("status", "unknown")
        estados[status] = estados.get(status, 0) + 1

    print()
    print("Estados:")

    for status, cantidad in estados.items():
        print(f"  - {status}: {cantidad}")

    # ========================================================
    # 7. GUARDAR JSON
    # ========================================================

    output_file = guardar_resultados_json(
        resultados,
        jugadores,
        output_file
    )

    print()
    print("✓ Resultado guardado:")
    print(output_file)

    print()
    print("=" * 70)
    print("✅ SCRAPING TERMINADO")
    print("=" * 70)


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":
    scrape_historico()