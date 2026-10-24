import csv
import random
import re
import time
import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURACIÓN
# ============================================================

OUTPUT_COMPETICIONES = "clubes_competiciones_reintento.csv"
OUTPUT_LIGA = "clubes_liga_reintento.csv"
ERRORES_CSV = "errores_transfermarkt_reintento.csv"
BASE_URL = "https://www.transfermarkt.es"
MIN_SLEEP = 3
MAX_SLEEP = 6
MAX_RETRIES = 3


# ============================================================
# CLUBES A REINTENTAR
# ============================================================

CLUBES_REINTENTO = {
    "2441": {
        "club": "AEK Athens",
        "slug": "aek-athen",
        "liga": "Super League 1"
    },
    "294": {
        "club": "Benfica",
        "slug": "benfica-lissabon",
        "liga": "Liga Portugal"
    },
    "41": {
        "club": "Hamburg",
        "slug": "hamburger-sv",
        "liga": "Bundesliga"
    },
    "237": {
        "club": "Mallorca",
        "slug": "rcd-mallorca",
        "liga": "LaLiga"
    },
}


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,"
        "*/*;q=0.8"
    ),
}

session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# PAUSA
# ============================================================

def dormir():
    segundos = random.uniform(MIN_SLEEP, MAX_SLEEP)
    print(f"Esperando {segundos:.2f} segundos...")
    time.sleep(segundos)


# ============================================================
# REQUEST
# ============================================================

def hacer_request(url):
    for intento in range(1, MAX_RETRIES + 1):
        try:
            print(f"GET: {url}")
            response = session.get(url, timeout=40, allow_redirects=True)
            print(f"HTTP {response.status_code}")
            if response.status_code == 200:
                return response
            print(f"Error HTTP {response.status_code} (intento {intento}/{MAX_RETRIES})")
        except requests.RequestException as e:
            print(f"Error de conexión: {e}")

        if intento < MAX_RETRIES:
            espera = intento * 10
            print(f"Esperando {espera} segundos...")
            time.sleep(espera)
    return None


# ============================================================
# EXTRAER TABLA DE POSICIONES
# ============================================================

def extraer_platzierungen(html, club_id, nombre_club, liga_actual):
    soup = BeautifulSoup(html, "html.parser")
    resultados = []
    tablas = soup.find_all("table")

    for tabla in tablas:
        filas = tabla.find_all("tr")
        if not filas:
            continue

        encabezados = [
            celda.get_text(" ", strip=True)
            for celda in filas[0].find_all(["th", "td"])
        ]
        encabezados_lower = [x.lower() for x in encabezados]

        tiene_temporada = any("temporada" in x for x in encabezados_lower)
        tiene_posicion = any("posición" in x or "posicion" in x for x in encabezados_lower)

        if not (tiene_temporada and tiene_posicion):
            continue

        indice_temporada = next(i for i, x in enumerate(encabezados_lower) if "temporada" in x)
        indice_posicion = next(i for i, x in enumerate(encabezados_lower) if "posición" in x or "posicion" in x)

        for fila in filas[1:]:
            celdas = fila.find_all(["td", "th"])
            valores = [celda.get_text(" ", strip=True) for celda in celdas]

            if len(valores) <= max(indice_temporada, indice_posicion):
                continue

            temporada = valores[indice_temporada]
            posicion = valores[indice_posicion]

            if not re.match(r"^\d{2}/\d{2}$|^\d{4}/\d{2}$", temporada):
                continue

            match_pos = re.search(r"\d+", posicion)
            if not match_pos:
                continue

            posicion = match_pos.group()

            resultados.append({
                "club_id": club_id,
                "club": nombre_club,
                "temporada": temporada,
                "liga": liga_actual,
                "posicion": posicion
            })

        if resultados:
            break

    return resultados


# ============================================================
# EXTRAER POKALHISTORIE
# ============================================================

def extraer_pokalhistorie(html, club_id, nombre_club):
    soup = BeautifulSoup(html, "html.parser")
    resultados = []
    tablas = soup.find_all("table")

    for tabla in tablas:
        filas = tabla.find_all("tr")
        if len(filas) < 2:
            continue

        competicion_actual = None

        for fila in filas:
            clases = fila.get("class", [])

            # FILA DE COMPETICIÓN
            if "bg_blau_20" in clases and "hauptlink" in clases:
                enlace = fila.select_one("td > a")
                if enlace:
                    texto_competicion = enlace.get_text(" ", strip=True)
                    if texto_competicion:
                        competicion_actual = texto_competicion
                        print(f"Competición encontrada: {competicion_actual}")
                continue

            # SI NO TENEMOS COMPETICIÓN
            if not competicion_actual:
                continue

            # CELDAS
            celdas = fila.find_all(["td", "th"])
            if not celdas:
                continue

            valores = [celda.get_text(" ", strip=True) for celda in celdas]
            if len(valores) < 2:
                continue

            # TEMPORADA
            temporada = None
            indice_temporada = None

            for i, valor in enumerate(valores):
                if re.match(r"^\d{2}/\d{2}$|^\d{4}/\d{2}$", valor):
                    temporada = valor
                    indice_temporada = i
                    break

            if temporada is None:
                continue

            # RONDA
            ronda = ""
            if indice_temporada is not None and indice_temporada + 1 < len(valores):
                ronda = valores[indice_temporada + 1]

            # FALLBACK
            if not ronda:
                for i, valor in enumerate(valores):
                    if i == indice_temporada:
                        continue
                    if valor:
                        ronda = valor
                        break

            # GUARDAR
            resultados.append({
                "club_id": club_id,
                "club": nombre_club,
                "temporada": temporada,
                "competicion": competicion_actual,
                "ronda": ronda
            })

    return resultados


# ============================================================
# LEER CSV COMPETICIONES EXISTENTE
# ============================================================

def leer_competiciones():
    try:
        with open(OUTPUT_COMPETICIONES, "r", encoding="utf-8-sig", newline="") as archivo:
            reader = csv.DictReader(archivo)
            return list(reader)
    except FileNotFoundError:
        print(f"⚠️ No existe {OUTPUT_COMPETICIONES}")
        return []


# ============================================================
# LEER CSV LIGA EXISTENTE
# ============================================================

def leer_liga():
    try:
        with open(OUTPUT_LIGA, "r", encoding="utf-8-sig", newline="") as archivo:
            reader = csv.DictReader(archivo)
            return list(reader)
    except FileNotFoundError:
        print(f"⚠️ No existe {OUTPUT_LIGA}")
        return []


# ============================================================
# ELIMINAR REGISTROS DE LOS 4 CLUBES
# ============================================================

def eliminar_clubes_reintento(resultados):
    ids = set(CLUBES_REINTENTO.keys())
    return [
        fila for fila in resultados
        if str(fila.get("club_id", "")) not in ids
    ]


# ============================================================
# GUARDAR COMPETICIONES
# ============================================================

def guardar_competiciones(resultados):
    with open(OUTPUT_COMPETICIONES, "w", encoding="utf-8", newline="") as archivo:
        campos = ["club_id", "club", "temporada", "competicion", "ronda"]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)


# ============================================================
# GUARDAR LIGA
# ============================================================

def guardar_liga(resultados):
    with open(OUTPUT_LIGA, "w", encoding="utf-8", newline="") as archivo:
        campos = ["club_id", "club", "temporada", "liga", "posicion"]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)


# ============================================================
# GUARDAR ERRORES
# ============================================================

def guardar_errores(errores):
    with open(ERRORES_CSV, "w", encoding="utf-8", newline="") as archivo:
        campos = ["club_id", "clubes", "error"]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(errores)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 75)
    print("SCRAPER DE REINTENTO TRANSFERMARKT")
    print("4 CLUBES CON ERRORES")
    print("=" * 75)

    todas_competiciones = leer_competiciones()
    todas_ligas = leer_liga()

    print()
    print(f"Registros competiciones existentes: {len(todas_competiciones)}")
    print(f"Registros liga existentes: {len(todas_ligas)}")

    # ELIMINAR LOS REGISTROS ANTIGUOS DE ESTOS CLUBES
    todas_competiciones = eliminar_clubes_reintento(todas_competiciones)
    todas_ligas = eliminar_clubes_reintento(todas_ligas)

    print()
    print("Registros anteriores de los 4 clubes eliminados para reemplazarlos.")

    errores = []

    # PROCESAR LOS 4 CLUBES
    for numero, (club_id, datos) in enumerate(CLUBES_REINTENTO.items(), start=1):
        nombre_club = datos["club"]
        slug = datos["slug"]
        liga_fija = datos["liga"]

        print()
        print()
        print("=" * 75)
        print(f"[{numero}/{len(CLUBES_REINTENTO)}] {nombre_club}")
        print(f"ID: {club_id}")
        print(f"Slug: {slug}")

        # URLS
        url_perfil = f"{BASE_URL}/{slug}/startseite/verein/{club_id}"
        url_pokal = f"{BASE_URL}/{slug}/pokalhistorie/verein/{club_id}"
        url_liga = f"{BASE_URL}/{slug}/platzierungen/verein/{club_id}"

        print()
        print("Perfil:")
        print(url_perfil)
        print("Pokalhistorie:")
        print(url_pokal)
        print("Platzierungen:")
        print(url_liga)

        # LIGA
        liga_actual = liga_fija

        if liga_actual:
            print()
            print(f"🏆 Liga asignada manualmente: {liga_actual}")
        else:
            print()
            print("Buscando liga en el perfil...")
            dormir()
            response_perfil = hacer_request(url_perfil)

            if response_perfil is not None:
                soup = BeautifulSoup(response_perfil.text, "html.parser")
                candidatos = []

                for enlace in soup.find_all("a", href=True):
                    texto = enlace.get_text(" ", strip=True)
                    href = enlace["href"]

                    if not texto:
                        continue

                    if "/startseite/wettbewerb/" in href:
                        if texto not in candidatos:
                            candidatos.append(texto)

                if candidatos:
                    liga_actual = candidatos[0]
                    print(f"🏆 Liga encontrada: {liga_actual}")

            if not liga_actual:
                print("❌ No se pudo obtener la liga.")
                errores.append({
                    "club_id": club_id,
                    "clubes": nombre_club,
                    "error": "No se pudo extraer la liga actual"
                })

        # POKALHISTORIE
        dormir()
        response_pokal = hacer_request(url_pokal)

        if response_pokal is None:
            print("❌ Error en pokalhistorie.")
            errores.append({
                "club_id": club_id,
                "clubes": nombre_club,
                "error": "Error HTTP pokalhistorie"
            })
            continue

        competiciones = extraer_pokalhistorie(response_pokal.text, club_id, nombre_club)
        print()
        print(f"Competiciones extraídas: {len(competiciones)}")

        # PLATZIERUNGEN
        dormir()
        response_liga = hacer_request(url_liga)

        if response_liga is None:
            print("❌ Error en platzierungen.")
            errores.append({
                "club_id": club_id,
                "clubes": nombre_club,
                "error": "Error HTTP platzierungen"
            })
            continue

        liga = extraer_platzierungen(response_liga.text, club_id, nombre_club, liga_actual or "")
        print()
        print(f"Temporadas de liga extraídas: {len(liga)}")

        # AÑADIR RESULTADOS
        todas_competiciones.extend(competiciones)
        todas_ligas.extend(liga)

        # GUARDAR
        guardar_competiciones(todas_competiciones)
        guardar_liga(todas_ligas)

        print()
        print("✅ Club procesado.")
        dormir()

    # GUARDAR FINAL
    guardar_competiciones(todas_competiciones)
    guardar_liga(todas_ligas)
    guardar_errores(errores)

    # RESUMEN
    print()
    print()
    print("=" * 75)
    print("REINTENTO FINALIZADO")
    print("=" * 75)
    print(f"Clubes procesados: {len(CLUBES_REINTENTO)}")
    print(f"Registros de competiciones: {len(todas_competiciones)}")
    print(f"Registros de liga: {len(todas_ligas)}")
    print(f"Errores: {len(errores)}")
    print()
    print(f"CSV competiciones: {OUTPUT_COMPETICIONES}")
    print(f"CSV liga: {OUTPUT_LIGA}")
    print(f"CSV errores: {ERRORES_CSV}")


if __name__ == "__main__":
    main()