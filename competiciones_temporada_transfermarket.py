import csv
import random
import re
import time
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_CSV = "clubes_con_ids.csv"
OUTPUT_COMPETICIONES = "clubes_competiciones.csv"
OUTPUT_LIGA = "clubes_liga.csv"
ERRORES_CSV = "errores_transfermarkt.csv"
BASE_URL = "https://www.transfermarkt.es"
SEARCH_URL = BASE_URL + "/schnellsuche/ergebnis/schnellsuche?query={}"
MIN_SLEEP = 3
MAX_SLEEP = 6
MAX_RETRIES = 3


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
# LEER CSV
# ============================================================

def leer_clubes():
    clubes = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as archivo:
        reader = csv.DictReader(archivo, skipinitialspace=True)
        columnas = reader.fieldnames or []

        if "club" not in columnas:
            raise ValueError("El CSV no tiene la columna 'club'.")

        if "id_transfermarkt" not in columnas:
            raise ValueError("El CSV no tiene la columna 'id_transfermarkt'.")

        for fila in reader:
            club = (fila["club"] or "").strip()
            club_id = (fila["id_transfermarkt"] or "").strip()

            if not club or not club_id:
                continue

            clubes.append({
                "club": club,
                "id_transfermarkt": club_id
            })

    return clubes


# ============================================================
# AGRUPAR POR ID
# ============================================================

def agrupar_por_id(clubes):
    clubes_por_id = {}
    for club in clubes:
        club_id = club["id_transfermarkt"]
        if club_id not in clubes_por_id:
            clubes_por_id[club_id] = []
        clubes_por_id[club_id].append(club["club"])
    return clubes_por_id


# ============================================================
# RESOLVER SLUG
# ============================================================

def obtener_slug(nombre_club, club_id):
    """
    Busca el club por nombre y obtiene la URL cuyo
    ID de Transfermarkt coincide con el ID solicitado.

    Ejemplo:
        /ac-milan/startseite/verein/5
    """
    query = quote_plus(nombre_club)
    url = SEARCH_URL.format(query)
    print(f"\nBuscando slug para: {nombre_club}")

    response = hacer_request(url)
    if response is None:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    enlaces = soup.find_all("a", href=True)

    for enlace in enlaces:
        href = enlace["href"]
        match = re.search(r"/([^/]+)/startseite/verein/(\d+)", href)
        if not match:
            continue

        slug = match.group(1)
        resultado_id = match.group(2)
        texto = enlace.get_text(" ", strip=True)
        print(f"Resultado encontrado: {texto} → {resultado_id}")

        if resultado_id == str(club_id):
            return slug

    return None


# ============================================================
# OBTENER SLUG POR ID
# ============================================================

def resolver_slug(club_id, nombres):
    """
    Intenta encontrar el slug utilizando los nombres
    asociados al ID.
    """
    for nombre in nombres:
        slug = obtener_slug(nombre, club_id)
        if slug:
            print(f"Slug encontrado para {club_id}: {slug}")
            return slug
        dormir()

    return None


# ============================================================
# EXTRAER LIGA ACTUAL DEL PERFIL DEL CLUB
# ============================================================

def extraer_liga_actual(html, club_id, nombre_club):
    """
    Extrae la competición/liga actual que aparece
    en la cabecera del perfil del club.
    """
    soup = BeautifulSoup(html, "html.parser")
    enlaces = soup.find_all("a", href=True)
    candidatos = []

    for enlace in enlaces:
        texto = enlace.get_text(" ", strip=True)
        if not texto:
            continue
        href = enlace["href"]

        if "/startseite/wettbewerb/" in href:
            candidatos.append(texto)

    if candidatos:
        liga = candidatos[0].strip()
        print(f"Liga actual de {nombre_club}: {liga}")
        return liga

    texto_pagina = soup.get_text(" ", strip=True)
    match = re.search(r"(.+?)\s+Liga:\s+", texto_pagina)

    if match:
        liga = match.group(1).strip()
        palabras = liga.split()
        if len(palabras) <= 8:
            print(f"Liga actual de {nombre_club}: {liga}")
            return liga

    print(f"⚠️ No se pudo encontrar la liga actual de {nombre_club} (ID {club_id})")
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
        tiene_liga = any(x == "liga" for x in encabezados_lower)

        if not (tiene_temporada and tiene_posicion and tiene_liga):
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

            # Fallback
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
# GUARDAR CSV COMPETICIONES
# ============================================================

def guardar_competiciones(resultados):
    with open(OUTPUT_COMPETICIONES, "w", encoding="utf-8", newline="") as archivo:
        campos = ["club_id", "club", "temporada", "competicion", "ronda"]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        for fila in resultados:
            writer.writerow(fila)


# ============================================================
# GUARDAR CSV LIGA
# ============================================================

def guardar_liga(resultados):
    with open(OUTPUT_LIGA, "w", encoding="utf-8", newline="") as archivo:
        campos = ["club_id", "club", "temporada", "liga", "posicion"]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        for fila in resultados:
            writer.writerow(fila)


# ============================================================
# GUARDAR ERRORES
# ============================================================

def guardar_errores(errores):
    with open(ERRORES_CSV, "w", encoding="utf-8", newline="") as archivo:
        campos = ["club_id", "clubes", "error"]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        for fila in errores:
            writer.writerow(fila)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 75)
    print("SCRAPER TRANSFERMARKT")
    print("HISTORIAL DE COMPETICIONES + POSICIONES DE LIGA")
    print("=" * 75)

    clubes = leer_clubes()
    print(f"Filas del CSV: {len(clubes)}")

    clubes_por_id = agrupar_por_id(clubes)
    print(f"IDs únicos: {len(clubes_por_id)}\n")

    for club_id, nombres in clubes_por_id.items():
        print(f"{club_id}: {', '.join(nombres)}")

    todas_competiciones = []
    todas_ligas = []
    errores = []

    ids = list(clubes_por_id.keys())

    for numero, club_id in enumerate(ids, start=1):
        nombres = clubes_por_id[club_id]
        nombre_principal = nombres[0]

        print(f"\n\n{'=' * 75}\n[{numero}/{len(ids)}] {nombre_principal}\nID: {club_id}")

        slug = resolver_slug(club_id, nombres)
        if not slug:
            print("❌ No se pudo encontrar el slug.")
            errores.append({
                "club_id": club_id,
                "clubes": " | ".join(nombres),
                "error": "No se pudo obtener slug"
            })
            continue

        url_perfil = f"{BASE_URL}/{slug}/startseite/verein/{club_id}"
        url_pokal = f"{BASE_URL}/{slug}/pokalhistorie/verein/{club_id}"
        url_liga = f"{BASE_URL}/{slug}/platzierungen/verein/{club_id}"

        print(f"\nPerfil:\n{url_perfil}\nPokalhistorie:\n{url_pokal}\nPlatzierungen:\n{url_liga}")

        # PERFIL DEL CLUB → LIGA ACTUAL
        dormir()
        response_perfil = hacer_request(url_perfil)

        if response_perfil is None:
            print("❌ Error en el perfil del club.")
            errores.append({
                "club_id": club_id,
                "clubes": " | ".join(nombres),
                "error": "Error HTTP perfil"
            })
            continue

        liga_actual = extraer_liga_actual(response_perfil.text, club_id, nombre_principal)
        if not liga_actual:
            print("❌ No se pudo obtener la liga actual.")
            errores.append({
                "club_id": club_id,
                "clubes": " | ".join(nombres),
                "error": "No se pudo extraer la liga actual"
            })
            continue

        print(f"\n🏆 Liga actual: {liga_actual}")

        # POKALHISTORIE
        dormir()
        response_pokal = hacer_request(url_pokal)

        if response_pokal is None:
            print("❌ Error en pokalhistorie.")
            errores.append({
                "club_id": club_id,
                "clubes": " | ".join(nombres),
                "error": "Error HTTP pokalhistorie"
            })
            continue

        competiciones = extraer_pokalhistorie(response_pokal.text, club_id, nombre_principal)
        print(f"Competiciones extraídas: {len(competiciones)}")

        # LIGA
        dormir()
        response_liga = hacer_request(url_liga)

        if response_liga is None:
            print("❌ Error en platzierungen.")
            errores.append({
                "club_id": club_id,
                "clubes": " | ".join(nombres),
                "error": "Error HTTP platzierungen"
            })
            continue

        liga = extraer_platzierungen(response_liga.text, club_id, nombre_principal, liga_actual)
        print(f"Temporadas de liga extraídas: {len(liga)}")

        todas_competiciones.extend(competiciones)
        todas_ligas.extend(liga)

        guardar_competiciones(todas_competiciones)
        guardar_liga(todas_ligas)
        print("✅ Club procesado.")
        dormir()

    guardar_competiciones(todas_competiciones)
    guardar_liga(todas_ligas)
    guardar_errores(errores)

    print(f"\n\n{'=' * 75}\nSCRAPER FINALIZADO\n{'=' * 75}")
    print(f"IDs únicos procesados: {len(ids)}")
    print(f"Registros de competiciones: {len(todas_competiciones)}")
    print(f"Registros de liga: {len(todas_ligas)}")
    print(f"Errores: {len(errores)}")
    print(f"\nCSV competiciones: {OUTPUT_COMPETICIONES}\nCSV liga: {OUTPUT_LIGA}\nErrores: {ERRORES_CSV}")


if __name__ == "__main__":
    main()