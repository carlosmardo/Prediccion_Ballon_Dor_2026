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

INPUT_CSV = "nacionalidades_unicas.csv"
OUTPUT_COMPETICIONES = "selecciones_competiciones.csv"
ERRORES_CSV = "errores_selecciones_transfermarkt.csv"
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
# LEER CSV DE PAÍSES
# ============================================================

def leer_paises():
    paises = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as archivo:
        reader = csv.DictReader(archivo, skipinitialspace=True)
        columnas = reader.fieldnames or []
        columna_pais = None

        for columna in columnas:
            if columna.lower().strip() in ("pais", "país"):
                columna_pais = columna
                break

        if columna_pais is None:
            raise ValueError("El CSV debe tener una columna 'pais' o 'País'.")

        for fila in reader:
            pais = (fila[columna_pais] or "").strip()
            if not pais:
                continue
            paises.append(pais)

    return paises


# ============================================================
# ELIMINAR DUPLICADOS
# ============================================================

def quitar_duplicados(paises):
    resultado = []
    vistos = set()

    for pais in paises:
        clave = pais.strip().lower()
        if clave in vistos:
            continue
        vistos.add(clave)
        resultado.append(pais)

    return resultado


# ============================================================
# NORMALIZAR TEXTO
# ============================================================

def normalizar_texto(texto):
    texto = texto.strip().lower()
    texto = texto.replace("-", " ")
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


# ============================================================
# COMPROBAR SI ES EQUIPO NO SENIOR
# ============================================================

def es_equipo_no_senior(texto, slug):
    texto_normalizado = normalizar_texto(texto)
    slug_normalizado = normalizar_texto(slug)
    contenido = texto_normalizado + " " + slug_normalizado

    patrones = [
        r"\bsub\s*23\b",
        r"\bsub\s*21\b",
        r"\bsub\s*20\b",
        r"\bsub\s*19\b",
        r"\bsub\s*18\b",
        r"\bsub\s*17\b",
        r"\bsub\s*16\b",
        r"\bsub\s*15\b",
        r"\bu23\b",
        r"\bu21\b",
        r"\bu20\b",
        r"\bu19\b",
        r"\bu18\b",
        r"\bu17\b",
        r"\bu16\b",
        r"\bu15\b",
        r"\bolimpic",
        r"\bolimpico",
        r"\bolímpico",
        r"\bolympic",
        r"\bolympia",
        r"\bfemenin",
        r"\bwomen\b",
        r"\bwomens\b",
        r"\bwomen's\b",
        r"\bfrauen\b",
    ]

    for patron in patrones:
        if re.search(patron, contenido, flags=re.IGNORECASE):
            return True

    return False


# ============================================================
# COMPROBAR SI UNA PÁGINA ES DE UNA SELECCIÓN NACIONAL
# ============================================================

def verificar_seleccion(slug, seleccion_id):
    """
    Comprueba la página /startseite/verein/ID para evitar
    confundir clubes con selecciones.
    """
    url = f"{BASE_URL}/{slug}/startseite/verein/{seleccion_id}"
    print(f"\nVerificando candidato:\n{url}")

    response = hacer_request(url)
    if response is None:
        print("❌ No se pudo verificar el candidato.")
        return False

    soup = BeautifulSoup(response.text, "html.parser")
    texto_pagina = soup.get_text(" ", strip=True)
    texto_normalizado = normalizar_texto(texto_pagina)

    indicadores_fuertes = [
        "ranking mundial fifa",
        "fifa ranking",
        "ranking fifa",
        "confederacion",
        "seleccion nacional",
        "nationalmannschaft",
        "national team",
    ]

    encontrados = []
    for indicador in indicadores_fuertes:
        indicador_normalizado = normalizar_texto(indicador)
        if indicador_normalizado in texto_normalizado:
            encontrados.append(indicador)

    print(f"Indicadores encontrados: {len(encontrados)}")
    if encontrados:
        for indicador in encontrados:
            print(f"  ✓ {indicador}")
        print("✅ El candidato parece ser una selección.")
        return True

    enlaces_national = soup.find_all("a", href=True)
    for enlace in enlaces_national:
        href = enlace.get("href", "").lower()
        texto = enlace.get_text(" ", strip=True).lower()

        if "nationalspieler" in href or "nationalteam" in href or "nationalmannschaft" in href:
            print("✅ Se encontró sección de jugadores de la selección.")
            return True

        if "jugadores de la selección" in texto or "jugadores selección" in texto:
            print("✅ Se encontró información de jugadores de selección.")
            return True

    print("❌ El candidato no parece ser una selección.")
    return False


# ============================================================
# EXTRAER CANDIDATOS DE LA BÚSQUEDA
# ============================================================

def obtener_candidatos_busqueda(soup):
    candidatos = []
    enlaces = soup.find_all("a", href=True)

    for enlace in enlaces:
        href = enlace["href"]
        match = re.search(r"^/([^/]+)/startseite/verein/(\d+)", href)
        if not match:
            continue

        slug = match.group(1)
        seleccion_id = match.group(2)
        texto = enlace.get_text(" ", strip=True)

        if not texto:
            texto = enlace.get("title") or slug

        if es_equipo_no_senior(texto, slug):
            print(f"Descartado no senior: {texto} → ID {seleccion_id}")
            continue

        existe = False
        for candidato in candidatos:
            if candidato["id"] == seleccion_id and candidato["slug"] == slug:
                existe = True
                break

        if existe:
            continue

        candidatos.append({
            "slug": slug,
            "id": seleccion_id,
            "texto": texto,
            "href": href
        })

    return candidatos


# ============================================================
# RESOLVER SELECCIÓN
# ============================================================

def obtener_seleccion(nombre_pais):
    """
    Busca la selección absoluta correspondiente al país.
    """
    query = quote_plus(nombre_pais)
    url = SEARCH_URL.format(query)
    print(f"\nBuscando selección: {nombre_pais}")

    response = hacer_request(url)
    if response is None:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    candidatos = obtener_candidatos_busqueda(soup)

    print(f"\nCandidatos de equipo encontrados: {len(candidatos)}")
    for candidato in candidatos:
        print(f"  {candidato['texto']} → ID {candidato['id']} → slug {candidato['slug']}")

    if not candidatos:
        print("\n❌ No se encontraron candidatos de equipo.")
        return None

    nombre_normalizado = normalizar_texto(nombre_pais)
    candidatos_ordenados = []

    for candidato in candidatos:
        texto_normalizado = normalizar_texto(candidato["texto"])
        slug_normalizado = normalizar_texto(candidato["slug"])
        puntuacion = 0

        if texto_normalizado == nombre_normalizado:
            puntuacion += 100
        if slug_normalizado == nombre_normalizado:
            puntuacion += 90
        if nombre_normalizado and nombre_normalizado in texto_normalizado:
            puntuacion += 50
        if nombre_normalizado and nombre_normalizado in slug_normalizado:
            puntuacion += 40

        candidato_copia = candidato.copy()
        candidato_copia["puntuacion"] = puntuacion
        candidatos_ordenados.append(candidato_copia)

    candidatos_ordenados.sort(key=lambda x: x["puntuacion"], reverse=True)

    print("\nCandidatos ordenados:")
    for candidato in candidatos_ordenados:
        print(f"  [{candidato['puntuacion']}] {candidato['texto']} → {candidato['id']}")

    seleccion_valida = []
    for candidato in candidatos_ordenados:
        print(f"\n{'-' * 60}\nVerificando: {candidato['texto']}\nID: {candidato['id']}")

        es_seleccion = verificar_seleccion(candidato["slug"], candidato["id"])
        if not es_seleccion:
            print(f"❌ Descartado: {candidato['texto']} no es una selección.")
            continue

        print(f"✅ Selección válida: {candidato['texto']} → ID {candidato['id']}")
        seleccion_valida.append(candidato)

        if candidato["puntuacion"] >= 90:
            break

    if not seleccion_valida:
        print("\n❌ Ningún candidato pudo verificarse como selección nacional.")
        return None

    seleccion_valida.sort(key=lambda x: x["puntuacion"], reverse=True)
    mejor = seleccion_valida[0]

    if len(seleccion_valida) > 1:
        segundo = seleccion_valida[1]
        if mejor["puntuacion"] == segundo["puntuacion"]:
            print("\n⚠️ Hay varias selecciones válidas con la misma puntuación.")
            print("No se elegirá una arbitrariamente.")
            for candidato in seleccion_valida:
                print(f"  {candidato['texto']} → ID {candidato['id']}")
            return None

    print(f"\n{'=' * 48}\n✅ SELECCIÓN IDENTIFICADA\nPaís CSV: {nombre_pais}\nNombre Transfermarkt: {mejor['texto']}\nID: {mejor['id']}\nSlug: {mejor['slug']}\n{'=' * 48}")

    return {
        "slug": mejor["slug"],
        "id": mejor["id"]
    }


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

            if not competicion_actual:
                continue

            celdas = fila.find_all(["td", "th"])
            if not celdas:
                continue

            valores = [celda.get_text(" ", strip=True) for celda in celdas]
            if len(valores) < 2:
                continue

            temporada = None
            indice_temporada = None

            for i, valor in enumerate(valores):
                if re.match(r"^\d{2}/\d{2}$|^\d{4}/\d{2}$", valor):
                    temporada = valor
                    indice_temporada = i
                    break

            if temporada is None:
                continue

            ronda = ""
            if indice_temporada is not None and indice_temporada + 1 < len(valores):
                ronda = valores[indice_temporada + 1]

            if not ronda:
                for i, valor in enumerate(valores):
                    if i == indice_temporada:
                        continue
                    if valor:
                        ronda = valor
                        break

            resultados.append({
                "club_id": club_id,
                "club": nombre_club,
                "temporada": temporada,
                "competicion": competicion_actual,
                "ronda": ronda
            })

    return resultados


# ============================================================
# GUARDAR COMPETICIONES
# ============================================================

def guardar_competiciones(resultados):
    with open(OUTPUT_COMPETICIONES, "w", encoding="utf-8", newline="") as archivo:
        campos = ["club_id", "club", "temporada", "competicion", "ronda"]
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
    print("SELECCIONES NACIONALES - HISTORIAL DE COMPETICIONES")
    print("=" * 75)

    paises = leer_paises()
    print(f"Países encontrados: {len(paises)}")

    paises = quitar_duplicados(paises)
    print(f"Países únicos: {len(paises)}\n")

    todas_competiciones = []
    errores = []

    for numero, pais in enumerate(paises, start=1):
        print(f"\n\n{'=' * 75}\n[{numero}/{len(paises)}] {pais}")

        seleccion = obtener_seleccion(pais)
        if not seleccion:
            print("\n❌ No se pudo encontrar la selección.")
            errores.append({
                "club_id": "",
                "clubes": pais,
                "error": "No se pudo obtener slug/ID"
            })
            guardar_competiciones(todas_competiciones)
            guardar_errores(errores)
            dormir()
            continue

        slug = seleccion["slug"]
        club_id = seleccion["id"]

        url_pokal = f"{BASE_URL}/{slug}/pokalhistorie/verein/{club_id}"
        print(f"\nPokalhistorie:\n{url_pokal}")

        dormir()
        response_pokal = hacer_request(url_pokal)

        if response_pokal is None:
            print("\n❌ Error en pokalhistorie.")
            errores.append({
                "club_id": club_id,
                "clubes": pais,
                "error": "Error HTTP pokalhistorie"
            })
            guardar_competiciones(todas_competiciones)
            guardar_errores(errores)
            dormir()
            continue

        competiciones = extraer_pokalhistorie(response_pokal.text, club_id, pais)
        print(f"\nCompeticiones extraídas: {len(competiciones)}")

        if not competiciones:
            print("\n⚠️ No se encontraron competiciones.")
            errores.append({
                "club_id": club_id,
                "clubes": pais,
                "error": "Pokalhistorie sin registros"
            })
        else:
            todas_competiciones.extend(competiciones)
            print("\n✅ Selección procesada.")

        guardar_competiciones(todas_competiciones)
        guardar_errores(errores)
        dormir()

    guardar_competiciones(todas_competiciones)
    guardar_errores(errores)

    print(f"\n\n{'=' * 75}\nSCRAPER FINALIZADO\n{'=' * 75}")
    print(f"Países procesados: {len(paises)}")
    print(f"Registros de competiciones: {len(todas_competiciones)}")
    print(f"Errores: {len(errores)}")
    print(f"\nCSV competiciones: {OUTPUT_COMPETICIONES}\nErrores: {ERRORES_CSV}")


if __name__ == "__main__":
    main()