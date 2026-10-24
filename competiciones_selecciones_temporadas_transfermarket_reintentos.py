import csv
import random
import re
import time
import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_CSV = "csv_fuentes/nacionalidades_unicas_reintento.csv"
OUTPUT_COMPETICIONES = "selecciones_competiciones_reintento.csv"
ERRORES_CSV = "errores_selecciones_transfermarkt.csv"
BASE_URL = "https://www.transfermarkt.es"
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
# LEER CSV DE PAÍSES, IDS Y SLUGS
# ============================================================

def leer_paises():
    paises = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as archivo:
        reader = csv.DictReader(archivo, skipinitialspace=True)
        columnas = reader.fieldnames or []

        columna_pais = None
        columna_id = None
        columna_slug = None

        for columna in columnas:
            nombre_columna = columna.strip().lower()
            if nombre_columna in ("pais", "país"):
                columna_pais = columna
            elif nombre_columna == "id":
                columna_id = columna
            elif nombre_columna == "slug":
                columna_slug = columna

        if columna_pais is None:
            raise ValueError("El CSV debe tener una columna 'Pais' o 'País'.")
        if columna_id is None:
            raise ValueError("El CSV debe tener una columna 'id'.")
        if columna_slug is None:
            raise ValueError("El CSV debe tener una columna 'slug'.")

        for fila in reader:
            pais = (fila.get(columna_pais, "") or "").strip()
            seleccion_id = (fila.get(columna_id, "") or "").strip()
            slug = (fila.get(columna_slug, "") or "").strip()

            if not pais:
                continue
            if not seleccion_id:
                print(f"⚠️ Sin ID: {pais}")
                continue
            if not slug:
                print(f"⚠️ Sin slug: {pais}")
                continue

            paises.append({
                "pais": pais,
                "id": seleccion_id,
                "slug": slug
            })

    return paises


# ============================================================
# ELIMINAR DUPLICADOS
# ============================================================

def quitar_duplicados(paises):
    resultado = []
    vistos = set()

    for pais in paises:
        clave = (
            pais["pais"].strip().lower(),
            pais["id"].strip(),
            pais["slug"].strip().lower()
        )
        if clave in vistos:
            continue
        vistos.add(clave)
        resultado.append(pais)

    return resultado


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
    print(f"\nPaíses encontrados: {len(paises)}")

    paises = quitar_duplicados(paises)
    print(f"Países únicos: {len(paises)}\n")

    for pais in paises:
        print(f"  {pais['pais']} → ID {pais['id']} → slug {pais['slug']}")

    todas_competiciones = []
    errores = []

    for numero, pais_info in enumerate(paises, start=1):
        pais = pais_info["pais"]
        club_id = pais_info["id"]
        slug = pais_info["slug"]

        print(f"\n\n{'=' * 75}\n[{numero}/{len(paises)}] {pais}\nID Transfermarkt: {club_id}\nSlug Transfermarkt: {slug}\n{'=' * 75}")

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

            for registro in competiciones[:5]:
                print(f"  {registro['temporada']} | {registro['competicion']} | {registro['ronda']}")

            if len(competiciones) > 5:
                print(f"  ... y {len(competiciones) - 5} registros más")

        guardar_competiciones(todas_competiciones)
        guardar_errores(errores)
        print("\n💾 CSV actualizado.")

        if numero < len(paises):
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