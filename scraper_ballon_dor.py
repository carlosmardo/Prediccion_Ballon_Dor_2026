from playwright.sync_api import sync_playwright
import pandas as pd
import time

def scrape_ballon_dor():
    results = []
    
    # Rango de años del 2000 al 2025, saltando el 2020
    years = [y for y in range(2000, 2026) if y != 2020]

    with sync_playwright() as p:
        # 1. CAMBIO CLAVE: headless=False abre una ventana real para evitar el bloqueo anti-bots
        browser = p.chromium.launch(headless=False)
        
        # 2. CAMBIO CLAVE: Camuflamos el navegador para que parezca un usuario normal
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for year in years:
            url = f"https://ballondor.com/es/all-rankings?year={year}"
            print(f"Scrapeando año {year}...")
            
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                
                # Esperamos a que aparezcan las tarjetas
                page.wait_for_selector("div.flex-1.min-w-0", timeout=15000)
                
                player_cards = page.locator("div.flex-1.min-w-0").all()
                
                for card in player_cards:
                    # 1. Posición
                    rank_loc = card.locator("span.text-2xl.font-bold")
                    rank = rank_loc.inner_text().strip() if rank_loc.count() > 0 else "N/A"
                    
                    if rank.lower() == "ganador":
                        rank = "1"
                    
                    # 2. Nombre
                    name_locs = card.locator("span.text-primary").all()
                    name_parts = [n.inner_text().strip() for n in name_locs if n.inner_text().strip()]
                    name = "_".join(name_parts)
                    
                    # 3. Club
                    club_loc = card.locator("span.text-white.font-bold.text-lg")
                    club = club_loc.inner_text().strip() if club_loc.count() > 0 else "N/A"
                    
                    # 4. Nacionalidad
                    nat_loc = card.locator("span.text-white.font-normal.text-lg")
                    nationality = nat_loc.inner_text().strip() if nat_loc.count() > 0 else "N/A"
                    
                    if name:
                        results.append({
                            'Año': year,
                            'Posición': rank,
                            'Nombre': name,
                            'Club': club,
                            'Nacionalidad': nationality
                        })
                        
            except Exception as e:
                print(f"Error procesando el año {year}: {e}")
                
            # Pausa de 5 segundos para no hacer demasiadas peticiones de golpe
            time.sleep(5) 

        browser.close()

    if results:
        df = pd.DataFrame(results)
        df.to_csv('ballon_dor_ranking.csv', index=False, encoding='utf-8-sig')
        print(f"\n¡Completado! {len(df)} registros guardados exitosamente en 'ballon_dor_ranking.csv'.")
    else:
        print("No se encontraron resultados.")

if __name__ == "__main__":
    scrape_ballon_dor()