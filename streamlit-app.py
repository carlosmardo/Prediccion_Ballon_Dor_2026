from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Balón de Oro · Explorador ML", page_icon="⚽", layout="wide")

SEASONS = ["2020/21", "2021/22", "2022/23", "2023/24", "2024/25", "2025/26"]
DATASET_PATH = Path("data/dataset_ml_base.csv")
ML_DIR = Path("data/ml")

# En la tabla se muestran primero las probabilidades y después los scores.
RANK_COLS = [
    "temporada", "posicion_predicha", "posicion_real", "nombre", "club", "posicion",
    "prob_duelo_contra_lider", "prob_duelo_contra_anterior",
    "score_ensamble", "score_modelo_posicional", "score_modelo_global",
]

# Mismas estadísticas base que utiliza el modelo, más las cuatro excepciones pedidas.
ADDITIVE_STATS = [
    "goalsAssistsSum", "bigChancesCreated", "bigChancesMissed", "keyPasses",
    "successfulDribbles", "accuratePasses", "totalAttemptAssist", "aerialDuelsWon",
    "totalDuelsWon", "tacklesWon", "interceptions", "outfielderBlocks",
    "ballRecovery", "dispossessed", "dribbledPast", "clearances",
    "errorLeadToGoal", "errorLeadToShot", "saves", "goalsPrevented",
    "goalsConceded", "highClaims", "punches", "successfulRunsOut",
    "cleanSheet", "ownGoals", "penaltyConceded", "possessionWonAttThird",
]
PERCENT_PAIRS = {
    "accuratePassesPercentage": ("accuratePasses", "totalPasses"),
    "successfulDribblesPercentage": ("successfulDribbles", "totalDribbles"),
    "aerialDuelsWonPercentage": ("aerialDuelsWon", "totalAerialDuels"),
    "totalDuelsWonPercentage": ("totalDuelsWon", "totalDuels"),
    "tacklesWonPercentage": ("tacklesWon", "totalTackles"),
    "goalConversionPercentage": ("goals", "totalShots"),
}
EXTRA_STATS = ["goals", "assists", "appearances", "minutesPlayed"]
MODEL_STATS = ADDITIVE_STATS + list(PERCENT_PAIRS) + ["goalsAssistsPerMins"] + EXTRA_STATS
GK_ALLOWED = {
    "saves", "goalsPrevented", "goalsConceded", "highClaims", "punches",
    "successfulRunsOut", "cleanSheet", "ownGoals", "penaltyConceded",
    "errorLeadToGoal", "errorLeadToShot", "accuratePassesPercentage",
}
GK_EXCLUSIVE = {"saves", "goalsPrevented", "highClaims", "punches", "successfulRunsOut"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.35rem; padding-bottom: 2rem;}
      html, body, [class*="css"] {font-size: 1.04rem;}
      button[data-baseweb="tab"] {font-size: 1.08rem; padding: 0.65rem 1.05rem; min-height: 3rem;}
      button[data-baseweb="tab"] p {font-size: 1.08rem;}
      [data-testid="stMetricValue"] {font-size: 1.55rem;}
      .small-note {opacity: .75; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("⚽ Balón de Oro · Explorador del modelo")
st.caption("Predicciones históricas, scores del ensamble, estadísticas por competición y comparación de jugadores.")


def clean_cols(df):
    df = df.copy()
    df.columns = [
        re.sub(r"^&#x[0-9a-fA-F]+;", "", str(c).replace("**", "").strip()).strip()
        for c in df.columns
    ]
    return df.loc[:, ~pd.Index(df.columns).duplicated()]


def season_norm(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("-", "/").replace("_", "/")
    m = re.match(r"^(\d{4})/(\d{2}|\d{4})$", s)
    return f"{m.group(1)}/{m.group(2)[-2:]}" if m else s


def read_path(path):
    if not path.exists():
        return None
    try:
        try:
            return clean_cols(pd.read_csv(path, low_memory=False))
        except UnicodeDecodeError:
            return clean_cols(pd.read_csv(path, encoding="latin-1", low_memory=False))
    except Exception as e:
        st.warning(f"No se pudo leer `{path}`: {e}")
        return None


def season_from_name(name):
    m = re.search(r"(20\d{2})[-_](\d{2}|20\d{2})", name)
    return season_norm(f"{m.group(1)}/{m.group(2)}") if m else None


def norm_ranking(df, season):
    df = clean_cols(df)
    aliases = {
        "posición_predicha": "posicion_predicha", "posición_real": "posicion_real",
        "jugador": "nombre", "player": "nombre", "equipo": "club", "posición": "posicion",
    }
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    if "temporada" not in df:
        df["temporada"] = season
    df["temporada"] = df["temporada"].map(season_norm).replace("", season)
    for c in RANK_COLS:
        if c not in df:
            df[c] = np.nan
    for c in ["posicion_predicha", "posicion_real", "score_ensamble",
              "score_modelo_posicional", "score_modelo_global"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("posicion_predicha", na_position="last")


def fold_text(value):
    """Búsqueda flexible: ignora mayúsculas, tildes y espacios repetidos."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).casefold().strip()


def model_stat_columns(rows):
    """Estadísticas disponibles para el jugador, respetando el filtrado por posición del modelo."""
    available = [c for c in MODEL_STATS if c in rows.columns]
    if "goalsAssistsPerMins" not in available and "minutesPlayed" in rows.columns and (
        "goalsAssistsSum" in rows.columns or {"goals", "assists"}.issubset(rows.columns)
    ):
        available.append("goalsAssistsPerMins")

    positions = set(rows.get("posicion", pd.Series(dtype=str)).dropna().astype(str).str.strip())
    if positions and positions == {"Goalkeeper"}:
        allowed = GK_ALLOWED | {"goals", "assists", "appearances", "minutesPlayed", "goalsAssistsSum", "goalsAssistsPerMins"}
        available = [c for c in available if c in allowed]
    elif positions and "Goalkeeper" not in positions:
        available = [c for c in available if c not in GK_EXCLUSIVE]
    return available


def player_rows(dataset, season, player):
    if dataset.empty or not {"temporada", "nombre"}.issubset(dataset.columns):
        return pd.DataFrame()
    d = dataset[dataset["temporada"].map(season_norm) == season].copy()
    name = str(player.get("nombre", "")).strip()
    club = str(player.get("club", "")).strip()
    d = d[d["nombre"].map(fold_text) == fold_text(name)]
    if club and fold_text(club) not in {"nan", "none", ""} and "club" in d:
        dc = d[d["club"].map(fold_text) == fold_text(club)]
        if not dc.empty:
            d = dc
    return d


def number_series(rows, col):
    if col not in rows:
        return pd.Series(dtype=float)
    return pd.to_numeric(rows[col], errors="coerce")


def ratio_from_totals(rows, num_col, den_col):
    if num_col not in rows or den_col not in rows:
        return np.nan
    num = number_series(rows, num_col).sum(min_count=1)
    den = number_series(rows, den_col).sum(min_count=1)
    if pd.isna(num) or pd.isna(den):
        return np.nan
    if den > 0:
        return 100.0 * num / den
    if den == 0:
        return 0.0
    return np.nan


def infer_denominator(rows, numerator_col, percentage_col):
    """Replica la inferencia de denominadores auxiliares del pipeline del modelo."""
    if numerator_col not in rows or percentage_col not in rows:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    numerator = number_series(rows, numerator_col)
    percentage = number_series(rows, percentage_col)
    valid = percentage.dropna()
    if not valid.empty and valid.max() <= 1.0:
        percentage = percentage * 100.0
    denominator = numerator / (percentage / 100.0).where(percentage > 0)
    zeros = numerator.eq(0) & percentage.eq(0) & numerator.notna() & percentage.notna()
    return denominator.mask(zeros, 0.0)


def prepare_percent_denominators(rows):
    rows = rows.copy()
    # El modelo rellena los denominadores que faltan a partir del numerador y porcentaje.
    for pct, (num, den) in PERCENT_PAIRS.items():
        if num not in rows or pct not in rows:
            continue
        inferred = infer_denominator(rows, num, pct)
        if den not in rows:
            rows[den] = inferred
        else:
            rows[den] = number_series(rows, den).fillna(inferred)
    return rows


def aggregate_stats(rows, include_competition_result=False):
    """Agrega las estadísticas con las mismas reglas del pipeline del modelo."""
    if rows.empty:
        return {}
    rows = prepare_percent_denominators(rows)
    vals = {}
    for col in model_stat_columns(rows):
        if col in ADDITIVE_STATS or col in {"goals", "assists", "appearances", "minutesPlayed"}:
            vals[col] = number_series(rows, col).sum(min_count=1)
        elif col in PERCENT_PAIRS:
            num, den = PERCENT_PAIRS[col]
            vals[col] = ratio_from_totals(rows, num, den)
        elif col == "goalsAssistsPerMins":
            ga_col = "goalsAssistsSum"
            if ga_col in rows and "minutesPlayed" in rows:
                ga = number_series(rows, ga_col).sum(min_count=1)
                minutes = number_series(rows, "minutesPlayed").sum(min_count=1)
                vals[col] = ga / minutes if pd.notna(ga) and pd.notna(minutes) and minutes != 0 else np.nan
            elif {"goals", "assists", "minutesPlayed"}.issubset(rows.columns):
                ga = number_series(rows, "goals").sum(min_count=1) + number_series(rows, "assists").sum(min_count=1)
                minutes = number_series(rows, "minutesPlayed").sum(min_count=1)
                vals[col] = ga / minutes if pd.notna(ga) and pd.notna(minutes) and minutes != 0 else np.nan
        # Los resultados se presentan como texto aparte, no como estadística numérica.
    for c in ["temporada", "nombre", "club", "posicion", "nacionalidad"]:
        if c in rows and rows[c].notna().any():
            vals[c] = rows[c].dropna().iloc[0]
    if include_competition_result and "resultado_competicion" in rows:
        results = rows[["competicion", "resultado_competicion"]].dropna(subset=["resultado_competicion"]).drop_duplicates()
        vals["resultado_competicion"] = " · ".join(
            f"{r.competicion}: {r.resultado_competicion}" for r in results.itertuples(index=False)
        ) or "—"
    return vals


def display_stat_table(values, stats, title, key, result_text=None):
    st.markdown(f"**{title}**")
    data = []
    for col in stats:
        v = values.get(col, np.nan)
        if pd.notna(v):
            data.append({"Estadística": col, "Valor": v})
    if result_text:
        data.append({"Estadística": "resultado_competicion", "Valor": result_text})
    if not data:
        st.info("No hay estadísticas disponibles para mostrar.")
        return
    table = pd.DataFrame(data)
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        height=min(620, 42 + len(table) * 34),
        column_config={
            "Estadística": st.column_config.TextColumn(width="large"),
            "Valor": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "Descargar estadísticas CSV", table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"estadisticas_{re.sub(r'\\W+', '_', title)}.csv",
        mime="text/csv", key=key,
    )


def result_for_competition(rows):
    if "resultado_competicion" not in rows:
        return None
    values = rows["resultado_competicion"].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return " · ".join(dict.fromkeys(values.tolist())) if not values.empty else None


def player_detail(dataset, season, player, prefix):
    rows = player_rows(dataset, season, player)
    if rows.empty:
        st.info(f"No se encontraron estadísticas para {player.get('nombre', 'este jugador')} en {season}. Comprueba la temporada y el nombre del dataset.")
        return
    stats = model_stat_columns(rows)
    global_values = aggregate_stats(rows, include_competition_result=True)
    comps = rows["competicion"].fillna("Sin competición").astype(str).drop_duplicates().tolist() if "competicion" in rows else ["Sin competición"]
    comp_tabs = st.tabs(["🌐 Global"] + comps)
    with comp_tabs[0]:
        st.caption("Se muestran las estadísticas del modelo y las cuatro adicionales solicitadas. Los porcentajes globales se recalculan sumando numeradores y denominadores.")
        display_stat_table(global_values, stats, "Estadísticas globales", f"{prefix}_global",
                           global_values.get("resultado_competicion"))
    for i, comp in enumerate(comps):
        with comp_tabs[i + 1]:
            sub = rows[rows["competicion"].fillna("Sin competición").astype(str) == comp]
            vals = aggregate_stats(sub)
            display_stat_table(vals, stats, f"Estadísticas · {comp}", f"{prefix}_comp_{i}",
                               result_for_competition(sub))


def format_probability(value):
    if pd.isna(value) or str(value).strip() == "":
        return "—"
    try:
        raw = str(value).strip()
        n = float(raw.replace("%", "").replace(",", "."))
        return f"{n:.2f}%" if "%" in raw or n > 1 else f"{n * 100:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def ranking_tab(season, ranking, dataset):
    if ranking is None or ranking.empty:
        st.warning(f"No hay ranking cargado para {season}. Coloca el CSV de predicción correspondiente en `data/ml/`.")
        return
    d = norm_ranking(ranking, season)
    if d["temporada"].nunique(dropna=True) > 1:
        d = d[d["temporada"].map(season_norm) == season]

    q = st.text_input("Buscar jugador o club", key=f"search_{season}", placeholder="Ej. Cubarsi, Messi, Real Madrid…")
    filtered = d.copy()
    if q.strip():
        query = fold_text(q)
        mask = pd.Series(False, index=filtered.index)
        for col in ["nombre", "club", "posicion"]:
            if col in filtered:
                mask |= filtered[col].map(fold_text).str.contains(re.escape(query), regex=True, na=False)
        filtered = filtered[mask]

    shown = filtered[[x for x in RANK_COLS if x in filtered]].copy()
    for col in ["prob_duelo_contra_lider", "prob_duelo_contra_anterior"]:
        if col in shown:
            shown[col] = shown[col].map(format_probability)
    st.dataframe(
        shown, use_container_width=True, hide_index=True,
        height=min(660, 46 + len(shown) * 36),
        column_config={
            "posicion_predicha": st.column_config.NumberColumn("Posición predicha", format="%d"),
            "posicion_real": st.column_config.NumberColumn("Posición real", format="%.1f"),
            "prob_duelo_contra_lider": st.column_config.TextColumn("Prob. contra líder"),
            "prob_duelo_contra_anterior": st.column_config.TextColumn("Prob. contra anterior"),
            "score_ensamble": st.column_config.NumberColumn("Score ensamble", format="%.4f"),
            "score_modelo_posicional": st.column_config.NumberColumn("Score posicional", format="%.4f"),
            "score_modelo_global": st.column_config.NumberColumn("Score global", format="%.4f"),
        },
    )
    st.download_button(
        f"Descargar ranking {season} CSV", d.to_csv(index=False).encode("utf-8-sig"),
        f"ranking_{season.replace('/', '-')}.csv", "text/csv", key=f"dl_rank_{season}",
    )
    st.divider()
    st.markdown("#### Ficha individual")
    names = d["nombre"].dropna().astype(str).drop_duplicates().tolist()
    if not names:
        st.info("El ranking no contiene nombres de jugadores.")
        return
    # El selector también admite búsqueda sin tildes porque las opciones están normalizadas para filtrar.
    name = st.selectbox("Selecciona un jugador", names, key=f"player_{season}")
    p = d[d.nombre.astype(str) == name].iloc[0]
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Posición predicha", "—" if pd.isna(p.posicion_predicha) else str(int(p.posicion_predicha)))
    x2.metric("Posición real", "—" if pd.isna(p.posicion_real) else str(p.posicion_real))
    x3.metric("Club", str(p.club) if pd.notna(p.club) else "—")
    x4.metric("Posición", str(p.posicion) if pd.notna(p.posicion) else "—")

    # Scores y probabilidades en una única línea.
    metrics = [
        ("score_ensamble", "Score ensamble", "score"),
        ("score_modelo_posicional", "Score posicional", "score"),
        ("score_modelo_global", "Score global", "score"),
        ("prob_duelo_contra_lider", "Prob. contra líder", "prob"),
        ("prob_duelo_contra_anterior", "Prob. contra anterior", "prob"),
    ]
    boxes = st.columns(5)
    for (col, label, kind), box in zip(metrics, boxes):
        val = p.get(col)
        if kind == "score":
            num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            shown_val = "—" if pd.isna(num) else f"{num:.4f}"
        else:
            shown_val = format_probability(val)
        box.metric(label, shown_val)

    with st.expander("Ver estadísticas del modelo por competición y global", expanded=True):
        player_detail(dataset, season, p, f"{season}_{re.sub(r'\\W+', '_', name)}")


def comparison_tab(dataset):
    st.subheader("Comparador de jugadores")
    if dataset.empty or not {"temporada", "nombre"}.issubset(dataset.columns):
        st.warning("No se encuentra `data/dataset_ml_base.csv` para comparar estadísticas.")
        return
    available = {season_norm(x) for x in dataset.temporada.dropna()}
    season_options = [s for s in SEASONS if s in available] or sorted(available)
    default = "2025/26" if "2025/26" in season_options else season_options[-1]
    season = st.selectbox("Temporada de estadísticas", season_options, index=season_options.index(default), key="compare_season")
    sd = dataset[dataset.temporada.map(season_norm) == season].copy()
    if sd.empty:
        st.info("No hay datos para esta temporada.")
        return
    pair = sd[["nombre", "club"]].dropna(subset=["nombre"]).drop_duplicates().copy()
    pair["label"] = pair.nombre.astype(str) + " — " + pair.club.fillna("Club desconocido").astype(str)
    options = pair.label.tolist()
    chosen = st.multiselect("Elige dos o más jugadores", options, default=options[:2], key=f"compare_players_{season}")
    if len(chosen) < 2:
        st.info("Selecciona al menos dos jugadores.")
        return
    per_player, output_rows = {}, []
    for _, p in pair[pair.label.isin(chosen)].iterrows():
        matched = player_rows(dataset, season, {"nombre": p.nombre, "club": p.club})
        if matched.empty:
            continue
        vals = aggregate_stats(matched, include_competition_result=True)
        per_player[p.label] = matched
        item = {"Jugador": p.label}
        for col in model_stat_columns(matched):
            if col in vals and pd.notna(vals[col]):
                item[col] = vals[col]
        output_rows.append(item)
    if len(output_rows) < 2:
        st.info("No se pudieron preparar estadísticas para al menos dos jugadores.")
        return
    comp = pd.DataFrame(output_rows).set_index("Jugador")
    stats = list(comp.columns)
    recommended = [x for x in ["goals", "assists", "goalsAssistsSum", "minutesPlayed", "appearances", "goalConversionPercentage", "accuratePassesPercentage"] if x in stats]
    chosen_stats = st.multiselect("Estadísticas globales a comparar", stats, default=recommended or stats[:8], key=f"compare_stats_{season}")
    if chosen_stats:
        st.markdown("##### Comparación global")
        global_table = comp[chosen_stats].T.rename_axis("Estadística").reset_index()
        st.dataframe(global_table, use_container_width=True, hide_index=True)
        st.download_button("Descargar comparación global CSV", global_table.to_csv(index=False).encode("utf-8-sig"),
                           f"comparacion_global_{season.replace('/', '-')}.csv", "text/csv", key=f"cmp_dl_{season}")
    st.markdown("##### Comparación por competición")
    competitions = sorted({str(c) for d in per_player.values() for c in d.get("competicion", pd.Series(dtype=str)).dropna().unique()})
    chosen_comps = st.multiselect("Competiciones", competitions, default=competitions, key=f"compare_comps_{season}")
    for comp_name in chosen_comps:
        with st.expander(comp_name, expanded=False):
            result = []
            for label, d_player in per_player.items():
                sub = d_player[d_player.competicion.astype(str) == comp_name]
                if sub.empty:
                    continue
                vals = aggregate_stats(sub)
                item = {"Jugador": label}
                for c in chosen_stats:
                    item[c] = vals.get(c, np.nan)
                result.append(item)
            if result:
                st.dataframe(pd.DataFrame(result), use_container_width=True, hide_index=True)
            else:
                st.caption("Ninguno de los jugadores seleccionados tiene datos en esta competición.")


# Barra lateral destinada exclusivamente a descargas, sin carga de archivos.
with st.sidebar:
    st.header("Descargar archivos")
    st.caption("Archivos locales disponibles en la carpeta `data/`.")
    if DATASET_PATH.exists():
        try:
            dataset_bytes = DATASET_PATH.read_bytes()
            st.download_button("Descargar dataset_ml_base.csv", dataset_bytes, "dataset_ml_base.csv", "text/csv", key="download_dataset")
        except OSError as e:
            st.warning(f"No se pudo leer el dataset: {e}")
    else:
        st.info("No se encuentra `data/dataset_ml_base.csv` en el directorio de la aplicación.")

    prediction_files = []
    if ML_DIR.exists():
        prediction_files = sorted(
            path for path in ML_DIR.glob("prediccion_ensamble*.csv")
            if path.is_file()
        )
    if prediction_files:
        st.markdown("**Predicciones del ranking**")
        for path in prediction_files:
            try:
                st.download_button(
                    path.name.replace("_", " ").replace(".csv", ""),
                    path.read_bytes(), path.name, "text/csv",
                    key=f"download_{path.name}",
                )
            except OSError as e:
                st.warning(f"No se pudo leer {path.name}: {e}")
    else:
        st.info("No se encontraron CSV de predicciones `prediccion_ensamble*.csv` en `data/ml/`.")

dataset = read_path(DATASET_PATH)
if dataset is None:
    dataset = pd.DataFrame()
if "temporada" in dataset:
    dataset["temporada"] = dataset.temporada.map(season_norm)

rankings = {}
if ML_DIR.exists():
    for path in sorted(ML_DIR.glob("prediccion_ensamble*.csv")):
        frame = read_path(path)
        if frame is None or frame.empty:
            continue
        season = season_from_name(path.name)
        if season is None and "temporada" in frame:
            uniques = frame.temporada.dropna().map(season_norm).unique().tolist()
            if len(uniques) == 1:
                season = uniques[0]
        if season in SEASONS:
            rankings[season] = norm_ranking(frame, season)

if not rankings:
    st.info("No se han encontrado rankings de predicción. Coloca los CSV `prediccion_ensamble*.csv` en `data/ml/`. Para las fichas y comparaciones también se necesita `data/dataset_ml_base.csv`.")

tabs = st.tabs(SEASONS + ["Comparador"])
for i, season in enumerate(SEASONS):
    with tabs[i]:
        st.markdown(f"### {season} · {'Predicción final' if season == '2025/26' else 'Backtest'}")
        ranking_tab(season, rankings.get(season), dataset)
with tabs[-1]:
    comparison_tab(dataset)

st.divider()
st.caption("Las estadísticas mostradas corresponden a las variables base del modelo, más goles, asistencias, apariciones y minutos jugados. Los porcentajes globales se recalculan con los numeradores y denominadores agregados.")