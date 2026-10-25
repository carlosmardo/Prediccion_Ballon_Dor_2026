from pathlib import Path
import re
import unicodedata
from html import escape as html_escape

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# Requiere Streamlit >= 1.55.0 para la ejecución diferida
# de pestañas y expanders mediante on_change="rerun".
# ============================================================

st.set_page_config(
    page_title="Balón de Oro · Explorador ML",
    page_icon="⚽",
    layout="wide",
)

SEASONS = [
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
    "2025/26",
]

DATASET_PATH = Path("data/dataset_ml_base.csv")
ML_DIR = Path("data/ml")

RANK_COLS = [
    "temporada",
    "posicion_predicha",
    "posicion_real",
    "nombre",
    "club",
    "posicion",
    "prob_duelo_contra_lider",
    "prob_duelo_contra_anterior",
    "score_modelo",
    "score_ranking",
]

ADDITIVE_STATS = [
    "goalsAssistsSum",
    "bigChancesCreated",
    "bigChancesMissed",
    "keyPasses",
    "successfulDribbles",
    "accuratePasses",
    "totalAttemptAssist",
    "aerialDuelsWon",
    "totalDuelsWon",
    "tacklesWon",
    "interceptions",
    "outfielderBlocks",
    "ballRecovery",
    "dispossessed",
    "dribbledPast",
    "clearances",
    "errorLeadToGoal",
    "errorLeadToShot",
    "saves",
    "goalsPrevented",
    "goalsConceded",
    "highClaims",
    "punches",
    "successfulRunsOut",
    "cleanSheet",
    "ownGoals",
    "penaltyConceded",
    "possessionWonAttThird",
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
    "saves",
    "goalsPrevented",
    "goalsConceded",
    "highClaims",
    "punches",
    "successfulRunsOut",
    "cleanSheet",
    "ownGoals",
    "penaltyConceded",
    "errorLeadToGoal",
    "errorLeadToShot",
    "accuratePassesPercentage",
}

GK_EXCLUSIVE = {
    "saves",
    "goalsPrevented",
    "highClaims",
    "punches",
    "successfulRunsOut",
}


st.markdown(
    """
    <style>
      .block-container {padding-top: 1.35rem; padding-bottom: 2rem;}
      html, body, [class*="css"] {font-size: 1.04rem;}
      button[data-baseweb="tab"] {font-size: 1.08rem; padding: 0.65rem 1.05rem; min-height: 3rem;}
      button[data-baseweb="tab"] p {font-size: 1.08rem;}
      [data-testid="stMetricValue"] {font-size: 1.55rem;}
      .small-note {opacity: .75; font-size: .92rem;}
      .ranking-card {
        border: 1px solid rgba(128, 128, 128, .28);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 8px 0 0 0;
        background: rgba(128, 128, 128, .045);
        transition: border-color .15s ease, background .15s ease;
      }
      .ranking-card:hover {
        border-color: rgba(128, 128, 128, .52);
        background: rgba(128, 128, 128, .075);
      }
      .ranking-card-top {
        display: grid;
        grid-template-columns: 54px minmax(160px, 1fr) 112px 112px;
        gap: 12px;
        align-items: center;
      }
      .ranking-rank {
        font-size: 1.25rem; font-weight: 750; letter-spacing: -.04em;
        text-align: center; color: var(--text-color);
      }
      .ranking-player-name {font-size: 1.02rem; font-weight: 650; line-height: 1.25; color: var(--text-color);}
      .ranking-player-meta, .ranking-label {color: var(--text-color); opacity: .66; font-size: .78rem; line-height: 1.35;}
      .ranking-top-value {text-align: right; overflow-wrap: anywhere;}
      .ranking-top-value strong {display: block; font-size: .94rem; font-weight: 600; font-variant-numeric: tabular-nums;}
      .ranking-card-metrics {
        display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px;
        margin: 13px 0 0 66px; padding-top: 11px;
        border-top: 1px solid rgba(128, 128, 128, .16);
      }
      .ranking-metric {min-width: 0;}
      .ranking-metric strong {display: block; margin-top: 2px; font-size: .9rem; font-weight: 600; overflow-wrap: anywhere; font-variant-numeric: tabular-nums;}
      @media (max-width: 900px) {
        .ranking-card-top {grid-template-columns: 42px minmax(120px, 1fr) 76px 76px; gap: 8px;}
        .ranking-card {padding: 12px;}
        .ranking-card-metrics {margin-left: 50px; gap: 7px;}
        .ranking-label {font-size: .7rem;}
        .ranking-metric strong {font-size: .8rem;}
      }
      @media (max-width: 600px) {
        .ranking-card-top {grid-template-columns: 36px minmax(0, 1fr) 64px;}
        .ranking-real-position {display: none;}
        .ranking-card-metrics {grid-template-columns: repeat(2, minmax(0, 1fr)); margin-left: 44px;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Balón de Oro · Explorador del modelo")
st.caption(
    "Predicciones históricas, scores del modelo, estadísticas por competición "
    "y comparación de jugadores."
)


# ============================================================
# FUNCIONES DE LIMPIEZA, NORMALIZACIÓN Y CACHÉ
# ============================================================


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
    value = str(x).strip().replace("-", "/").replace("_", "/")
    match = re.match(r"^(\d{4})/(\d{2}|\d{4})$", value)
    return f"{match.group(1)}/{match.group(2)[-2:]}" if match else value


def fold_text(value):
    """Búsqueda flexible: ignora mayúsculas, tildes y espacios repetidos."""
    text_value = unicodedata.normalize("NFKD", str(value))
    text_value = "".join(
        char for char in text_value if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", text_value).casefold().strip()


@st.cache_data(show_spinner=False)
def _read_path_cached(path_str, mtime_ns, file_size):
    """Lee y limpia un CSV; la modificación/tamaño invalidan la caché."""
    path = Path(path_str)
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    return clean_cols(df)


def read_path(path):
    if not path.exists():
        return None
    try:
        file_stat = path.stat()
        return _read_path_cached(
            str(path.resolve()),
            file_stat.st_mtime_ns,
            file_stat.st_size,
        )
    except Exception as exc:
        st.warning(f"No se pudo leer `{path}`: {exc}")
        return None


@st.cache_data(show_spinner=False)
def _read_bytes_cached(path_str, mtime_ns, file_size):
    return Path(path_str).read_bytes()


def file_bytes(path):
    file_stat = path.stat()
    return _read_bytes_cached(
        str(path.resolve()),
        file_stat.st_mtime_ns,
        file_stat.st_size,
    )


def season_from_name(name):
    # Admite prediccion_solo_temporada_backtest_2020_21.csv y ..._2025_26.csv.
    match = re.search(r"(20\d{2})[-_](\d{2}|20\d{2})", name)
    return season_norm(f"{match.group(1)}/{match.group(2)}") if match else None


def norm_ranking(df, season):
    df = clean_cols(df)
    aliases = {
        "posición_predicha": "posicion_predicha",
        "posición_real": "posicion_real",
        "jugador": "nombre",
        "player": "nombre",
        "equipo": "club",
        "posición": "posicion",
        # Compatibilidad con rankings anteriores.
        "score_modelo_posicional": "score_modelo",
        "score_ensamble": "score_modelo",
    }
    df = df.rename(
        columns={
            col: aliases[col]
            for col in df.columns
            if col in aliases and aliases[col] not in df.columns
        }
    )

    if "temporada" not in df:
        df["temporada"] = season

    df["temporada"] = df["temporada"].map(season_norm).replace("", season)

    for col in RANK_COLS:
        if col not in df:
            df[col] = np.nan

    for col in [
        "posicion_predicha",
        "posicion_real",
        "score_modelo",
        "score_ranking",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["posicion_predicha"] = df["posicion_predicha"].round().astype("Int64")

    # Columnas auxiliares para que las búsquedas no normalicen todos los textos
    # cada vez que Streamlit vuelve a ejecutar el script.
    for col in ["nombre", "club", "posicion"]:
        if col in df.columns:
            df[f"_{col}_norm"] = df[col].map(fold_text)

    return df.sort_values("posicion_predicha", na_position="last")


def player_rows(dataset, season, player):
    if dataset.empty or not {"temporada", "nombre"}.issubset(dataset.columns):
        return pd.DataFrame()

    season_values = (
        dataset["_temporada_norm"]
        if "_temporada_norm" in dataset.columns
        else dataset["temporada"].map(season_norm)
    )
    rows = dataset.loc[season_values.eq(season)]

    name = str(player.get("nombre", "")).strip()
    club = str(player.get("club", "")).strip()

    name_values = (
        rows["_nombre_norm"]
        if "_nombre_norm" in rows.columns
        else rows["nombre"].map(fold_text)
    )
    rows = rows.loc[name_values.eq(fold_text(name))]

    if club and fold_text(club) not in {"nan", "none", ""} and "club" in rows:
        club_values = (
            rows["_club_norm"]
            if "_club_norm" in rows.columns
            else rows["club"].map(fold_text)
        )
        club_rows = rows.loc[club_values.eq(fold_text(club))]
        if not club_rows.empty:
            rows = club_rows

    return rows


# ============================================================
# FUNCIONES DE CÁLCULO Y PRESENTACIÓN DE ESTADÍSTICAS
# ============================================================


def number_series(rows, col):
    if col not in rows:
        return pd.Series(dtype=float, index=rows.index)
    return pd.to_numeric(rows[col], errors="coerce")


def ratio_from_totals(rows, num_col, den_col):
    if num_col not in rows or den_col not in rows:
        return np.nan

    numerator = number_series(rows, num_col).sum(min_count=1)
    denominator = number_series(rows, den_col).sum(min_count=1)

    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    if denominator > 0:
        return 100.0 * numerator / denominator
    if denominator == 0:
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
    zeros = (
        numerator.eq(0)
        & percentage.eq(0)
        & numerator.notna()
        & percentage.notna()
    )
    return denominator.mask(zeros, 0.0)


def prepare_percent_denominators(rows):
    rows = rows.copy()
    for percentage_col, (numerator_col, denominator_col) in PERCENT_PAIRS.items():
        if numerator_col not in rows or percentage_col not in rows:
            continue

        inferred = infer_denominator(rows, numerator_col, percentage_col)
        if denominator_col not in rows:
            rows[denominator_col] = inferred
        else:
            rows[denominator_col] = number_series(rows, denominator_col).fillna(
                inferred
            )
    return rows


def model_stat_columns(rows):
    available = [col for col in MODEL_STATS if col in rows.columns]

    if (
        "goalsAssistsPerMins" not in available
        and "minutesPlayed" in rows.columns
        and (
            "goalsAssistsSum" in rows.columns
            or {"goals", "assists"}.issubset(rows.columns)
        )
    ):
        available.append("goalsAssistsPerMins")

    positions = set(
        rows.get("posicion", pd.Series(dtype=str)).dropna().astype(str).str.strip()
    )

    if positions and positions == {"Goalkeeper"}:
        allowed = GK_ALLOWED | {
            "goals",
            "assists",
            "appearances",
            "minutesPlayed",
            "goalsAssistsSum",
            "goalsAssistsPerMins",
        }
        available = [col for col in available if col in allowed]
    elif positions and "Goalkeeper" not in positions:
        available = [col for col in available if col not in GK_EXCLUSIVE]

    return available


@st.cache_data(show_spinner=False)
def aggregate_stats(rows, include_competition_result=False):
    """Agrega estadísticas siguiendo las reglas de presentación de la ficha."""
    if rows.empty:
        return {}

    rows = prepare_percent_denominators(rows)
    values = {}

    for col in model_stat_columns(rows):
        if col in ADDITIVE_STATS or col in {
            "goals",
            "assists",
            "appearances",
            "minutesPlayed",
        }:
            values[col] = number_series(rows, col).sum(min_count=1)

        elif col in PERCENT_PAIRS:
            numerator_col, denominator_col = PERCENT_PAIRS[col]
            values[col] = ratio_from_totals(rows, numerator_col, denominator_col)

        elif col == "goalsAssistsPerMins":
            ga_col = "goalsAssistsSum"
            if ga_col in rows and "minutesPlayed" in rows:
                ga = number_series(rows, ga_col).sum(min_count=1)
                minutes = number_series(rows, "minutesPlayed").sum(min_count=1)
                values[col] = (
                    ga / minutes
                    if pd.notna(ga) and pd.notna(minutes) and minutes != 0
                    else np.nan
                )
            elif {"goals", "assists", "minutesPlayed"}.issubset(rows.columns):
                ga = (
                    number_series(rows, "goals").sum(min_count=1)
                    + number_series(rows, "assists").sum(min_count=1)
                )
                minutes = number_series(rows, "minutesPlayed").sum(min_count=1)
                values[col] = (
                    ga / minutes
                    if pd.notna(ga) and pd.notna(minutes) and minutes != 0
                    else np.nan
                )

    for col in ["temporada", "nombre", "club", "posicion", "nacionalidad"]:
        if col in rows and rows[col].notna().any():
            values[col] = rows[col].dropna().iloc[0]

    if include_competition_result and "resultado_competicion" in rows.columns:
        result_cols = [
            col
            for col in ["competicion", "resultado_competicion"]
            if col in rows.columns
        ]
        if "resultado_competicion" in result_cols:
            results = rows[result_cols].dropna(
                subset=["resultado_competicion"]
            ).drop_duplicates()

            if "competicion" in results:
                values["resultado_competicion"] = " · ".join(
                    f"{row['competicion']}: {row['resultado_competicion']}"
                    for _, row in results.iterrows()
                ) or "—"
            else:
                values["resultado_competicion"] = " · ".join(
                    results["resultado_competicion"].astype(str).unique()
                ) or "—"

    return values


def display_stat_table(values, stats, title, key, result_text=None):
    st.markdown(f"**{title}**")
    data = []

    for col in stats:
        value = values.get(col, np.nan)
        if pd.notna(value):
            data.append({"Estadística": col, "Valor": value})

    if result_text:
        data.append(
            {"Estadística": "resultado_competicion", "Valor": result_text}
        )

    if not data:
        st.info("No hay estadísticas disponibles para mostrar.")
        return

    table = pd.DataFrame(data)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=min(620, 42 + len(table) * 34),
        column_config={
            "Estadística": st.column_config.TextColumn(width="large"),
            "Valor": st.column_config.TextColumn(width="large"),
        },
    )

    safe_name = re.sub(r"\W+", "_", title).strip("_") or "estadisticas"
    st.download_button(
        "Descargar estadísticas CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"estadisticas_{safe_name}.csv",
        mime="text/csv",
        key=key,
    )


def result_for_competition(rows):
    if "resultado_competicion" not in rows:
        return None

    values = rows["resultado_competicion"].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return " · ".join(dict.fromkeys(values.tolist())) if not values.empty else None


def player_detail(dataset, season, player, prefix):
    rows = player_rows(dataset, season, player)
    player_name = player.get("nombre", "este jugador")

    if rows.empty:
        st.info(
            f"No se encontraron estadísticas para {player_name} en {season}. "
            "Comprueba la temporada y el nombre del dataset."
        )
        return

    stats = model_stat_columns(rows)
    global_values = aggregate_stats(rows, include_competition_result=True)
    competitions = (
        rows["competicion"]
        .fillna("Sin competición")
        .astype(str)
        .drop_duplicates()
        .tolist()
        if "competicion" in rows
        else ["Sin competición"]
    )

    # Solo se calcula el contenido de la pestaña seleccionada.
    comp_tabs = st.tabs(
        ["🌐 Global"] + competitions,
        on_change="rerun",
        key=f"player_comp_tabs_{prefix}",
    )

    if comp_tabs[0].open:
        with comp_tabs[0]:
            st.caption(
                "Se muestran las estadísticas del modelo y las cuatro adicionales "
                "solicitadas. Los porcentajes globales se recalculan sumando "
                "numeradores y denominadores."
            )
            display_stat_table(
                global_values,
                stats,
                "Estadísticas globales",
                f"{prefix}_global",
                global_values.get("resultado_competicion"),
            )

    for index, competition in enumerate(competitions):
        tab = comp_tabs[index + 1]
        if not tab.open:
            continue

        with tab:
            if "competicion" in rows:
                sub = rows[
                    rows["competicion"].fillna("Sin competición").astype(str)
                    == competition
                ]
            else:
                sub = rows

            values = aggregate_stats(sub)
            display_stat_table(
                values,
                stats,
                f"Estadísticas · {competition}",
                f"{prefix}_comp_{index}",
                result_for_competition(sub),
            )


def format_probability(value):
    if pd.isna(value) or str(value).strip() == "":
        return "—"

    try:
        raw = str(value).strip()
        number = float(raw.replace("%", "").replace(",", "."))
        return (
            f"{number:.2f}%"
            if "%" in raw or number > 1
            else f"{number * 100:.2f}%"
        )
    except (ValueError, TypeError):
        return str(value)


# ============================================================
# PESTAÑAS DE RANKING
# ============================================================


def ranking_tab(season, ranking, dataset):
    if ranking is None or ranking.empty:
        st.warning(
            f"No hay ranking cargado para {season}. Coloca el CSV "
            "`prediccion_solo_temporada...` correspondiente en `data/ml/`."
        )
        return

    # Los rankings ya se normalizan al cargarlos; no se repite ese trabajo.
    df = ranking
    if df["temporada"].nunique(dropna=True) > 1:
        df = df[df["temporada"].eq(season)]

    query_text = st.text_input(
        "Buscar jugador o club",
        key=f"search_{season}",
        placeholder="Ej. Cubarsi, Messi, Real Madrid…",
    )

    filtered = df
    if query_text.strip():
        query = fold_text(query_text)
        mask = pd.Series(False, index=df.index)

        for col in ["nombre", "club", "posicion"]:
            if col not in df.columns:
                continue
            normalized_col = f"_{col}_norm"
            values = (
                df[normalized_col]
                if normalized_col in df.columns
                else df[col].map(fold_text)
            )
            mask |= values.str.contains(re.escape(query), regex=True, na=False)

        filtered = df.loc[mask]

    st.caption(f"Mostrando {len(filtered):,} jugadores".replace(",", "."))

    def display_int(value):
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "—" if pd.isna(number) else str(int(number))

    def display_score(value):
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "—" if pd.isna(number) else f"{number:.4f}"

    def card_text(value, fallback="—"):
        return html_escape(
            fallback if pd.isna(value) or str(value).strip() == "" else str(value)
        )

    for card_idx, (_, row) in enumerate(filtered.iterrows()):
        rank = display_int(row.get("posicion_predicha"))
        player_name_raw = str(row.get("nombre", "Jugador desconocido"))
        name = card_text(row.get("nombre"), "Jugador desconocido")
        club = card_text(row.get("club"))
        position = card_text(row.get("posicion"))
        actual = display_int(row.get("posicion_real"))
        prob_leader = html_escape(
            format_probability(row.get("prob_duelo_contra_lider"))
        )
        prob_previous = html_escape(
            format_probability(row.get("prob_duelo_contra_anterior"))
        )
        score_model = html_escape(display_score(row.get("score_modelo")))
        score_ranking = html_escape(display_score(row.get("score_ranking")))

        card = f"""
        <div class="ranking-card">
          <div class="ranking-card-top">
            <div class="ranking-rank">{html_escape(rank)}</div>
            <div>
              <div class="ranking-player-name">{name}</div>
              <div class="ranking-player-meta">{club} · {position}</div>
            </div>
            <div class="ranking-top-value ranking-real-position">
              <span class="ranking-label">Posición real</span>
              <strong>{html_escape(actual)}</strong>
            </div>
            <div class="ranking-top-value">
              <span class="ranking-label">Temporada</span>
              <strong>{html_escape(season)}</strong>
            </div>
          </div>
          <div class="ranking-card-metrics">
            <div class="ranking-metric"><span class="ranking-label">Prob. contra líder</span><strong>{prob_leader}</strong></div>
            <div class="ranking-metric"><span class="ranking-label">Prob. contra anterior</span><strong>{prob_previous}</strong></div>
            <div class="ranking-metric"><span class="ranking-label">Score modelo</span><strong>{score_model}</strong></div>
            <div class="ranking-metric"><span class="ranking-label">Score ranking</span><strong>{score_ranking}</strong></div>
          </div>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)

        player_dict = row.to_dict()
        safe_player = (
            re.sub(r"\W+", "_", player_name_raw).strip("_")
            or f"jugador_{card_idx}"
        )
        season_key = season.replace("/", "_")

        details = st.expander(
            f"📊 Ver estadísticas · {player_name_raw}",
            expanded=False,
            key=f"player_details_{season_key}_{safe_player}_{card_idx}",
            on_change="rerun",
        )

        # Las métricas y estadísticas solo se calculan si se abre la ficha.
        if details.open:
            with details:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Posición predicha", rank)
                col2.metric("Posición real", actual)
                col3.metric(
                    "Club",
                    str(row.get("club"))
                    if pd.notna(row.get("club"))
                    else "—",
                )
                col4.metric(
                    "Posición",
                    str(row.get("posicion"))
                    if pd.notna(row.get("posicion"))
                    else "—",
                )

                score_boxes = st.columns(4)
                score_boxes[0].metric(
                    "Score modelo", display_score(row.get("score_modelo"))
                )
                score_boxes[1].metric(
                    "Score ranking", display_score(row.get("score_ranking"))
                )
                score_boxes[2].metric(
                    "Prob. contra líder",
                    format_probability(row.get("prob_duelo_contra_lider")),
                )
                score_boxes[3].metric(
                    "Prob. contra anterior",
                    format_probability(row.get("prob_duelo_contra_anterior")),
                )

                player_detail(
                    dataset,
                    season,
                    player_dict,
                    f"{season_key}_{safe_player}_{card_idx}",
                )

    st.download_button(
        f"Descargar ranking {season} CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        f"ranking_{season.replace('/', '-')}.csv",
        "text/csv",
        key=f"dl_rank_{season}",
    )


# ============================================================
# COMPARADOR DE JUGADORES
# ============================================================


def comparison_tab(dataset):
    st.subheader("Comparador de jugadores")

    if dataset.empty or not {"temporada", "nombre"}.issubset(dataset.columns):
        st.warning(
            "No se encuentra `data/dataset_ml_base.csv` para comparar estadísticas."
        )
        return

    season_values = (
        dataset["_temporada_norm"]
        if "_temporada_norm" in dataset.columns
        else dataset["temporada"].map(season_norm)
    )
    available = set(season_values.dropna())
    season_options = [season for season in SEASONS if season in available]
    if not season_options:
        season_options = sorted(season for season in available if season)

    if not season_options:
        st.info("El dataset no contiene temporadas válidas.")
        return

    default_season = (
        "2025/26" if "2025/26" in season_options else season_options[-1]
    )
    season = st.selectbox(
        "Temporada de estadísticas",
        season_options,
        index=season_options.index(default_season),
        key="compare_season",
    )

    sd = dataset.loc[season_values.eq(season)]
    if sd.empty:
        st.info("No hay datos para esta temporada.")
        return

    pair_columns = [col for col in ["nombre", "club"] if col in sd.columns]
    pair = (
        sd[pair_columns]
        .dropna(subset=["nombre"])
        .drop_duplicates()
        .copy()
    )

    if "club" not in pair.columns:
        pair["club"] = "Club desconocido"

    pair["label"] = (
        pair["nombre"].astype(str)
        + " — "
        + pair["club"].fillna("Club desconocido").astype(str)
    )
    options = pair["label"].tolist()

    chosen_players = st.multiselect(
        "Elige dos o más jugadores",
        options,
        default=options[:2],
        key=f"compare_players_{season.replace('/', '_')}",
    )

    if len(chosen_players) < 2:
        st.info("Selecciona al menos dos jugadores.")
        return

    per_player = {}
    output_rows = []
    selected_pair = pair[pair["label"].isin(chosen_players)]

    for _, player in selected_pair.iterrows():
        matched = player_rows(
            sd,
            season,
            {"nombre": player["nombre"], "club": player["club"]},
        )
        if matched.empty:
            continue

        values = aggregate_stats(matched, include_competition_result=True)
        player_label = player["label"]
        per_player[player_label] = matched

        item = {"Jugador": player_label}
        for col in model_stat_columns(matched):
            if col in values and pd.notna(values[col]):
                item[col] = values[col]

        # Variable categórica por competición, disponible para la comparación global.
        item["resultado_competicion"] = values.get("resultado_competicion", "—")
        output_rows.append(item)

    if len(output_rows) < 2:
        st.info("No se pudieron preparar estadísticas para al menos dos jugadores.")
        return

    comparison = pd.DataFrame(output_rows).set_index("Jugador")
    stats = list(comparison.columns)
    recommended = [
        col
        for col in [
            "goals",
            "assists",
            "goalsAssistsSum",
            "minutesPlayed",
            "appearances",
            "goalConversionPercentage",
            "accuratePassesPercentage",
        ]
        if col in stats
    ]

    chosen_stats = st.multiselect(
        "Estadísticas globales a comparar (incluye resultado_competicion)",
        stats,
        default=recommended or stats[:8],
        key=f"compare_stats_{season.replace('/', '_')}",
    )

    if chosen_stats:
        st.markdown("##### Comparación global")
        global_table = (
            comparison[chosen_stats]
            .T.rename_axis("Estadística")
            .reset_index()
        )
        st.dataframe(global_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar comparación global CSV",
            global_table.to_csv(index=False).encode("utf-8-sig"),
            f"comparacion_global_{season.replace('/', '-')}.csv",
            "text/csv",
            key=f"cmp_dl_{season}",
        )

    st.markdown("##### Comparación por competición")
    competitions = sorted(
        {
            str(competition)
            for player_rows_df in per_player.values()
            for competition in player_rows_df.get(
                "competicion", pd.Series(dtype=str)
            ).dropna().unique()
        }
    )

    chosen_competitions = st.multiselect(
        "Competiciones",
        competitions,
        default=competitions,
        key=f"compare_comps_{season.replace('/', '_')}",
    )

    for comp_idx, competition_name in enumerate(chosen_competitions):
        competition_expander = st.expander(
            competition_name,
            expanded=False,
            key=(
                f"compare_comp_{season.replace('/', '_')}_"
                f"{comp_idx}"
            ),
            on_change="rerun",
        )

        # Solo se agregan estadísticas de una competición si se abre su desplegable.
        if not competition_expander.open:
            continue

        with competition_expander:
            result = []

            for label, player_rows_df in per_player.items():
                if "competicion" not in player_rows_df:
                    continue

                sub = player_rows_df[
                    player_rows_df["competicion"].astype(str) == competition_name
                ]
                if sub.empty:
                    continue

                values = aggregate_stats(sub)
                item = {"Jugador": label}

                for col in chosen_stats:
                    if col == "resultado_competicion":
                        item[col] = result_for_competition(sub) or "—"
                    else:
                        item[col] = values.get(col, np.nan)

                result.append(item)

            if result:
                st.dataframe(
                    pd.DataFrame(result),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption(
                    "Ninguno de los jugadores seleccionados tiene datos "
                    "en esta competición."
                )


# ============================================================
# CARGA DE DATOS Y BARRA LATERAL
# ============================================================

# Carga primero los datos para que las descargas reutilicen la misma estrategia
# de caché que el resto de la aplicación.
dataset = read_path(DATASET_PATH)
if dataset is None:
    dataset = pd.DataFrame()

if "temporada" in dataset.columns:
    dataset["temporada"] = dataset["temporada"].map(season_norm)
    dataset["_temporada_norm"] = dataset["temporada"]

if "nombre" in dataset.columns:
    dataset["_nombre_norm"] = dataset["nombre"].map(fold_text)

if "club" in dataset.columns:
    dataset["_club_norm"] = dataset["club"].map(fold_text)


rankings = {}
if ML_DIR.exists():
    ranking_files = sorted(ML_DIR.glob("prediccion_solo_temporada*.csv"))
    for path in ranking_files:
        frame = read_path(path)
        if frame is None or frame.empty:
            continue

        season = season_from_name(path.name)
        if season is None and "temporada" in frame.columns:
            unique_seasons = (
                frame["temporada"].dropna().map(season_norm).unique().tolist()
            )
            if len(unique_seasons) == 1:
                season = unique_seasons[0]

        if season in SEASONS:
            rankings[season] = norm_ranking(frame, season)

with st.sidebar:
    st.header("Descargar archivos")
    st.caption("Archivos locales disponibles en la carpeta `data/`. ")

    if DATASET_PATH.exists():
        try:
            st.download_button(
                "Descargar dataset_ml_base.csv",
                file_bytes(DATASET_PATH),
                "dataset_ml_base.csv",
                "text/csv",
                key="download_dataset",
            )
        except OSError as exc:
            st.warning(f"No se pudo leer el dataset: {exc}")
    else:
        st.info(
            "No se encuentra `data/dataset_ml_base.csv` en el directorio de la aplicación."
        )

    prediction_files = (
        sorted(
            path
            for path in ML_DIR.glob("prediccion_solo_temporada*.csv")
            if path.is_file()
        )
        if ML_DIR.exists()
        else []
    )

    if prediction_files:
        st.markdown("**Predicciones del ranking**")
        for path in prediction_files:
            try:
                st.download_button(
                    path.name.replace("_", " ").replace(".csv", ""),
                    file_bytes(path),
                    path.name,
                    "text/csv",
                    key=f"download_{path.name}",
                )
            except OSError as exc:
                st.warning(f"No se pudo leer {path.name}: {exc}")
    else:
        st.info(
            "No se encontraron CSV `prediccion_solo_temporada*.csv` en `data/ml/`."
        )

if not rankings:
    st.info(
        "No se han encontrado rankings de predicción. Coloca los CSV "
        "`prediccion_solo_temporada...csv` en `data/ml/`. Para las fichas y "
        "comparaciones también se necesita `data/dataset_ml_base.csv`."
    )


# ============================================================
# INTERFAZ PRINCIPAL: SOLO SE RENDERIZA LA PESTAÑA ACTIVA
# ============================================================

main_tabs = st.tabs(
    SEASONS + ["Comparador"],
    on_change="rerun",
    key="main_tabs",
)

for index, season in enumerate(SEASONS):
    tab = main_tabs[index]
    if not tab.open:
        continue

    with tab:
        st.markdown(
            f"### {season} · "
            f"{'Predicción final' if season == '2025/26' else 'Backtest'}"
        )
        ranking_tab(season, rankings.get(season), dataset)

if main_tabs[-1].open:
    with main_tabs[-1]:
        comparison_tab(dataset)

st.divider()
st.caption(
    "Las estadísticas mostradas corresponden a las variables base del modelo, "
    "más goles, asistencias, apariciones y minutos jugados."
)
