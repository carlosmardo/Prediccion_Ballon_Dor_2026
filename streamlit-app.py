from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Balón de Oro · Explorador ML', page_icon='⚽', layout='wide')
SEASONS = ['2020/21','2021/22','2022/23','2023/24','2024/25','2025/26']
DATASET_PATH = Path('data/dataset_ml_base.csv')
ML_DIR = Path('data/ml')
RANK_COLS = ['temporada','posicion_predicha','posicion_real','nombre','club','posicion','score_ensamble','score_modelo_posicional','score_modelo_global','prob_duelo_contra_lider','prob_duelo_contra_anterior']
ID_COLS = {'temporada','posicion_balon_oro','club','posicion','nacionalidad','nombre','id_sofascore','competicion','resultado_competicion','dataset'}
RATE_COLS = {'accurateCrossesPercentage','accurateLongBallsPercentage','accuratePassesPercentage','aerialDuelsWonPercentage','groundDuelsWonPercentage','goalConversionPercentage','penaltyConversion','scoringFrequency','successfulDribblesPercentage','tacklesWonPercentage','totalDuelsWonPercentage','kilometersCoveredPer90','numberOfSprintsPer90','topSpeed'}
PERCENT_PAIRS = {
 'accurateCrossesPercentage':('accurateCrosses','totalCross'),
 'accurateLongBallsPercentage':('accurateLongBalls','totalLongBalls'),
 'accuratePassesPercentage':('accuratePasses','totalPasses'),
 'goalConversionPercentage':('goals','totalShots'),
 'successfulDribblesPercentage':('successfulDribbles','totalContest'),
 'tacklesWonPercentage':('tacklesWon','tackles'),
 'totalDuelsWonPercentage':('totalDuelsWon','duelLost+totalDuelsWon'),
}

st.markdown('''<style>.block-container{padding-top:1.3rem}.small-note{opacity:.75;font-size:.9rem}</style>''', unsafe_allow_html=True)
st.title('⚽ Balón de Oro · Explorador del modelo')
st.caption('Predicciones históricas, scores del ensamble, estadísticas por competición y comparación de jugadores.')


def clean_cols(df):
    df = df.copy()
    df.columns = [re.sub(r'^&#x[0-9a-fA-F]+;', '', str(c).replace('**','').strip()).strip() for c in df.columns]
    return df.loc[:, ~pd.Index(df.columns).duplicated()]


def season_norm(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('-', '/').replace('_','/')
    m = re.match(r'^(\d{4})/(\d{2}|\d{4})$', s)
    return f'{m.group(1)}/{m.group(2)[-2:]}' if m else s


def read_csv_obj(obj):
    if obj is None: return None
    try:
        obj.seek(0)
        try: df = pd.read_csv(obj, low_memory=False)
        except UnicodeDecodeError:
            obj.seek(0); df = pd.read_csv(obj, encoding='latin-1', low_memory=False)
        return clean_cols(df)
    except Exception as e:
        st.error(f'No se pudo leer {getattr(obj, "name", obj)}: {e}')
        return None


def read_path(path):
    if not path.exists(): return None
    try:
        try: return clean_cols(pd.read_csv(path, low_memory=False))
        except UnicodeDecodeError: return clean_cols(pd.read_csv(path, encoding='latin-1', low_memory=False))
    except Exception as e:
        st.warning(f'No se pudo leer `{path}`: {e}'); return None


def season_from_name(name):
    m = re.search(r'(20\d{2})[-_](\d{2}|20\d{2})', name)
    return season_norm(f'{m.group(1)}/{m.group(2)}') if m else None


def norm_ranking(df, season):
    df = clean_cols(df)
    aliases = {'posición_predicha':'posicion_predicha','posición_real':'posicion_real','jugador':'nombre','player':'nombre','equipo':'club','posición':'posicion'}
    df = df.rename(columns={c:aliases[c] for c in df.columns if c in aliases})
    if 'temporada' not in df: df['temporada'] = season
    df['temporada'] = df['temporada'].map(season_norm).replace('', season)
    for c in RANK_COLS:
        if c not in df: df[c] = np.nan
    for c in ['posicion_predicha','posicion_real','score_ensamble','score_modelo_posicional','score_modelo_global']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.sort_values('posicion_predicha', na_position='last')


def numeric_stats(df):
    out=[]
    for c in df.columns:
        if c in ID_COLS: continue
        if pd.api.types.is_numeric_dtype(df[c]): out.append(c)
        elif len(df) and pd.to_numeric(df[c], errors='coerce').notna().sum() >= max(1, int(len(df)*.6)): out.append(c)
    return out


def player_rows(dataset, season, player):
    if dataset.empty or not {'temporada','nombre'}.issubset(dataset.columns): return pd.DataFrame()
    d=dataset[dataset['temporada'].map(season_norm)==season].copy()
    name=str(player.get('nombre','')).strip(); club=str(player.get('club','')).strip()
    d=d[d['nombre'].astype(str).str.strip().str.casefold()==name.casefold()]
    if club and club.casefold() not in {'nan','none',''} and 'club' in d:
        dc=d[d['club'].astype(str).str.strip().str.casefold()==club.casefold()]
        if not dc.empty: d=dc
    return d


def global_stats(rows):
    if rows.empty: return pd.DataFrame()
    vals={}; stats=numeric_stats(rows)
    for c in stats:
        v=pd.to_numeric(rows[c], errors='coerce')
        if c in RATE_COLS or c.endswith('Percentage') or c.endswith('Per90') or c=='topSpeed': vals[c]=v.mean()
        else: vals[c]=v.sum(min_count=1)
    for rate,(num,den) in PERCENT_PAIRS.items():
        if rate not in rows or num not in rows: continue
        numerator=pd.to_numeric(rows[num],errors='coerce').sum(min_count=1)
        if '+' in den:
            a,b=den.split('+')
            if a not in rows or b not in rows: continue
            denominator=pd.to_numeric(rows[a],errors='coerce').sum(min_count=1)+pd.to_numeric(rows[b],errors='coerce').sum(min_count=1)
        else:
            if den not in rows: continue
            denominator=pd.to_numeric(rows[den],errors='coerce').sum(min_count=1)
        if pd.notna(numerator) and pd.notna(denominator) and denominator: vals[rate]=100*numerator/denominator
    for c in ['temporada','nombre','club','posicion','nacionalidad']:
        if c in rows and rows[c].notna().any(): vals[c]=rows[c].dropna().iloc[0]
    vals['competicion']='GLOBAL (todas las competiciones)'
    return pd.DataFrame([vals])


def show_stat_table(row, stats, title, key):
    st.markdown(f'**{title}**')
    data=[]
    for c in stats:
        v=row.get(c, np.nan)
        if pd.notna(v): data.append({'Estadística':c,'Valor':v})
    if not data:
        st.info('No hay estadísticas numéricas disponibles.'); return
    table=pd.DataFrame(data)
    st.dataframe(table, use_container_width=True, hide_index=True, height=min(600, 38+len(table)*32),
                 column_config={'Estadística':st.column_config.TextColumn(width='large'),'Valor':st.column_config.NumberColumn(format='%.3f')})
    st.download_button('Descargar estadísticas CSV', table.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'estadisticas_{re.sub(r"\W+","_",title)}.csv', mime='text/csv', key=key)


def player_detail(dataset, season, player, prefix):
    rows=player_rows(dataset, season, player)
    if rows.empty:
        st.info(f'No se encontraron estadísticas para {player.get("nombre", "este jugador")} en {season}. Comprueba temporada y nombre en dataset_ml_base.csv.'); return
    stats=sorted([c for c in numeric_stats(rows) if c not in {'posicion_balon_oro','id_sofascore'}], key=str.casefold)
    global_row=global_stats(rows).iloc[0].to_dict()
    comps=rows['competicion'].fillna('Sin competición').astype(str).tolist() if 'competicion' in rows else ['Sin competición']
    # Evitar pestañas duplicadas si el CSV repite una competición.
    unique_comps=list(dict.fromkeys(comps))
    comp_tabs=st.tabs(['🌐 Global']+unique_comps)
    with comp_tabs[0]:
        st.caption('Las estadísticas acumulables se suman; las tasas y métricas por 90 se promedian, y algunas tasas se recalculan si hay numerador/denominador disponibles.')
        show_stat_table(global_row, stats, 'Estadísticas globales', f'{prefix}_global')
    for i, comp in enumerate(unique_comps):
        with comp_tabs[i+1]:
            sub=rows[rows['competicion'].fillna('Sin competición').astype(str)==comp]
            row=global_stats(sub).iloc[0].to_dict() if len(sub)>1 else sub.iloc[0].to_dict()
            show_stat_table(row, stats, f'Estadísticas · {comp}', f'{prefix}_comp_{i}')


def ranking_tab(season, ranking, dataset):
    if ranking is None or ranking.empty:
        st.warning(f'No hay ranking cargado para {season}. Sube el CSV o colócalo en `data/ml/`.'); return
    d=norm_ranking(ranking, season)
    if d['temporada'].nunique(dropna=True)>1: d=d[d['temporada'].map(season_norm)==season]
    a,b,c,e=st.columns(4)
    a.metric('Jugadores',f'{len(d):,}'); b.metric('Temporada',season)
    c.metric('Con posición real',f'{pd.to_numeric(d.posicion_real,errors="coerce").notna().sum():,}')
    score=pd.to_numeric(d.score_ensamble,errors='coerce'); e.metric('Score medio',f'{score.mean():.3f}' if score.notna().any() else '—')
    q=st.text_input('Buscar jugador o club',key=f'search_{season}',placeholder='Ej. Messi, Real Madrid…')
    filtered=d.copy()
    if q.strip():
        mask=pd.Series(False,index=filtered.index)
        for col in ['nombre','club','posicion']:
            if col in filtered: mask |= filtered[col].astype(str).str.contains(re.escape(q.strip()),case=False,na=False)
        filtered=filtered[mask]
    shown=filtered[[x for x in RANK_COLS if x in filtered]].copy()
    for col in ['prob_duelo_contra_lider','prob_duelo_contra_anterior']:
        if col in shown:
            def pct(x):
                if pd.isna(x) or str(x).strip()=='': return '—'
                try:
                    raw=str(x); n=float(raw.replace('%','').replace(',','.'))
                    return f'{n:.2f}%' if '%' in raw or n>1 else f'{n*100:.2f}%'
                except: return str(x)
            shown[col]=shown[col].map(pct)
    st.dataframe(shown,use_container_width=True,hide_index=True,height=min(620,42+len(shown)*34),column_config={
        'posicion_predicha':st.column_config.NumberColumn('Posición predicha',format='%d'),
        'posicion_real':st.column_config.NumberColumn('Posición real',format='%.1f'),
        'score_ensamble':st.column_config.NumberColumn('Score ensamble',format='%.4f'),
        'score_modelo_posicional':st.column_config.NumberColumn('Score posicional',format='%.4f'),
        'score_modelo_global':st.column_config.NumberColumn('Score global',format='%.4f'),})
    st.download_button(f'Descargar ranking {season} CSV',d.to_csv(index=False).encode('utf-8-sig'),f'ranking_{season.replace("/","-")}.csv','text/csv',key=f'dl_rank_{season}')
    st.divider(); st.markdown('#### Ficha individual')
    names=d['nombre'].dropna().astype(str).drop_duplicates().tolist()
    if not names: st.info('El ranking no contiene nombres de jugadores.'); return
    name=st.selectbox('Selecciona un jugador',names,key=f'player_{season}')
    p=d[d.nombre.astype(str)==name].iloc[0]
    x1,x2,x3,x4=st.columns(4)
    x1.metric('Posición predicha','—' if pd.isna(p.posicion_predicha) else str(int(p.posicion_predicha)))
    x2.metric('Posición real','—' if pd.isna(p.posicion_real) else str(p.posicion_real))
    x3.metric('Club',str(p.club) if pd.notna(p.club) else '—'); x4.metric('Posición',str(p.posicion) if pd.notna(p.posicion) else '—')
    s1,s2,s3=st.columns(3)
    for col,label,box in [('score_ensamble','Score ensamble',s1),('score_modelo_posicional','Score posicional',s2),('score_modelo_global','Score global',s3)]:
        val=pd.to_numeric(pd.Series([p.get(col)]),errors='coerce').iloc[0]; box.metric(label,'—' if pd.isna(val) else f'{val:.4f}')
    d1,d2=st.columns(2)
    for col,label,box in [('prob_duelo_contra_lider','Prob. contra líder',d1),('prob_duelo_contra_anterior','Prob. contra anterior',d2)]:
        val=p.get(col); box.metric(label,'—' if pd.isna(val) or str(val)=='' else str(val))
    with st.expander('Ver todas las estadísticas por competición y global',expanded=True): player_detail(dataset,season,p,f'{season}_{re.sub(r"\W+","_",name)}')


def comparison_tab(dataset):
    st.subheader('Comparador de jugadores')
    if dataset.empty or not {'temporada','nombre'}.issubset(dataset.columns):
        st.warning('Carga dataset_ml_base.csv para comparar estadísticas.'); return
    avail={season_norm(x) for x in dataset.temporada.dropna()}
    season_options=[s for s in SEASONS if s in avail] or sorted(avail)
    default='2025/26' if '2025/26' in season_options else season_options[-1]
    season=st.selectbox('Temporada de estadísticas',season_options,index=season_options.index(default),key='compare_season')
    sd=dataset[dataset.temporada.map(season_norm)==season].copy()
    if sd.empty: st.info('No hay datos para esta temporada.'); return
    pair=sd[['nombre','club']].dropna(subset=['nombre']).drop_duplicates().copy()
    pair['label']=pair.nombre.astype(str)+' — '+pair.club.fillna('Club desconocido').astype(str)
    options=pair.label.tolist()
    chosen=st.multiselect('Elige dos o más jugadores',options,default=options[:2],key=f'compare_players_{season}')
    if len(chosen)<2: st.info('Selecciona al menos dos jugadores.'); return
    per_player={}; rows=[]
    for _,p in pair[pair.label.isin(chosen)].iterrows():
        matched=player_rows(dataset,season,{'nombre':p.nombre,'club':p.club})
        if matched.empty: continue
        vals=global_stats(matched).iloc[0].to_dict(); per_player[p.label]=matched
        item={'Jugador':p.label}
        for col in numeric_stats(matched):
            if col in vals and pd.notna(vals[col]): item[col]=vals[col]
        rows.append(item)
    if len(rows)<2: st.info('No se pudieron preparar estadísticas para al menos dos jugadores.'); return
    comp=pd.DataFrame(rows).set_index('Jugador')
    stats=sorted(comp.columns,key=str.casefold)
    recommended=[x for x in ['goals','assists','expectedGoals','expectedAssists','minutesPlayed','appearances','totalShots','accuratePasses','tackles','interceptions'] if x in stats]
    chosen_stats=st.multiselect('Estadísticas globales a comparar',stats,default=recommended or stats[:8],key=f'compare_stats_{season}')
    if chosen_stats:
        st.markdown('##### Comparación global')
        st.dataframe(comp[chosen_stats].T.rename_axis('Estadística').reset_index(),use_container_width=True,hide_index=True)
        st.download_button('Descargar comparación global CSV',comp[chosen_stats].T.rename_axis('Estadística').to_csv().encode('utf-8-sig'),f'comparacion_global_{season.replace("/","-")}.csv','text/csv',key=f'cmp_dl_{season}')
    st.markdown('##### Comparación por competición')
    competitions=sorted({str(c) for d in per_player.values() for c in d.get('competicion',pd.Series(dtype=str)).dropna().unique()})
    chosen_comps=st.multiselect('Competiciones',competitions,default=competitions,key=f'compare_comps_{season}')
    for comp_name in chosen_comps:
        with st.expander(comp_name,expanded=False):
            result=[]
            for label,d in per_player.items():
                sub=d[d.competicion.astype(str)==comp_name]
                if sub.empty: continue
                vals=global_stats(sub).iloc[0].to_dict(); item={'Jugador':label}
                for c in chosen_stats: item[c]=vals.get(c,np.nan)
                result.append(item)
            if result: st.dataframe(pd.DataFrame(result),use_container_width=True,hide_index=True)
            else: st.caption('Ninguno de los jugadores seleccionados tiene datos en esta competición.')

with st.sidebar:
    st.header('Carga de archivos')
    dataset_upload=st.file_uploader('dataset_ml_base.csv',type=['csv'],key='dataset_upload',help='También se busca automáticamente data/dataset_ml_base.csv')
    ranking_uploads=st.file_uploader('CSV de rankings (puedes elegir varios)',type=['csv'],accept_multiple_files=True,key='ranking_uploads',help='Ej.: ranking_2020-21.csv, ranking_2025-26.csv')
    st.caption('Rutas locales por defecto: `data/dataset_ml_base.csv` y `data/ml/*.csv`.')

dataset=read_csv_obj(dataset_upload) if dataset_upload else read_path(DATASET_PATH)
if dataset is None: dataset=pd.DataFrame()
if 'temporada' in dataset: dataset['temporada']=dataset.temporada.map(season_norm)
rankings={}
if ML_DIR.exists():
    for path in sorted(ML_DIR.glob('*.csv')):
        df=read_path(path)
        if df is None or df.empty: continue
        season=season_from_name(path.name)
        if season is None and 'temporada' in df:
            uniques=df.temporada.dropna().map(season_norm).unique().tolist()
            if len(uniques)==1: season=uniques[0]
        if season in SEASONS: rankings[season]=norm_ranking(df,season)
for uploaded in ranking_uploads or []:
    df=read_csv_obj(uploaded)
    if df is None or df.empty: continue
    season=season_from_name(uploaded.name)
    if season is None and 'temporada' in df:
        uniques=df.temporada.dropna().map(season_norm).unique().tolist()
        if len(uniques)==1: season=uniques[0]
    if season in SEASONS: rankings[season]=norm_ranking(df,season)
    else: st.sidebar.warning(f'No se detectó la temporada de {uploaded.name}. Inclúyela en el nombre o en la columna temporada.')

if not dataset.empty: st.sidebar.success(f'Dataset: {len(dataset):,} filas · {len(dataset.columns):,} columnas')
else: st.sidebar.warning('No se ha cargado el dataset de estadísticas.')
if not rankings: st.info('Sube los CSV de rankings en la barra lateral o colócalos en `data/ml/`. Para fichas y comparaciones, carga también `data/dataset_ml_base.csv`.')

tabs=st.tabs(SEASONS+['Comparador'])
for i,season in enumerate(SEASONS):
    with tabs[i]:
        st.markdown(f'### {season} · {"Predicción final" if season=="2025/26" else "Backtest"}')
        ranking_tab(season,rankings.get(season),dataset)
with tabs[-1]: comparison_tab(dataset)
st.divider()
st.caption('La fila global suma estadísticas acumulables entre competiciones. Las tasas y métricas por 90 se promedian, excepto algunas tasas que se recalculan cuando el dataset incluye numerador y denominador compatibles. Revisa estas agregaciones si las definiciones de SofaScore difieren.')
