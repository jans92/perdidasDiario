"""
dashboard_hybrid.py - Versión Híbrida Corregida FINAL
=====================================================
FIXES INTEGRADOS:
- Clasificación de Falsas Alarmas (Color Rojo)
- Integración de Causa Raíz (SHAP) en Timeline
- Protección contra errores NaN en el ancho de barras (width)
- Manteniendo 100% de la funcionalidad de análisis y comparación
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from io import BytesIO

load_dotenv()

# ============================================================================
# CONFIGURACIÓN DE COLORES
# ============================================================================
COLOR_MAP = {
    'Breakdown - Mechanical': '#00BCD4',
    'Breakdown - Electrical': '#9C27B0',
    'Breakdown - Instrumentation': '#673AB7',
    'Breakdown - Process': '#4CAF50',
    'Predicción (Alerta)': '#FF9800', # Naranja
    'Predicción Acertada': '#66BB6A', # Verde
    'Falsa Alarma': '#EF5350',        # Rojo
    'Otros': '#9E9E9E'
}

# ============================================================================
# CONFIGURACIÓN STREAMLIT
# ============================================================================
st.set_page_config(page_title="Monitor Predictivo - Unilever", page_icon="🏭", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main-header { background: linear-gradient(135deg, #9C27B0 0%, #00BCD4 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 25px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15); }
    .main-header h2 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { margin: 8px 0 0 0; font-size: 0.95rem; opacity: 0.9; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #00BCD4; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONEXIONES A BD
# ============================================================================
@st.cache_resource(ttl=3600)
def get_connections():
    conns = {}
    for env in ['DB_PROD', 'DB_DEV']:
        try:
            host = os.getenv(f'{env}_HOST')
            user = os.getenv(f'{env}_USER')
            password = os.getenv(f'{env}_PASSWORD')
            database = os.getenv(f'{env}_NAME')
            port = os.getenv(f'{env}_PORT', '3306')
            
            if not all([host, user, password, database]): continue
            
            conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(conn_str, pool_pre_ping=True, pool_recycle=3600,
                                   connect_args={'ssl': {'fake_flag_to_enable_tls': True}})
            conns[env] = engine
        except Exception as e:
            st.error(f"❌ Error conectando a {env}: {e}")
    return conns

# ============================================================================
# FUNCIONES DE DATOS
# ============================================================================

@st.cache_data(ttl=600)
def get_structure_data(_engine_prod):
    query = text("SELECT DISTINCT f.id_factory, f.de_factory, l.id_linea, l.de_linea FROM bui_line l INNER JOIN bui_factory f ON l.id_fabrica = f.id_factory WHERE l.fl_activo = 1 AND l.fl_eliminado = 0 ORDER BY f.de_factory, l.de_linea")
    return pd.read_sql(query, _engine_prod)

@st.cache_data(ttl=300)
def get_timeline_data(_engine_prod, _engine_dev, id_linea, fecha_inicio, fecha_fin):
    # A) BREAKDOWNS REALES
    query_real = text("""
        SELECT fe_inicio as Start, fe_fin as Finish,
            CASE 
                WHEN de_perdida_3 LIKE '%Mechanical%' OR de_perdida_3 LIKE '%Mechanic%' THEN 'Breakdown - Mechanical'
                WHEN de_perdida_3 LIKE '%Electrical%' OR de_perdida_3 LIKE '%Electric%' THEN 'Breakdown - Electrical'
                WHEN de_perdida_3 LIKE '%Instrumentation%' OR de_perdida_3 LIKE '%Instrument%' THEN 'Breakdown - Instrumentation'
                WHEN de_perdida_3 LIKE '%Process%' THEN 'Breakdown - Process'
                ELSE 'Otros'
            END as Tipo,
            id_maquina_dfos, de_maquina_dfos, TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin) as Duracion,
            CONCAT(de_perdida_3, ' - ', TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin), ' min') as Detalle,
            'Breakdown Real' as Origen
        FROM bui_perdida
        WHERE id_linea = :id_linea AND de_perdida_2 <> '' AND DATE(fe_inicio) BETWEEN :fecha_inicio AND :fecha_fin
        ORDER BY fe_inicio
    """)
    df_real = pd.read_sql(query_real, _engine_prod, params={"id_linea": id_linea, "fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin})
    
    # B) PREDICCIONES (FIX: Añadido de_causas_raiz)
    query_pred = text("""
        SELECT fe_ventana, id_maquina_dfos, nm_score, fl_pred_modelo, fl_target_real, fl_acierto, de_causas_raiz
        FROM bui_predicciones_hora
        WHERE id_linea = :id_linea AND DATE(fe_ventana) BETWEEN :fecha_inicio AND :fecha_fin AND fl_pred_modelo = 1
        ORDER BY fe_ventana
    """)
    df_pred_raw = pd.read_sql(query_pred, _engine_dev, params={"id_linea": id_linea, "fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin})
    
    df_pred = pd.DataFrame()
    if not df_pred_raw.empty:
        df_pred = df_pred_raw.copy()
        df_pred['Start'] = pd.to_datetime(df_pred['fe_ventana'])
        df_pred['Finish'] = df_pred['Start'] + pd.Timedelta(hours=1)
        
        # FIX LOGICA ROJO (Falsa Alarma)
        df_pred['Tipo'] = df_pred.apply(
            lambda row: 'Predicción Acertada' if row['fl_target_real'] == 1 
                       else ('Falsa Alarma' if row['fl_target_real'] == 0 
                             else 'Predicción (Alerta)'),
            axis=1
        )
        
        df_pred['Detalle'] = 'Score: ' + df_pred['nm_score'].round(3).astype(str)
        df_pred['de_causas_raiz'] = df_pred['de_causas_raiz'].fillna("Normal")
        df_pred['de_maquina_dfos'] = df_pred['id_maquina_dfos']
        df_pred['Duracion'] = 60
        df_pred['Origen'] = 'Modelo IA'
    
    return pd.concat([df_real, df_pred], ignore_index=True)

@st.cache_data(ttl=600)
def get_breakdown_analysis(_engine_prod, id_linea, days):
    query = text("SELECT CASE WHEN de_perdida_3 LIKE '%Mechanical%' THEN 'Mechanical' WHEN de_perdida_3 LIKE '%Electrical%' THEN 'Electrical' WHEN de_perdida_3 LIKE '%Instrumentation%' THEN 'Instrumentation' WHEN de_perdida_3 LIKE '%Process%' THEN 'Process' ELSE 'Otros' END as Categoria, de_perdida_3 as Causa, COUNT(*) as Frecuencia, SUM(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as Minutos_Totales, AVG(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as Duracion_Promedio FROM bui_perdida WHERE id_linea = :id_linea AND de_perdida_2 <> '' AND fe_inicio >= DATE_SUB(NOW(), INTERVAL :days DAY) GROUP BY de_perdida_3 ORDER BY Minutos_Totales DESC")
    return pd.read_sql(query, _engine_prod, params={"id_linea": id_linea, "days": days})

@st.cache_data(ttl=600)
def get_calendar_heatmap(_engine_prod, id_linea, days):
    query = text("SELECT DATE(fe_inicio) as fecha, SUM(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as minutos FROM bui_perdida WHERE id_linea = :id_linea AND de_perdida_2 <> '' AND fe_inicio >= DATE_SUB(NOW(), INTERVAL :days DAY) GROUP BY DATE(fe_inicio) ORDER BY fecha")
    return pd.read_sql(query, _engine_prod, params={"id_linea": id_linea, "days": days})

@st.cache_data(ttl=600)
def get_lines_comparison(_engine_prod, id_lineas, days):
    query = text("SELECT l.de_linea as Linea, COUNT(*) as Num_Breakdowns, SUM(TIMESTAMPDIFF(MINUTE, p.fe_inicio, p.fe_fin)) as Minutos_Totales, AVG(TIMESTAMPDIFF(MINUTE, p.fe_inicio, p.fe_fin)) as Duracion_Promedio FROM bui_perdida p INNER JOIN bui_line l ON p.id_linea = l.id_linea WHERE p.id_linea IN :id_lineas AND p.de_perdida_2 = 'Breakdown & Equipment Failure Time' AND p.fe_inicio >= DATE_SUB(NOW(), INTERVAL :days DAY) GROUP BY l.de_linea ORDER BY Minutos_Totales DESC")
    return pd.read_sql(query, _engine_prod, params={"id_lineas": tuple(id_lineas), "days": days})

def export_to_excel(df_timeline, df_analysis):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_timeline.to_excel(writer, sheet_name='Timeline', index=False)
        if not df_analysis.empty: df_analysis.to_excel(writer, sheet_name='Análisis', index=False)
    output.seek(0)
    return output

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.markdown('<div class="main-header"><h2>🏭 Monitor Predictivo de Mantenimiento</h2><p>Análisis de Fallas y Predicciones en Tiempo Real</p></div>', unsafe_allow_html=True)
    conns = get_connections()
    if 'DB_PROD' not in conns: return
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        df_struct = get_structure_data(conns['DB_PROD'])
        factories = df_struct['de_factory'].unique().tolist()
        sel_factory = st.selectbox("Fábrica", factories, index=factories.index("Veszprém") if "Veszprém" in factories else 0)
        lines_df = df_struct[df_struct['de_factory'] == sel_factory]
        lines_list = lines_df['de_linea'].tolist()
        sel_line = st.selectbox("Línea", lines_list, index=lines_list.index("BALATON") if "BALATON" in lines_list else 0)
        sel_line_id = int(lines_df[lines_df['de_linea'] == sel_line]['id_linea'].iloc[0])
        
        col_dates1, col_dates2 = st.columns(2)
        with col_dates1: fecha_desde = st.date_input("Desde", value=datetime.now() - timedelta(days=8), max_value=datetime.now())
        with col_dates2: fecha_hasta = st.date_input("Hasta", value=datetime.now() + timedelta(days=1))
        
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        
        lines_map = {row['de_linea']: row['id_linea'] for _, row in lines_df.iterrows()}
        multi_lines = st.multiselect("Líneas a comparar", options=list(lines_map.keys()), default=None)

    tab1, tab2, tab3 = st.tabs(["📈 Timeline (Causa Raíz)", "📊 Análisis de Causas", "🆚 Comparador de Líneas"])
    
    with tab1:
        st.markdown(f"### 📅 Eventos en **{sel_line}**")
        try:
            df_timeline = get_timeline_data(conns['DB_PROD'], conns.get('DB_DEV'), sel_line_id, fecha_desde, fecha_hasta)
            if not df_timeline.empty:
                # KPIs
                c1, c2, c3 = st.columns(3)
                num_breaks = len(df_timeline[df_timeline['Origen'] == 'Breakdown Real'])
                c1.metric("🔴 EVENTOS REALES", f"{num_breaks}")
                c2.warning(f"⚠️ ALERTAS: {len(df_timeline[df_timeline['Tipo'] == 'Predicción (Alerta)'])}")
                c3.success(f"✅ ACIERTOS: {len(df_timeline[df_timeline['Tipo'] == 'Predicción Acertada'])}")
                
                fig = go.Figure()
                # GRUPO 1: REALIDAD
                df_real_plot = df_timeline[df_timeline['Origen'] == 'Breakdown Real'].copy().dropna(subset=['Start', 'Finish'])
                if not df_real_plot.empty:
                    df_real_plot['width'] = (pd.to_datetime(df_real_plot['Finish']) - pd.to_datetime(df_real_plot['Start'])).dt.total_seconds() * 1000
                    df_real_plot.loc[df_real_plot['width'] <= 0, 'width'] = 900000 # 15 min min
                    for tipo in df_real_plot['Tipo'].unique():
                        df_tipo = df_real_plot[df_real_plot['Tipo'] == tipo]
                        fig.add_trace(go.Bar(x=df_tipo['Start'], y=[1]*len(df_tipo), width=df_tipo['width'], name=tipo, 
                                           marker_color=COLOR_MAP.get(tipo, '#9E9E9E'), hovertemplate='<b>%{customdata[0]}</b><br>Inicio: %{x}<extra></extra>',
                                           customdata=df_tipo[['Detalle']].values, yaxis='y'))
                
                # GRUPO 2: IA (FIX SHAP + COLOR ROJO)
                df_pred_plot = df_timeline[df_timeline['Origen'] == 'Modelo IA'].copy()
                if not df_pred_plot.empty:
                    df_pred_plot['width'] = 3600000 # 1 hora exacta
                    for tipo in df_pred_plot['Tipo'].unique():
                        df_tipo = df_pred_plot[df_pred_plot['Tipo'] == tipo]
                        fig.add_trace(go.Bar(x=df_tipo['Start'], y=[1]*len(df_tipo), width=df_tipo['width'], name=tipo, 
                                           marker_color=COLOR_MAP.get(tipo, '#FF9800'), 
                                           hovertemplate='<b>%{customdata[0]}</b><br>Hora: %{x}<br><b>Causa:</b> %{customdata[1]}<extra></extra>',
                                           customdata=df_tipo[['Detalle', 'de_causas_raiz']].values, yaxis='y2'))
                
                fig.update_layout(height=500, barmode='overlay', hovermode='x unified', plot_bgcolor='rgba(248,249,250,1)',
                                  yaxis=dict(title="Evento Real", domain=[0.6, 1], tickvals=[]),
                                  yaxis2=dict(title="Modelo IA", domain=[0, 0.4], tickvals=[], anchor='x'))
                st.plotly_chart(fig, use_container_width=True)
                
                # Botón Excel y Heatmap (Final de Tab 1)
                df_an_data = get_breakdown_analysis(conns['DB_PROD'], sel_line_id, (fecha_hasta - fecha_desde).days + 1)
                st.download_button("⬇️ Exportar Excel", export_to_excel(df_timeline, df_an_data), f"{sel_line}.xlsx")
                df_cal = get_calendar_heatmap(conns['DB_PROD'], sel_line_id, 60)
                if not df_cal.empty:
                    st.plotly_chart(px.density_heatmap(df_cal, x='fecha', y=[1]*len(df_cal), z='minutos', color_continuous_scale='Teal', title="Fallas 60d"), use_container_width=True)
            else: st.info("🎉 Sin eventos")
        except Exception as e: st.error(f"Error: {e}")

    with tab2:
        df_analysis = get_breakdown_analysis(conns['DB_PROD'], sel_line_id, (fecha_hasta - fecha_desde).days + 1)
        if not df_analysis.empty:
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df_analysis.groupby('Categoria').sum().reset_index(), values='Minutos_Totales', names='Categoria', hole=0.4), use_container_width=True)
            c2.plotly_chart(px.bar(df_analysis.nlargest(10, 'Minutos_Totales'), x='Frecuencia', y='Causa', orientation='h'), use_container_width=True)
            st.dataframe(df_analysis, use_container_width=True)

    with tab3:
        if multi_lines:
            df_comp = get_lines_comparison(conns['DB_PROD'], [int(lines_map[n]) for n in multi_lines], (fecha_hasta - fecha_desde).days + 1)
            if not df_comp.empty: st.plotly_chart(px.bar(df_comp, x='Linea', y='Minutos_Totales', text='Minutos_Totales'), use_container_width=True)

if __name__ == "__main__":
    main()