import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ============================================================================
# CONFIGURACIÓN DE COLORES Y ESTILO
# ============================================================================
COLOR_MAP = {
    'Breakdown - Mechanical': '#00BCD4',       # Cyan Unilever
    'Breakdown - Electrical': '#9C27B0',       # Morado Unilever
    'Breakdown - Instrumentation': '#673AB7',  # Morado oscuro
    'Breakdown - Process': '#4CAF50',          # Verde
    'Predicción (Alerta)': '#FF9800',          # Naranja (Tu petición)
    'Otros': '#9E9E9E'                         # Gris
}

st.set_page_config(page_title="Monitor de Líneas & Predicciones", page_icon="🏭", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1A237E 0%, #00BCD4 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONEXIONES (A PRUEBA DE BALAS)
# ============================================================================
@st.cache_resource
def get_connections():
    conns = {}
    for env in ['DB_PROD', 'DB_DEV']:
        try:
            host = os.getenv(f'{env}_HOST')
            user = os.getenv(f'{env}_USER')
            password = os.getenv(f'{env}_PASSWORD')
            database = os.getenv(f'{env}_NAME')
            port = os.getenv(f'{env}_PORT', '3306')
            if host:
                str_conn = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
                engine = create_engine(str_conn, pool_pre_ping=True, pool_recycle=3600,
                                     connect_args={'ssl': {'fake_flag_to_enable_tls': True}})
                conns[env] = engine
        except Exception as e:
            st.error(f"Error conectando a {env}: {e}")
    return conns

# ============================================================================
# FUNCIONES SQL OPTIMIZADAS
# ============================================================================

# 1. ESTRUCTURA (Fábricas y Líneas)
@st.cache_data(ttl=600)
def get_structure_data(_engine_prod):
    query = text("""
        SELECT DISTINCT f.de_factory, l.id_linea, l.de_linea
        FROM bui_line l
        JOIN bui_factory f ON l.id_fabrica = f.id_factory
        INNER JOIN dfos_linea d ON l.id_linea = d.idLineaBA
        WHERE l.fl_activo = 1 AND l.fl_eliminado = 0
        ORDER BY f.de_factory, l.de_linea
    """)
    return pd.read_sql(query, _engine_prod)

# 2. COMPARADOR DE LÍNEAS (ARREGLADO)
@st.cache_data(ttl=600)
def get_lines_comparison(_engine_prod, line_ids_tuple, days=30):
    if not line_ids_tuple: return pd.DataFrame()
    
    # Truco: Convertimos la tupla a string para evitar errores del driver con IN ()
    ids_str = ','.join(map(str, line_ids_tuple))
    
    query = text(f"""
        SELECT 
            l.de_linea,
            COUNT(p.id_perdida) as num_paradas,
            COALESCE(SUM(TIMESTAMPDIFF(MINUTE, p.fe_inicio, p.fe_fin)), 0) as minutos_totales,
            COALESCE(AVG(TIMESTAMPDIFF(MINUTE, p.fe_inicio, p.fe_fin)), 0) as tiempo_medio
        FROM bui_line l
        LEFT JOIN bui_perdida p ON l.id_linea = p.id_linea 
            AND p.de_perdida_2 LIKE '%Breakdown%'
            AND p.fe_inicio >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
        WHERE l.id_linea IN ({ids_str}) 
        GROUP BY l.id_linea, l.de_linea
        ORDER BY num_paradas DESC
    """)
    return pd.read_sql(query, _engine_prod, params={"days": days})

# 3. DATOS PARA LÍNEA TEMPORAL (GANTT) - REAL + PREDICCIÓN
@st.cache_data(ttl=300)
def get_timeline_data(_engine_prod, _engine_dev, id_linea, days=7):
    # A) Obtener Averías Reales (PROD)
    query_real = text("""
        SELECT 
            fe_inicio as Start,
            fe_fin as Finish,
            de_perdida_3 as Tipo,
            TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin) as Duracion,
            'Realidad' as Origen
        FROM bui_perdida
        WHERE id_linea = :id
          AND de_perdida_2 LIKE '%Breakdown%'
          AND fe_inicio >= DATE_SUB(NOW(), INTERVAL :days DAY)
    """)
    df_real = pd.read_sql(query_real, _engine_prod, params={"id": id_linea, "days": days})
    
    # B) Obtener Predicciones/Alertas (DEV)
    query_pred = text("""
        SELECT 
            fe_ventana as Start,
            DATE_ADD(fe_ventana, INTERVAL 1 HOUR) as Finish,
            'Predicción (Alerta)' as Tipo,
            60 as Duracion,
            'Modelo IA' as Origen
        FROM bui_predicciones_hora
        WHERE id_linea = :id
          AND fl_pred_modelo = 1
          AND fe_ventana >= DATE_SUB(NOW(), INTERVAL :days DAY)
    """)
    df_pred = pd.read_sql(query_pred, _engine_dev, params={"id": id_linea, "days": days})
    
    # Unir todo
    return pd.concat([df_real, df_pred], ignore_index=True)

# 4. CALENDARIO DE CALOR (RESTORED FROM V2)
@st.cache_data(ttl=600)
def get_calendar_heatmap(_engine_prod, id_linea, days=60):
    query = text("""
        SELECT DATE(fe_inicio) as fecha, 
               SUM(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as minutos
        FROM bui_perdida
        WHERE id_linea = :id 
          AND de_perdida_2 LIKE '%Breakdown%'
          AND fe_inicio >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
        GROUP BY DATE(fe_inicio)
    """)
    return pd.read_sql(query, _engine_prod, params={"id": id_linea, "days": days})

# ============================================================================
# APP PRINCIPAL
# ============================================================================
def main():
    conns = get_connections()
    if not conns: st.stop()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60/00BCD4/FFFFFF?text=Unilever+Ops", use_container_width=True)
        st.markdown("### ⚙️ Panel de Control")
        
        df_struct = get_structure_data(conns['DB_PROD'])
        
        # Filtros
        all_factories = sorted(df_struct['de_factory'].unique())
        sel_factory = st.selectbox("🏭 Fábrica", all_factories)
        
        lines_df = df_struct[df_struct['de_factory'] == sel_factory]
        lines_map = dict(zip(lines_df['de_linea'], lines_df['id_linea']))
        
        # Selector PRINCIPAL
        sel_line_name = st.selectbox("📍 Línea a Analizar", sorted(lines_map.keys()))
        sel_line_id = lines_map[sel_line_name]
        
        st.markdown("---")
        days = st.slider("📅 Días a visualizar", 1, 30, 7)
        
        # Selector COMPARADOR
        st.markdown("### 🆚 Comparador")
        multi_lines = st.multiselect("Añadir líneas para comparar:", sorted(lines_map.keys()))
        
    # --- HEADER ---
    st.markdown(f"""
    <div class="main-header">
        <h2 style="margin:0">📊 Visión General: {sel_line_name}</h2>
    </div>
    """, unsafe_allow_html=True)

    # --- PESTAÑAS ---
    tab_timeline, tab_analysis, tab_compare = st.tabs([
        "⏱️ Cronología (Timeline)", 
        "🍩 Análisis de Causas", 
        "🆚 Comparador de Líneas"
    ])

    # ------------------------------------------------------------------------
    # TAB 1: CRONOLOGÍA (LO QUE PEDISTE)
    # ------------------------------------------------------------------------
    with tab_timeline:
        st.subheader("Línea de Tiempo: Realidad vs Predicciones")
        st.caption("Barra Naranja = Alerta del Modelo | Barras de Colores = Averías Reales")
        
        df_time = get_timeline_data(conns['DB_PROD'], conns['DB_DEV'], sel_line_id, days)
        
        if not df_time.empty:
            # Ordenar para que el gráfico salga bonito
            df_time = df_time.sort_values('Start')
            
            # Crear el Gantt
            fig_gantt = px.timeline(
                df_time, 
                x_start="Start", 
                x_end="Finish", 
                y="Origen",  # Eje Y: Realidad vs Modelo
                color="Tipo", # Color: Tipo de avería o Alerta
                hover_data=["Duracion", "Tipo"],
                color_discrete_map=COLOR_MAP,
                height=400,
                title=f"Eventos en los últimos {days} días"
            )
            
            # Personalizar eje X y colores
            fig_gantt.update_yaxes(autorange="reversed") # Para que 'Realidad' salga arriba si quieres
            fig_gantt.update_layout(
                xaxis_title="Hora/Día",
                legend_title="Tipo de Evento",
                bargap=0.2
            )
            st.plotly_chart(fig_gantt, use_container_width=True)
        else:
            st.info("🎉 No hay averías ni alertas en este periodo.")

        # VOLVEMOS A PONER EL CALENDARIO (LO QUE DESAPARECIÓ)
        st.markdown("#### 📅 Calendario de Intensidad")
        df_cal = get_calendar_heatmap(conns['DB_PROD'], sel_line_id, 60)
        if not df_cal.empty:
            df_cal['fecha'] = pd.to_datetime(df_cal['fecha'])
            fig_cal = px.density_heatmap(
                df_cal, x='fecha', y=[1]*len(df_cal), z='minutos',
                nbinsx=60, color_continuous_scale='Teal',
                title="Intensidad de Paradas (Últimos 2 meses)",
                labels={'minutos': 'Minutos Perdidos'}
            )
            fig_cal.update_layout(height=250, yaxis_visible=False)
            st.plotly_chart(fig_cal, use_container_width=True)

    # ------------------------------------------------------------------------
    # TAB 2: ANÁLISIS (DONUT + PARETO)
    # ------------------------------------------------------------------------
    with tab_analysis:
        col1, col2 = st.columns(2)
        
        # Consulta rápida para análisis
        query_analysis = text("""
            SELECT de_perdida_3 as Causa, 
                   COUNT(*) as Frecuencia,
                   SUM(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as Minutos
            FROM bui_perdida
            WHERE id_linea = :id AND de_perdida_2 LIKE '%Breakdown%'
              AND fe_inicio >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
            GROUP BY de_perdida_3
            ORDER BY Minutos DESC
        """)
        df_an = pd.read_sql(query_analysis, conns['DB_PROD'], params={"id": sel_line_id, "days": days})
        
        if not df_an.empty:
            with col1:
                st.markdown("##### 🍩 Distribución de Tiempo")
                fig_pie = px.pie(df_an, values='Minutos', names='Causa', hole=0.4,
                               color_discrete_sequence=px.colors.qualitative.Prism)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                st.markdown("##### 🏆 Top Causas (Pareto)")
                fig_bar = px.bar(df_an.head(5), x='Frecuencia', y='Causa', orientation='h',
                               color='Minutos', color_continuous_scale='Bluered')
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Sin datos suficientes para análisis.")

    # ------------------------------------------------------------------------
    # TAB 3: COMPARADOR (ARREGLADO)
    # ------------------------------------------------------------------------
    with tab_compare:
        if multi_lines:
            # Convertimos nombres a IDs
            ids_to_compare = [lines_map[name] for name in multi_lines]
            
            # Importante: Pasamos tupla
            df_comp = get_lines_comparison(conns['DB_PROD'], tuple(ids_to_compare), days)
            
            if not df_comp.empty:
                st.markdown(f"### Comparativa ({days} días)")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(df_comp.style.background_gradient(cmap='Reds', subset=['minutos_totales']), 
                               use_container_width=True)
                with c2:
                    fig_comp = px.bar(df_comp, x='de_linea', y='minutos_totales', 
                                    color='num_paradas', title="Minutos Totales Perdidos")
                    st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("Las líneas seleccionadas no tienen datos en este periodo.")
        else:
            st.info("👈 Selecciona líneas en la barra lateral ('Comparador') para ver datos aquí.")

if __name__ == "__main__":
    main()