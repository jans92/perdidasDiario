"""
dashboard_hybrid.py - Versión Híbrida Corregida
===============================================
FIX CRÍTICO:
- Timeline muestra eventos por HORA (no días completos)
- Predicciones acertadas solo marcan las horas específicas
- Breakdowns mantienen su duración real
- Valores por defecto: Veszprém / BALATON
- Filtro SQL de_perdida_2 y visibilidad para paros menores
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
    # Tipos de Breakdown
    'Breakdown - Mechanical': '#00BCD4',       # Cyan Unilever
    'Breakdown - Electrical': '#9C27B0',       # Morado Unilever
    'Breakdown - Instrumentation': '#673AB7',  # Morado oscuro
    'Breakdown - Process': '#4CAF50',          # Verde
    
    # Predicciones
    'Predicción (Alerta)': '#FF9800',          # Naranja
    'Predicción Acertada': '#66BB6A',          # Verde claro
    'Falsa Alarma': '#EF5350',                 # Rojo claro
    
    # Otros
    'Otros': '#9E9E9E'                         # Gris
}

# ============================================================================
# CONFIGURACIÓN STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Monitor Predictivo - Unilever",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #9C27B0 0%, #00BCD4 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    .main-header h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 8px 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #00BCD4;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 12px;
        font-weight: 600;
        color: #616161;
        text-transform: uppercase;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px 10px 0 0;
        padding: 12px 20px;
        color: #616161;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00BCD4 0%, #9C27B0 100%);
        color: white;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00BCD4 0%, #00ACC1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 188, 212, 0.3);
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 16px rgba(0, 188, 212, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONEXIONES A BD
# ============================================================================
@st.cache_resource(ttl=3600)
def get_connections():
    """Conexión dual: PROD (realidad) + DEV (predicciones)"""
    conns = {}
    for env in ['DB_PROD', 'DB_DEV']:
        try:
            host = os.getenv(f'{env}_HOST')
            user = os.getenv(f'{env}_USER')
            password = os.getenv(f'{env}_PASSWORD')
            database = os.getenv(f'{env}_NAME')
            port = os.getenv(f'{env}_PORT', '3306')
            
            if not all([host, user, password, database]):
                st.error(f"❌ Faltan variables de entorno para {env}")
                continue
            
            conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(
                conn_str,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=10,
                max_overflow=20,
                connect_args={'ssl': {'ssl_mode': 'PREFERRED'}}
            )
            
            # Test conexión
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            conns[env] = engine
            
        except Exception as e:
            st.error(f"❌ Error conectando a {env}: {e}")
    
    return conns

# ============================================================================
# FUNCIONES DE DATOS
# ============================================================================

@st.cache_data(ttl=600)
def get_structure_data(_engine_prod):
    """Obtiene fábricas y líneas disponibles"""
    query = text("""
        SELECT DISTINCT 
            f.id_factory,
            f.de_factory,
            l.id_linea,
            l.de_linea
        FROM bui_line l
        INNER JOIN bui_factory f ON l.id_fabrica = f.id_factory
        WHERE l.fl_activo = 1 
          AND l.fl_eliminado = 0
        ORDER BY f.de_factory, l.de_linea
    """)
    return pd.read_sql(query, _engine_prod)

@st.cache_data(ttl=300)
def get_timeline_data(_engine_prod, _engine_dev, id_linea, fecha_inicio, fecha_fin):
    """
    Timeline CORREGIDO:
    - Breakdowns reales con duración exacta
    - Predicciones: 1 barra por HORA (no por día)
    - Aciertos: solo las horas específicas donde acertó
    """
    
    # A) BREAKDOWNS REALES con clasificación por tipo
    query_real = text("""
        SELECT 
            fe_inicio as Start,
            fe_fin as Finish,
            CASE 
                WHEN de_perdida_3 LIKE '%Mechanical%' OR de_perdida_3 LIKE '%Mechanic%' 
                    THEN 'Breakdown - Mechanical'
                WHEN de_perdida_3 LIKE '%Electrical%' OR de_perdida_3 LIKE '%Electric%' 
                    THEN 'Breakdown - Electrical'
                WHEN de_perdida_3 LIKE '%Instrumentation%' OR de_perdida_3 LIKE '%Instrument%' 
                    THEN 'Breakdown - Instrumentation'
                WHEN de_perdida_3 LIKE '%Process%' 
                    THEN 'Breakdown - Process'
                ELSE 'Otros'
            END as Tipo,
            id_maquina_dfos,
            de_maquina_dfos,
            TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin) as Duracion,
            CONCAT(de_perdida_3, ' - ', TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin), ' min') as Detalle,
            'Breakdown Real' as Origen
        FROM bui_perdida
        WHERE id_linea = :id_linea
          AND de_perdida_2 <> '' 
          AND DATE(fe_inicio) BETWEEN :fecha_inicio AND :fecha_fin
        ORDER BY fe_inicio
    """)
    
    df_real = pd.read_sql(query_real, _engine_prod, params={
        "id_linea": id_linea,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin
    })
    
    # B) PREDICCIONES: POR HORA (NO POR DÍA)
    query_pred = text("""
        SELECT 
            fe_ventana,
            id_maquina_dfos,
            nm_score,
            fl_pred_modelo,
            fl_target_real,
            fl_acierto
        FROM bui_predicciones_hora_dia
        WHERE id_linea = :id_linea
          AND DATE(fe_ventana) BETWEEN :fecha_inicio AND :fecha_fin
          AND fl_pred_modelo = 1
        ORDER BY fe_ventana
    """)
    
    df_pred_raw = pd.read_sql(query_pred, _engine_dev, params={
        "id_linea": id_linea,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin
    })
    
    # Convertir predicciones a formato timeline POR HORA
    df_pred = pd.DataFrame()
    
    if not df_pred_raw.empty:
        df_pred_raw['fe_ventana'] = pd.to_datetime(df_pred_raw['fe_ventana'])
        
        # CRÍTICO: Mantener granularidad por HORA
        df_pred = df_pred_raw.copy()
        
        # Cada predicción dura 1 HORA (no 1 día)
        df_pred['Start'] = df_pred['fe_ventana']
        df_pred['Finish'] = df_pred['fe_ventana'] + pd.Timedelta(hours=1)
        
        # Clasificar según si acertó
        df_pred['Tipo'] = df_pred.apply(
            lambda row: 'Predicción Acertada' if row['fl_target_real'] == 1 
                       else 'Predicción (Alerta)',
            axis=1
        )
        
        df_pred['Detalle'] = 'Score: ' + df_pred['nm_score'].round(3).astype(str)
        df_pred['de_maquina_dfos'] = df_pred['id_maquina_dfos']
        df_pred['Duracion'] = 60  # 60 minutos
        df_pred['Origen'] = 'Modelo IA'
    
    # Combinar ambos DataFrames
    df_timeline = pd.concat([df_real, df_pred], ignore_index=True)
    
    return df_timeline

@st.cache_data(ttl=600)
def get_breakdown_analysis(_engine_prod, id_linea, days):
    """Análisis detallado de causas de breakdowns"""
    query = text("""
        SELECT 
            CASE 
                WHEN de_perdida_3 LIKE '%Mechanical%' THEN 'Mechanical'
                WHEN de_perdida_3 LIKE '%Electrical%' THEN 'Electrical'
                WHEN de_perdida_3 LIKE '%Instrumentation%' THEN 'Instrumentation'
                WHEN de_perdida_3 LIKE '%Process%' THEN 'Process'
                ELSE 'Otros'
            END as Categoria,
            de_perdida_3 as Causa,
            COUNT(*) as Frecuencia,
            SUM(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as Minutos_Totales,
            AVG(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as Duracion_Promedio
        FROM bui_perdida
        WHERE id_linea = :id_linea
          AND de_perdida_2 <> ''
          AND fe_inicio >= DATE_SUB(NOW(), INTERVAL :days DAY)
        GROUP BY de_perdida_3
        ORDER BY Minutos_Totales DESC
    """)
    
    return pd.read_sql(query, _engine_prod, params={"id_linea": id_linea, "days": days})

@st.cache_data(ttl=600)
def get_calendar_heatmap(_engine_prod, id_linea, days):
    """Calendario de calor de breakdowns"""
    query = text("""
        SELECT 
            DATE(fe_inicio) as fecha,
            SUM(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as minutos
        FROM bui_perdida
        WHERE id_linea = :id_linea
          AND de_perdida_2 <> ''
          AND fe_inicio >= DATE_SUB(NOW(), INTERVAL :days DAY)
        GROUP BY DATE(fe_inicio)
        ORDER BY fecha
    """)
    
    return pd.read_sql(query, _engine_prod, params={"id_linea": id_linea, "days": days})

@st.cache_data(ttl=600)
def get_lines_comparison(_engine_prod, id_lineas, days):
    """Comparación entre múltiples líneas"""
    query = text("""
        SELECT 
            l.de_linea as Linea,
            COUNT(*) as Num_Breakdowns,
            SUM(TIMESTAMPDIFF(MINUTE, p.fe_inicio, p.fe_fin)) as Minutos_Totales,
            AVG(TIMESTAMPDIFF(MINUTE, p.fe_inicio, p.fe_fin)) as Duracion_Promedio
        FROM bui_perdida p
        INNER JOIN bui_line l ON p.id_linea = l.id_linea
        WHERE p.id_linea IN :id_lineas
          AND p.de_perdida_2 = 'Breakdown & Equipment Failure Time'
          AND p.fe_inicio >= DATE_SUB(NOW(), INTERVAL :days DAY)
        GROUP BY l.de_linea
        ORDER BY Minutos_Totales DESC
    """)
    
    return pd.read_sql(query, _engine_prod, params={"id_lineas": tuple(id_lineas), "days": days})

def export_to_excel(df_timeline, df_analysis):
    """Exporta datos a Excel"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_timeline.to_excel(writer, sheet_name='Timeline', index=False)
        if not df_analysis.empty:
            df_analysis.to_excel(writer, sheet_name='Análisis', index=False)
    
    output.seek(0)
    return output

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h2>🏭 Monitor Predictivo de Mantenimiento</h2>
        <p>Sistema de análisis en tiempo real de breakdowns y predicciones IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Conexiones
    conns = get_connections()
    
    if 'DB_PROD' not in conns:
        st.error("❌ No se pudo conectar a la base de datos de producción")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        
        # Selector de fábrica/línea
        st.markdown("### 🏭 Fábrica")
        df_structure = get_structure_data(conns['DB_PROD'])
        
        if df_structure.empty:
            st.error("No hay líneas disponibles")
            return
        
        factories = df_structure['de_factory'].unique().tolist()
        # VALOR POR DEFECTO: Veszprém
        default_fac_idx = factories.index("Veszprém") if "Veszprém" in factories else 0
        sel_factory = st.selectbox("Selecciona fábrica", factories, index=default_fac_idx)
        
        # Filtrar líneas por fábrica
        lines_df = df_structure[df_structure['de_factory'] == sel_factory]
        
        st.markdown("### 📊 Línea a Analizar")
        lines_list = lines_df['de_linea'].tolist()
        # VALOR POR DEFECTO: BALATON
        default_line_idx = lines_list.index("BALATON") if "BALATON" in lines_list else 0
        sel_line = st.selectbox(
            "Selecciona línea",
            lines_list,
            index=default_line_idx
        )
        
        sel_line_id = int(lines_df[lines_df['de_linea'] == sel_line]['id_linea'].iloc[0])
        
        # Período de análisis
        st.markdown("### 📅 Período de Análisis")
        
        col_dates1, col_dates2 = st.columns(2)
        
        with col_dates1:
            fecha_desde = st.date_input(
                "Desde",
                value=datetime.now() - timedelta(days=8),
                max_value=datetime.now()
            )
        
        with col_dates2:
            fecha_hasta = st.date_input(
                "Hasta",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        days_range = (fecha_hasta - fecha_desde).days + 1
        
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Comparador de líneas
        st.markdown("---")
        st.markdown("### 🆚 Comparador")
        
        lines_map = {row['de_linea']: row['id_linea'] for _, row in lines_df.iterrows()}
        
        multi_lines = st.multiselect(
            "Líneas a comparar",
            options=list(lines_map.keys()),
            default=None
        )
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📈 Timeline (Realidad vs Predicciones)", "📊 Análisis de Causas", "🆚 Comparador de Líneas"])
    
    # ========================================================================
    # TAB 1: TIMELINE
    # ========================================================================
    with tab1:
        st.markdown(f"### 📅 Eventos en **{sel_line}** ({days_range} días)")
        
        try:
            df_timeline = get_timeline_data(
                conns['DB_PROD'],
                conns.get('DB_DEV'),
                sel_line_id,
                fecha_desde,
                fecha_hasta
            )
            
            if not df_timeline.empty:
                # KPIs
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    num_breaks = len(df_timeline[df_timeline['Origen'] == 'Breakdown Real'])
                    total_min = df_timeline[df_timeline['Origen'] == 'Breakdown Real']['Duracion'].sum()
                    st.metric(
                        "🔴 EVENTOS REALES",
                        f"{num_breaks}",
                        f"{total_min:,.0f} min" if total_min > 0 else "0 min"
                    )
                
                with col_sum2:
                    num_pred = len(df_timeline[df_timeline['Tipo'] == 'Predicción (Alerta)'])
                    st.warning(f"⚠️ **HORAS CON ALERTA:** {num_pred}")
                
                with col_sum3:
                    num_acertadas = len(df_timeline[df_timeline['Tipo'] == 'Predicción Acertada'])
                    if num_pred > 0:
                        accuracy = (num_acertadas / num_pred) * 100
                        st.success(f"✅ **Aciertos:** {num_acertadas} ({accuracy:.1f}%)")
                    else:
                        st.success("✅ **Aciertos:** N/A")
                
                # Timeline gráfico
                st.markdown("---")
                
                # Convertir a datetime
                df_timeline['Start'] = pd.to_datetime(df_timeline['Start'])
                df_timeline['Finish'] = pd.to_datetime(df_timeline['Finish'])
                
                # Crear figura con dos ejes Y
                fig = go.Figure()
                
                # GRUPO 1: Breakdowns reales (eje Y inferior)
                df_real_plot = df_timeline[df_timeline['Origen'] == 'Breakdown Real'].copy()
                if not df_real_plot.empty:
                    # Normalizar width
                    mask_na_finish = df_real_plot['Finish'].isna()
                    df_real_plot.loc[mask_na_finish, 'Finish'] = df_real_plot.loc[mask_na_finish, 'Start'] + pd.Timedelta(minutes=1)
                    df_real_plot['width'] = (df_real_plot['Finish'] - df_real_plot['Start']).dt.total_seconds() * 1000
                    
                    # FIX: Forzar a que cualquier pérdida dure visualmente al menos 15 mins (900,000 ms)
                    df_real_plot['width'] = df_real_plot['width'].fillna(900000)
                    df_real_plot.loc[df_real_plot['width'] <= 900000, 'width'] = 900000
                    
                    for tipo in df_real_plot['Tipo'].unique():
                        df_tipo = df_real_plot[df_real_plot['Tipo'] == tipo]
                        
                        fig.add_trace(go.Bar(
                            x=df_tipo['Start'],
                            y=[1] * len(df_tipo),
                            base=[0] * len(df_tipo),
                            width=df_tipo['width'],
                            name=tipo,
                            marker_color=COLOR_MAP.get(tipo, '#9E9E9E'),
                            hovertemplate='<b>%{customdata[0]}</b><br>Inicio: %{x}<br>Duración real: %{customdata[1]} min<extra></extra>',
                            customdata=df_tipo[['Detalle', 'Duracion']].values,
                            yaxis='y',
                            showlegend=True
                        ))
                
                # GRUPO 2: Predicciones (eje Y superior)
                df_pred_plot = df_timeline[df_timeline['Origen'] == 'Modelo IA'].copy()
                if not df_pred_plot.empty:
                    mask_na_finish = df_pred_plot['Finish'].isna()
                    df_pred_plot.loc[mask_na_finish, 'Finish'] = df_pred_plot.loc[mask_na_finish, 'Start'] + pd.Timedelta(hours=1)
                    df_pred_plot['width'] = (df_pred_plot['Finish'] - df_pred_plot['Start']).dt.total_seconds() * 1000
                    df_pred_plot['width'] = df_pred_plot['width'].fillna(3600000)
                    df_pred_plot.loc[df_pred_plot['width'] <= 0, 'width'] = 3600000  # 1h en ms
                    
                    for tipo in df_pred_plot['Tipo'].unique():
                        df_tipo = df_pred_plot[df_pred_plot['Tipo'] == tipo]
                        
                        fig.add_trace(go.Bar(
                            x=df_tipo['Start'],
                            y=[1] * len(df_tipo),
                            base=[0] * len(df_tipo),
                            width=df_tipo['width'],
                            name=tipo,
                            marker_color=COLOR_MAP.get(tipo, '#FF9800'),
                            hovertemplate='<b>%{customdata[0]}</b><br>Hora: %{x}<extra></extra>',
                            customdata=df_tipo[['Detalle']].values,
                            yaxis='y2',
                            showlegend=True
                        ))
                
                # Layout con dos ejes Y
                fig.update_layout(
                    title=dict(
                        text=f"<b><b>Eventos en {sel_line} ({days_range} días)</b></b>",
                    ),
                    xaxis=dict(
                        title="Fecha/Hora",
                        gridcolor='#E0E0E0',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title="Evento Real",
                        range=[0, 2],
                        showgrid=False,
                        tickvals=[],
                        domain=[0.6, 1]
                    ),
                    yaxis2=dict(
                        title="Modelo IA",
                        range=[0, 2],
                        showgrid=False,
                        tickvals=[],
                        domain=[0, 0.4],
                        anchor='x'
                    ),
                    barmode='overlay',
                    height=500,
                    hovermode='x unified',
                    plot_bgcolor='rgba(248,249,250,1)',
                    legend=dict(
                        title="Tipo de Evento",
                        bgcolor='rgba(255,255,255,1)',
                        bordercolor='rgba(0,0,0,0)'
                    ),
                    dragmode='zoom',
                )
                
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['fullscreen']
                })
                
                # Exportar
                st.markdown("---")
                col_exp1, col_exp2 = st.columns([3, 1])
                
                with col_exp2:
                    df_analysis = get_breakdown_analysis(conns['DB_PROD'], sel_line_id, days_range)
                    excel_data = export_to_excel(df_timeline, df_analysis)
                    
                    st.download_button(
                        label="⬇️ Exportar a Excel",
                        data=excel_data,
                        file_name=f"timeline_{sel_line}_{fecha_desde}_{fecha_hasta}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Calendario de calor
                st.markdown("---")
                st.markdown("#### 🔥 Calendario de Intensidad (últimos 60 días)")
                
                df_cal = get_calendar_heatmap(conns['DB_PROD'], sel_line_id, 60)
                
                if not df_cal.empty:
                    df_cal['fecha'] = pd.to_datetime(df_cal['fecha'])
                    
                    fig_cal = px.density_heatmap(
                        df_cal,
                        x='fecha',
                        y=[1]*len(df_cal),
                        z='minutos',
                        nbinsx=60,
                        color_continuous_scale='Teal',
                        title="Minutos perdidos por día",
                        labels={'minutos': 'Minutos'}
                    )
                    
                    fig_cal.update_layout(
                        height=250,
                        yaxis_visible=False,
                        plot_bgcolor='rgba(248,249,250,1)'
                    )
                    
                    st.plotly_chart(fig_cal, use_container_width=True)
                
            else:
                st.info("🎉 No hay eventos (ni breakdowns ni alertas) en este período")
        
        except Exception as e:
            st.error(f"❌ Error en timeline: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # ========================================================================
    # TAB 2: ANÁLISIS
    # ========================================================================
    with tab2:
        st.markdown("### 📊 Análisis Detallado de Causas")
        
        try:
            df_analysis = get_breakdown_analysis(conns['DB_PROD'], sel_line_id, days_range)
            
            if not df_analysis.empty:
                col_an1, col_an2 = st.columns(2)
                
                with col_an1:
                    st.markdown("#### 🍩 Distribución por Categoría")
                    
                    df_cat = df_analysis.groupby('Categoria').agg({
                        'Frecuencia': 'sum',
                        'Minutos_Totales': 'sum'
                    }).reset_index()
                    
                    fig_donut = px.pie(
                        df_cat,
                        values='Minutos_Totales',
                        names='Categoria',
                        hole=0.4,
                        color='Categoria',
                        color_discrete_map={
                            'Mechanical': COLOR_MAP['Breakdown - Mechanical'],
                            'Electrical': COLOR_MAP['Breakdown - Electrical'],
                            'Instrumentation': COLOR_MAP['Breakdown - Instrumentation'],
                            'Process': COLOR_MAP['Breakdown - Process'],
                            'Otros': COLOR_MAP['Otros']
                        }
                    )
                    
                    fig_donut.update_traces(textposition='outside', textinfo='label+percent')
                    fig_donut.update_layout(height=400)
                    
                    st.plotly_chart(fig_donut, use_container_width=True)
                
                with col_an2:
                    st.markdown("#### 📊 Top Causas (Pareto)")
                    
                    df_top = df_analysis.nlargest(10, 'Minutos_Totales')
                    
                    fig_bar = px.bar(
                        df_top,
                        x='Frecuencia',
                        y='Causa',
                        orientation='h',
                        color='Minutos_Totales',
                        color_continuous_scale='Reds',
                        hover_data=['Duracion_Promedio']
                    )
                    
                    fig_bar.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        height=400
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Tabla detallada
                st.markdown("---")
                st.markdown("#### 📋 Detalle Completo")
                
                st.dataframe(
                    df_analysis.style.format({
                        'Frecuencia': '{:,.0f}',
                        'Minutos_Totales': '{:,.0f}',
                        'Duracion_Promedio': '{:.1f}'
                    }).background_gradient(
                        subset=['Minutos_Totales'],
                        cmap='Reds'
                    ),
                    use_container_width=True,
                    height=400
                )
            
            else:
                st.info("✅ No hay eventos registrados en este período")
        
        except Exception as e:
            st.error(f"❌ Error en análisis: {e}")
    
    # ========================================================================
    # TAB 3: COMPARADOR
    # ========================================================================
    with tab3:
        st.markdown("### 🆚 Comparación entre Líneas")
        
        if multi_lines:
            try:
                # Convertir nombres a IDs
                ids_compare = [int(lines_map[name]) for name in multi_lines]
                
                df_comp = get_lines_comparison(conns['DB_PROD'], ids_compare, days_range)
                
                if not df_comp.empty:
                    col_comp1, col_comp2 = st.columns([1, 2])
                    
                    with col_comp1:
                        st.markdown("#### 📊 Tabla Comparativa")
                        st.dataframe(
                            df_comp.style.format({
                                'Num_Breakdowns': '{:,.0f}',
                                'Minutos_Totales': '{:,.0f}',
                                'Duracion_Promedio': '{:.1f}'
                            }).background_gradient(
                                subset=['Minutos_Totales'],
                                cmap='Reds'
                            ),
                            use_container_width=True
                        )
                    
                    with col_comp2:
                        st.markdown("#### 📊 Gráfico Comparativo")
                        
                        fig_comp = go.Figure()
                        
                        # Barras de minutos totales
                        fig_comp.add_trace(go.Bar(
                            x=df_comp['Linea'],
                            y=df_comp['Minutos_Totales'],
                            name='Minutos Perdidos',
                            marker_color='#00BCD4',
                            text=df_comp['Minutos_Totales'],
                            texttemplate='%{text:.0f}',
                            textposition='outside'
                        ))
                        
                        # Línea de número de breakdowns
                        fig_comp.add_trace(go.Scatter(
                            x=df_comp['Linea'],
                            y=df_comp['Num_Breakdowns'],
                            name='Número de Eventos',
                            mode='lines+markers',
                            marker=dict(size=12, color='#9C27B0'),
                            line=dict(width=3),
                            yaxis='y2'
                        ))
                        
                        fig_comp.update_layout(
                            yaxis=dict(title='Minutos Perdidos'),
                            yaxis2=dict(
                                title='Número de Eventos',
                                overlaying='y',
                                side='right'
                            ),
                            height=500,
                            hovermode='x unified',
                            plot_bgcolor='rgba(248,249,250,1)'
                        )
                        
                        st.plotly_chart(fig_comp, use_container_width=True)
                
                else:
                    st.warning("No hay datos para las líneas seleccionadas en este período")
            
            except Exception as e:
                st.error(f"❌ Error en comparador: {e}")
        
        else:
            st.info("👈 Selecciona líneas en la barra lateral para comparar")

if __name__ == "__main__":
    main()