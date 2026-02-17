"""
dashboard.py - Dashboard Predictivo con Colores Unilever
=========================================================
Colores corporativos Unilever:
- Turquesa/Cian: #00BCD4, #00ACC1
- Morado/Violeta: #9C27B0, #7B1FA2
- Gris: #616161, #424242
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import time

load_dotenv()

# ============================================================================
# CONFIGURACIÓN DE COLORES UNILEVER
# ============================================================================
UNILEVER_COLORS = {
    'primary_cyan': '#00BCD4',      # Turquesa brillante
    'secondary_cyan': '#00ACC1',    # Turquesa oscuro
    'primary_purple': '#9C27B0',    # Morado corporativo
    'secondary_purple': '#7B1FA2',  # Morado oscuro
    'dark_gray': '#424242',         # Gris oscuro (para breakdown)
    'medium_gray': '#616161',       # Gris medio
    'light_gray': '#9E9E9E',        # Gris claro
    'red': '#D32F2F',               # Rojo para alertas críticas
    'orange': '#FF9800',            # Naranja para alertas moderadas
    'green': '#4CAF50',             # Verde para OK
    'white': '#FFFFFF'
}

st.set_page_config(
    page_title="Dashboard Predictivo - Mantenimiento",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado con colores Unilever
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #9C27B0 0%, #00BCD4 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    /* Métricas personalizadas */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #00BCD4;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f5f5f5;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #424242;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00BCD4 0%, #9C27B0 100%);
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Botones */
    .stButton>button {
        background: linear-gradient(135deg, #00BCD4 0%, #00ACC1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #00ACC1 0%, #00BCD4 100%);
        box-shadow: 0 4px 8px rgba(0, 188, 212, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE CONEXIÓN (sin cambios)
# ============================================================================
@st.cache_resource(ttl=3600)
def get_db_connection(env_prefix: str):
    """Crea conexión a BD con SSL y pool de conexiones."""
    try:
        host = os.getenv(f'{env_prefix}_HOST')
        user = os.getenv(f'{env_prefix}_USER')
        password = os.getenv(f'{env_prefix}_PASSWORD')
        database = os.getenv(f'{env_prefix}_NAME')
        port = os.getenv(f'{env_prefix}_PORT', '3306')

        if not all([host, user, password, database]):
            st.error(f"❌ Faltan variables de entorno para {env_prefix}")
            st.stop()

        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(
            connection_string,
            connect_args={'ssl': {'ssl_mode': 'PREFERRED'}},
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
            echo=False
        )

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine

    except Exception as e:
        st.error(f"❌ Error conectando a {env_prefix}: {e}")
        st.stop()

# ============================================================================
# FUNCIONES DE DATOS (sin cambios en lógica)
# ============================================================================
@st.cache_data(ttl=300)
def get_factories(_engine_dev):
    """Obtiene lista de fábricas disponibles"""
    query = text("""
        SELECT DISTINCT
            p.id_linea,
            l.id_fabrica,
            f.de_factory as nombre_fabrica
        FROM bui_predicciones_hora p
        INNER JOIN bui_line l ON p.id_linea = l.id_linea
        INNER JOIN bui_factory f ON l.id_fabrica = f.id_factory
        ORDER BY f.de_factory
    """)
    df = pd.read_sql(query, _engine_dev)
    return df.drop_duplicates(subset=['id_fabrica'])

@st.cache_data(ttl=300)
def get_metrics_summary(_engine_dev, id_fabrica=None):
    """Obtiene métricas principales del sistema"""
    params = {}
    filtro_fabrica = ""
    
    if id_fabrica is not None:
        filtro_fabrica = """
        AND EXISTS (
            SELECT 1 
            FROM bui_line l 
            WHERE l.id_linea = p.id_linea 
              AND l.id_fabrica = :id_fabrica
        )
        """
        params["id_fabrica"] = id_fabrica

    query = text(f"""
        SELECT
            COUNT(*) as total_predicciones,
            SUM(CASE WHEN fl_pred_modelo = 1 THEN 1 ELSE 0 END) as alertas_generadas,
            COUNT(DISTINCT id_maquina_dfos) as maquinas_monitoreadas,
            SUM(CASE WHEN fl_target_real IS NOT NULL THEN 1 ELSE 0 END) as targets_validados,
            SUM(CASE WHEN fl_acierto = 1 THEN 1 ELSE 0 END) as aciertos
        FROM bui_predicciones_hora p
        WHERE 1=1
        {filtro_fabrica}
    """)
    
    df = pd.read_sql(query, _engine_dev, params=params)
    result = df.iloc[0].to_dict()
    
    if result['targets_validados'] > 0:
        result['accuracy'] = (result['aciertos'] / result['targets_validados']) * 100
    else:
        result['accuracy'] = None
    
    return result

@st.cache_data(ttl=300)
def get_daily_trend(_engine_dev, days=30, id_fabrica=None):
    """Obtiene tendencia diaria de predicciones y alertas"""
    params = {"days": days}
    filtro_fabrica = ""
    
    if id_fabrica is not None:
        filtro_fabrica = """
        AND EXISTS (
            SELECT 1 
            FROM bui_line l 
            WHERE l.id_linea = p.id_linea 
              AND l.id_fabrica = :id_fabrica
        )
        """
        params["id_fabrica"] = id_fabrica

    query = text(f"""
        SELECT
            DATE(fe_ventana) as fecha,
            COUNT(*) as predicciones,
            SUM(CASE WHEN fl_pred_modelo = 1 THEN 1 ELSE 0 END) as alertas,
            SUM(CASE WHEN fl_target_real = 1 THEN 1 ELSE 0 END) as breakdowns_reales
        FROM bui_predicciones_hora p
        WHERE fe_ventana >= DATE_SUB(NOW(), INTERVAL :days DAY)
        {filtro_fabrica}
        GROUP BY DATE(fe_ventana)
        ORDER BY fecha
    """)
    
    return pd.read_sql(query, _engine_dev, params=params)

@st.cache_data(ttl=300)
def get_confusion_matrix(_engine_dev, id_fabrica=None):
    """Obtiene matriz de confusión"""
    params = {}
    filtro_fabrica = ""
    
    if id_fabrica is not None:
        filtro_fabrica = """
        AND EXISTS (
            SELECT 1 
            FROM bui_line l 
            WHERE l.id_linea = p.id_linea 
              AND l.id_fabrica = :id_fabrica
        )
        """
        params["id_fabrica"] = id_fabrica

    query = text(f"""
        SELECT
            fl_pred_modelo,
            fl_target_real,
            COUNT(*) as cantidad
        FROM bui_predicciones_hora p
        WHERE fl_target_real IS NOT NULL
        {filtro_fabrica}
        GROUP BY fl_pred_modelo, fl_target_real
    """)
    
    df = pd.read_sql(query, _engine_dev, params=params)
    
    # Crear matriz 2x2
    matriz = pd.DataFrame({
        'Pred=0': [0, 0],
        'Pred=1': [0, 0]
    }, index=['Real=0', 'Real=1'])
    
    for _, row in df.iterrows():
        pred = int(row['fl_pred_modelo'])
        real = int(row['fl_target_real'])
        matriz.iloc[real, pred] = int(row['cantidad'])
    
    return matriz

@st.cache_data(ttl=300)
def get_top_machines(_engine_dev, top_n=10, id_fabrica=None):
    """Obtiene top máquinas con más alertas"""
    params = {"top_n": top_n}
    filtro_fabrica = ""
    
    if id_fabrica is not None:
        filtro_fabrica = """
        AND EXISTS (
            SELECT 1 
            FROM bui_line l 
            WHERE l.id_linea = p.id_linea 
              AND l.id_fabrica = :id_fabrica
        )
        """
        params["id_fabrica"] = id_fabrica

    query = text(f"""
        SELECT
            p.id_maquina_dfos,
            COUNT(*) as total_pred,
            SUM(CASE WHEN fl_pred_modelo = 1 THEN 1 ELSE 0 END) as alertas,
            AVG(nm_score) as score_promedio,
            SUM(CASE WHEN fl_target_real = 1 THEN 1 ELSE 0 END) as breakdowns_reales
        FROM bui_predicciones_hora p
        WHERE fe_ventana >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        {filtro_fabrica}
        GROUP BY p.id_maquina_dfos
        HAVING alertas > 0
        ORDER BY alertas DESC
        LIMIT :top_n
    """)
    
    return pd.read_sql(query, _engine_dev, params=params)

@st.cache_data(ttl=300)
def get_alert_distribution(_engine_dev, id_fabrica=None):
    """Obtiene distribución de alertas por nivel"""
    params = {}
    filtro_fabrica = ""
    
    if id_fabrica is not None:
        filtro_fabrica = """
        AND EXISTS (
            SELECT 1 
            FROM bui_line l 
            WHERE l.id_linea = p.id_linea 
              AND l.id_fabrica = :id_fabrica
        )
        """
        params["id_fabrica"] = id_fabrica

    query = text(f"""
        SELECT
            de_nivel_riesgo,
            COUNT(*) as cantidad
        FROM bui_predicciones_hora p
        WHERE fe_ventana >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        {filtro_fabrica}
        GROUP BY de_nivel_riesgo
    """)
    
    return pd.read_sql(query, _engine_dev, params=params)

@st.cache_data(ttl=300)
def get_recent_breakdowns(_engine_prod, days=7, id_fabrica=None):
    """Obtiene breakdowns recientes de producción"""
    params = {"days": days}
    filtro_fabrica = ""
    
    if id_fabrica is not None:
        filtro_fabrica = "AND l.id_fabrica = :id_fabrica"
        params["id_fabrica"] = id_fabrica

    query = text(f"""
        SELECT
            p.id_maquina_dfos,
            p.id_linea,
            p.de_perdida_3 as tipo_falla,
            p.fe_inicio,
            p.fe_fin,
            TIMESTAMPDIFF(MINUTE, p.fe_inicio, p.fe_fin) as duracion_min
        FROM bui_perdida p
        INNER JOIN bui_line l ON p.id_linea = l.id_linea
        WHERE p.de_perdida_2 = 'Breakdown & Equipment Failure Time'
          AND p.fe_inicio >= DATE_SUB(NOW(), INTERVAL :days DAY)
          {filtro_fabrica}
        ORDER BY p.fe_inicio DESC
        LIMIT 100
    """)
    
    return pd.read_sql(query, _engine_prod, params=params)

# ============================================================================
# MAIN APP CON COLORES UNILEVER
# ============================================================================
def main():
    # Header con gradiente Unilever
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;">🔧 Dashboard Predictivo - Mantenimiento</h1>
        <p style="margin:5px 0 0 0; opacity:0.9;">Sistema de Monitoreo en Tiempo Real</p>
    </div>
    """, unsafe_allow_html=True)

    # Conexiones
    engine_dev = get_db_connection('DB_DEV')
    engine_prod = get_db_connection('DB_PROD')

    # ========================================================================
    # SIDEBAR CON FILTROS
    # ========================================================================
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/9C27B0/FFFFFF?text=Unilever", use_container_width=True)
        
        st.markdown("### ⚙️ Configuración")
        
        # Filtro de fábrica
        try:
            factories_df = get_factories(engine_dev)
            
            factory_options = ["Todas las fábricas"] + factories_df['nombre_fabrica'].tolist()
            selected_factory = st.selectbox(
                "🏭 Fábrica",
                options=factory_options,
                index=0
            )
            
            if selected_factory == "Todas las fábricas":
                id_fabrica = None
            else:
                id_fabrica = int(factories_df[factories_df['nombre_fabrica'] == selected_factory]['id_fabrica'].iloc[0])
        
        except Exception as e:
            st.error(f"Error cargando fábricas: {e}")
            id_fabrica = None
        
        st.markdown("---")
        
        # Parámetros
        days_trend = st.slider("📅 Días de tendencia", 7, 90, 30)
        top_n_machines = st.slider("🏭 Top N Máquinas", 5, 20, 10)
        auto_refresh = st.checkbox("🔄 Auto-refresh (5 min)", value=False)
        
        st.markdown("---")
        st.caption(f"🕐 Última actualización: {datetime.now().strftime('%H:%M:%S')}")

    st.markdown("---")

    # ========================================================================
    # MÉTRICAS PRINCIPALES CON COLORES UNILEVER
    # ========================================================================
    try:
        metrics = get_metrics_summary(engine_dev, id_fabrica=id_fabrica)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Total Predicciones",
                value=f"{metrics['total_predicciones']:,}",
            )
        
        with col2:
            pct_alertas = (
                metrics['alertas_generadas'] / metrics['total_predicciones'] * 100
                if metrics['total_predicciones'] > 0 else 0
            )
            st.metric(
                label="🚨 Alertas Generadas",
                value=f"{metrics['alertas_generadas']:,}",
                delta=f"{pct_alertas:.1f}%"
            )
        
        with col3:
            accuracy_val = metrics.get('accuracy') or 0
            st.metric(
                label="🎯 Accuracy",
                value=f"{accuracy_val:.2f}%",
                delta=f"{metrics['targets_validados']:,} validados"
            )
        
        with col4:
            st.metric(
                label="🏭 Máquinas Monitoreadas",
                value=f"{metrics['maquinas_monitoreadas']}",
            )
        
        st.markdown("---")
    
    except Exception as e:
        st.error(f"❌ Error cargando métricas: {e}")
        st.stop()

    # ========================================================================
    # TABS CON GRÁFICOS EN COLORES UNILEVER
    # ========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Tendencias",
        "🎯 Matriz Confusión",
        "🏭 Top Máquinas",
        "⚠️ Breakdowns Recientes"
    ])

    # TAB 1: TENDENCIAS CON COLORES UNILEVER
    with tab1:
        st.subheader("Tendencia de Predicciones y Alertas")
        try:
            df_trend = get_daily_trend(engine_dev, days=days_trend, id_fabrica=id_fabrica)
            
            if not df_trend.empty:
                fig = go.Figure()
                
                # Predicciones en turquesa
                fig.add_trace(go.Scatter(
                    x=df_trend['fecha'],
                    y=df_trend['predicciones'],
                    name='Predicciones',
                    mode='lines+markers',
                    line=dict(color=UNILEVER_COLORS['primary_cyan'], width=3),
                    marker=dict(size=8)
                ))
                
                # Alertas en morado
                fig.add_trace(go.Scatter(
                    x=df_trend['fecha'],
                    y=df_trend['alertas'],
                    name='Alertas',
                    mode='lines+markers',
                    line=dict(color=UNILEVER_COLORS['primary_purple'], width=3),
                    marker=dict(size=8)
                ))
                
                # Breakdowns en gris oscuro
                fig.add_trace(go.Scatter(
                    x=df_trend['fecha'],
                    y=df_trend['breakdowns_reales'],
                    name='Breakdowns Reales',
                    mode='lines+markers',
                    line=dict(color=UNILEVER_COLORS['dark_gray'], width=3, dash='dot'),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Cantidad",
                    hovermode='x unified',
                    height=450,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        bgcolor="rgba(255,255,255,0.8)"
                    ),
                    plot_bgcolor='rgba(248,249,250,1)',
                    paper_bgcolor='white',
                    font=dict(family="Arial, sans-serif", size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info(f"📊 Promedio predicciones/día: **{df_trend['predicciones'].mean():.0f}**")
                with c2:
                    st.warning(f"🚨 Promedio alertas/día: **{df_trend['alertas'].mean():.0f}**")
                with c3:
                    st.error(f"⚠️ Promedio breakdowns/día: **{df_trend['breakdowns_reales'].mean():.1f}**")
            else:
                st.warning("No hay datos para el período seleccionado")
        
        except Exception as e:
            st.error(f"Error cargando tendencias: {e}")

    # TAB 2: MATRIZ CONFUSIÓN CON COLORES UNILEVER
    with tab2:
        st.subheader("Matriz de Confusión")
        try:
            matriz = get_confusion_matrix(engine_dev, id_fabrica=id_fabrica)
            
            if not matriz.empty and matriz.sum().sum() > 0:
                # Gradiente turquesa-morado para matriz
                fig_cm = px.imshow(
                    matriz,
                    text_auto=True,
                    labels=dict(x="Predicción", y="Real", color="Cantidad"),
                    x=['No Breakdown (0)', 'Breakdown (1)'],
                    y=['No Breakdown (0)', 'Breakdown (1)'],
                    color_continuous_scale=[
                        [0, '#E0F7FA'],      # Turquesa muy claro
                        [0.5, UNILEVER_COLORS['primary_cyan']],
                        [1, UNILEVER_COLORS['primary_purple']]
                    ]
                )
                
                fig_cm.update_layout(
                    height=400,
                    font=dict(size=14, family="Arial, sans-serif")
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Métricas derivadas
                tn = matriz.iloc[0, 0]
                fp = matriz.iloc[0, 1]
                fn = matriz.iloc[1, 0]
                tp = matriz.iloc[1, 1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    st.metric("Precisión", f"{precision:.2%}")
                
                with col2:
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    st.metric("Recall", f"{recall:.2%}")
                
                with col3:
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    st.metric("F1-Score", f"{f1:.2%}")
            else:
                st.warning("No hay datos validados (targets) para generar la matriz")
        
        except Exception as e:
            st.error(f"Error cargando matriz de confusión: {e}")

    # TAB 3: TOP MÁQUINAS CON COLORES UNILEVER
    with tab3:
        st.subheader(f"Top {top_n_machines} Máquinas con Más Alertas (últimos 30 días)")
        try:
            df_top = get_top_machines(engine_dev, top_n=top_n_machines, id_fabrica=id_fabrica)
            
            if not df_top.empty:
                # Gradiente turquesa-morado para barras
                fig_bar = px.bar(
                    df_top,
                    x='id_maquina_dfos',
                    y='alertas',
                    color='score_promedio',
                    labels={'alertas': 'Número de Alertas', 'id_maquina_dfos': 'Máquina'},
                    color_continuous_scale=[
                        [0, UNILEVER_COLORS['primary_cyan']],
                        [0.5, UNILEVER_COLORS['secondary_purple']],
                        [1, UNILEVER_COLORS['red']]
                    ],
                    hover_data=['breakdowns_reales']
                )
                
                fig_bar.update_layout(
                    height=450,
                    plot_bgcolor='rgba(248,249,250,1)',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.dataframe(
                    df_top.style.format({
                        'score_promedio': '{:.3f}',
                        'alertas': '{:,.0f}',
                        'total_pred': '{:,.0f}'
                    }),
                    use_container_width=True
                )
            else:
                st.warning("No hay datos de alertas en los últimos 30 días")
        
        except Exception as e:
            st.error(f"Error cargando top máquinas: {e}")

    # TAB 4: BREAKDOWNS RECIENTES
    with tab4:
        st.subheader("Breakdowns Recientes (últimos 7 días)")
        try:
            df_breaks = get_recent_breakdowns(engine_prod, days=7, id_fabrica=id_fabrica)
            
            if not df_breaks.empty:
                st.dataframe(
                    df_breaks.style.format({
                        'duracion_min': '{:.0f} min',
                        'fe_inicio': lambda x: x.strftime('%Y-%m-%d %H:%M'),
                        'fe_fin': lambda x: x.strftime('%Y-%m-%d %H:%M')
                    }),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total breakdowns", len(df_breaks))
                with col2:
                    avg_dur = df_breaks['duracion_min'].mean()
                    st.metric("Duración promedio", f"{avg_dur:.1f} min")
            else:
                st.info("No hay breakdowns recientes registrados")
        
        except Exception as e:
            st.error(f"Error cargando breakdowns: {e}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(300)
        st.rerun()

if __name__ == "__main__":
    main()