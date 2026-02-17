"""
Test de Inferencia - Sistema de Mantenimiento Predictivo
Prueba el pipeline completo desde extracción hasta predicción
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yaml
import logging

# Agregar el directorio raíz al path para imports
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Importar módulos propios
from src.db_connector import DatabaseConnector
from src.feature_engineering import FeatureEngineer
from src.predictor import Predictor

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_inference.log', mode='w')
    ]
)

log = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Cargar config.yaml

log.info("=" * 80)
log.info(" TEST DE INFERENCIA - SISTEMA DE MANTENIMIENTO PREDICTIVO")
log.info("=" * 80)
log.info(f"Timestamp: {datetime.now()}")


log.info(f"Directorio base: {BASE_DIR}")

from src.config_loader import load_config

try:
    config = load_config(str(BASE_DIR / 'config.yaml'))
    log.info(f" Config cargado desde: {BASE_DIR / 'config.yaml'}")
except Exception as e:
    log.error(f" Error cargando config: {e}")
    sys.exit(1)

# ============================================================================
# PASO 1: TEST DE CONEXIÓN A BASE DE DATOS
# ============================================================================

log.info("\n" + "=" * 80)
log.info(" PASO 1: TEST DE CONEXIÓN A BASE DE DATOS")
log.info("=" * 80)

try:
    db = DatabaseConnector(config)
    
    # Test básico
    test_ok = db.test_connection()
    
    if not test_ok:
        log.error(" Test de conexión falló")
        sys.exit(1)
    
    log.info(" Conexión a BD establecida correctamente")
    
except Exception as e:
    log.error(f" Error en conexión: {e}")
    import traceback
    log.error(traceback.format_exc())
    sys.exit(1)

# ============================================================================
# PASO 2: EXTRACCIÓN DE DATOS (7 DÍAS)
# ============================================================================

log.info("\n" + "=" * 80)
log.info(" PASO 2: EXTRACCIÓN DE DATOS (7 DÍAS)")
log.info("=" * 80)

try:
    # Calcular rango temporal
    # Descomentar para TEST (fecha fija)

    fecha_fin = datetime.strptime('2025-07-25', '%Y-%m-%d')

    #fecha_fin = datetime.now() #descomentar para pruebas reales PRODUCCION
    fecha_inicio = fecha_fin - timedelta(days=7)
    
    log.info(f"Extrayendo datos desde {fecha_inicio} hasta {fecha_fin}")
    
    # 1. Extraer bui_perdida
    log.info("\n Extrayendo bui_perdida...")
    query_perdida = f"""
    SELECT 
        id_perdida,
        id_linea,
        id_fabrica,
        id_maquina_dfos,
        de_perdida_1,
        de_perdida_2,
        de_perdida_3,
        fe_inicio,
        fe_fin,
        TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin) as duracion_minutos
    FROM bui_perdida
    WHERE fe_inicio >= '{fecha_inicio.strftime('%Y-%m-%d %H:%M:%S')}'
      AND fe_inicio <= '{fecha_fin.strftime('%Y-%m-%d %H:%M:%S')}'
    ORDER BY id_maquina_dfos, fe_inicio
    """
    
    bui_perdida = db.execute_query(query_perdida)
    log.info(f" bui_perdida extraída: {len(bui_perdida):,} registros")
    log.info(f"   Máquinas únicas: {bui_perdida['id_maquina_dfos'].nunique()}")
    log.info(f"   Rango temporal: {bui_perdida['fe_inicio'].min()} a {bui_perdida['fe_inicio'].max()}")
    
    # 2. Extraer bui_pm_ewo
    log.info("\n Extrayendo bui_pm_ewo...")

        
    fecha_inicio_ewo = fecha_fin - timedelta(days=60)
    
    query_ewo = f"""
    SELECT 
        id_ewo,
        id_linea,
        id_perdida,
        fl_mantenimiento,
        fe_inicio_averia,
        fe_inicio_prod,
        fl_borrador,
        fe_cerrar,                          
        fe_fin_mto,                         
        fe_fin_reparacion,                  
        fe_fin_diagnostico,                 
        fe_llegada_mto,                     
        fe_aviso_mto,                       
        TIMESTAMPDIFF(MINUTE, fe_inicio_averia, fe_inicio_prod) as duracion_reparacion_min
    FROM bui_pm_ewo
    WHERE fe_inicio_averia >= '{fecha_inicio_ewo.strftime('%Y-%m-%d %H:%M:%S')}'
      AND fe_inicio_averia <= '{fecha_fin.strftime('%Y-%m-%d %H:%M:%S')}'
    ORDER BY id_linea, fe_inicio_averia
    """
    
    bui_pm_ewo = db.execute_query(query_ewo)
    log.info(f" bui_pm_ewo extraída: {len(bui_pm_ewo):,} registros")
    log.info(f"   Rango: {fecha_inicio_ewo.strftime('%Y-%m-%d')} a {fecha_fin.strftime('%Y-%m-%d')} (60 días)")

    def calcular_estado_ewo(row):
        if pd.notna(row['fe_cerrar']):
            return 'CERRADA'
        elif pd.notna(row['fe_fin_mto']):
            return 'FINALIZADA_PENDIENTE_CIERRE'
        elif pd.notna(row['fe_fin_reparacion']):
            return 'REPARADA'
        elif pd.notna(row['fe_fin_diagnostico']):
            return 'DIAGNOSTICADA'
        elif pd.notna(row['fe_llegada_mto']):
            return 'EN_ATENCION'
        elif pd.notna(row['fe_aviso_mto']):
            return 'AVISADA'
        else:
            return 'CREADA_SIN_AVISO'

    bui_pm_ewo['estado_ewo'] = bui_pm_ewo.apply(calcular_estado_ewo, axis=1)
    log.info(f" estado_ewo calculado: {bui_pm_ewo['estado_ewo'].value_counts().to_dict()}")

    
    # 3. Extraer bui_line
    log.info("\n Extrayendo bui_line...")
    query_line = """
    SELECT 
        id_linea,
        id_subcategoria,
        id_fabrica_area
    FROM bui_line
    """
    
    bui_line = db.execute_query(query_line)
    log.info(f" bui_line extraída: {len(bui_line):,} registros")
    
    # Convertir fechas a datetime
    log.info("\n Convirtiendo tipos de datos...")
    bui_perdida['fe_inicio'] = pd.to_datetime(bui_perdida['fe_inicio'])
    bui_perdida['fe_fin'] = pd.to_datetime(bui_perdida['fe_fin'])
    bui_pm_ewo['fe_inicio_averia'] = pd.to_datetime(bui_pm_ewo['fe_inicio_averia'])
    bui_pm_ewo['fe_inicio_prod'] = pd.to_datetime(bui_pm_ewo['fe_inicio_prod'])
    
    log.info(" Datos extraídos correctamente")
    
    # Cerrar conexión a BD
    db.close()
    
except Exception as e:
    log.error(f" Error en extracción de datos: {e}")
    import traceback
    log.error(traceback.format_exc())
    sys.exit(1)

# ============================================================================
# PASO 3: FEATURE ENGINEERING
# ============================================================================

log.info("\n" + "=" * 80)
log.info(" PASO 3: FEATURE ENGINEERING")
log.info("=" * 80)

try:
    # Inicializar Feature Engineer
    fe = FeatureEngineer(config)
    
    # Transformar datos
    log.info("Iniciando transformación de datos...")
    df_features = fe.transform(
        bui_perdida=bui_perdida,
        bui_pm_ewo=bui_pm_ewo,
        bui_line=bui_line
    )
    
    log.info("\n FEATURE ENGINEERING COMPLETADO")
    log.info(f"   Shape: {df_features.shape}")
    log.info(f"   Ventanas: {len(df_features):,}")
    log.info(f"   Features: {len(df_features.columns)}")
    log.info(f"   Máquinas: {df_features['id_maquina_dfos'].nunique()}")
    
    # Mostrar primeras filas
    log.info("\n Primeras 3 ventanas:")
    log.info(f"\n{df_features[['timestamp_hora', 'id_maquina_dfos', 'run_time', 'total_events']].head(3)}")
    
except Exception as e:
    log.error(f" Error en feature engineering: {e}")
    import traceback
    log.error(traceback.format_exc())
    sys.exit(1)

# ============================================================================
# PASO 4: PREDICCIÓN
# ============================================================================

log.info("\n" + "=" * 80)
log.info(" PASO 4: PREDICCIÓN")
log.info("=" * 80)

try:
    # Inicializar Predictor
    predictor = Predictor(config)
    
    # Información del modelo
    model_info = predictor.get_model_info()
    log.info("\n Información del modelo:")
    log.info(f"   Versión: {model_info['version']}")
    log.info(f"   Features requeridas: {model_info['num_features']}")
    log.info(f"   Árboles en el modelo: {model_info['num_trees']}")
    log.info(f"   Umbrales:")
    log.info(f"      Crítico: {model_info['thresholds']['critical']}")
    log.info(f"      Moderado: {model_info['thresholds']['moderate']}")
    log.info(f"      Bajo: {model_info['thresholds']['low']}")
    
    # Generar predicciones
    log.info("\nGenerando predicciones...")
    predictions = predictor.predict(df_features)
    
    log.info("\n PREDICCIONES COMPLETADAS")
    
except Exception as e:
    log.error(f" Error en predicción: {e}")
    import traceback
    log.error(traceback.format_exc())
    sys.exit(1)

# ============================================================================
# PASO 5: ANÁLISIS DE RESULTADOS
# ============================================================================

log.info("\n" + "=" * 80)
log.info(" PASO 5: ANÁLISIS DE RESULTADOS")
log.info("=" * 80)

try:
    # Estadísticas generales
    log.info(f"\n Estadísticas de predicciones:")
    log.info(f"   Total ventanas predichas: {len(predictions):,}")
    log.info(f"   Máquinas únicas: {predictions['id_maquina_dfos'].nunique()}")
    log.info(f"   Líneas únicas: {predictions['id_linea'].nunique()}")
    
    # Distribución de scores
    log.info(f"\n Distribución de scores:")
    log.info(f"   Mínimo: {predictions['score'].min():.4f}")
    log.info(f"   Máximo: {predictions['score'].max():.4f}")
    log.info(f"   Media: {predictions['score'].mean():.4f}")
    log.info(f"   Mediana: {predictions['score'].median():.4f}")
    log.info(f"   Percentil 90: {predictions['score'].quantile(0.90):.4f}")
    log.info(f"   Percentil 95: {predictions['score'].quantile(0.95):.4f}")
    log.info(f"   Percentil 99: {predictions['score'].quantile(0.99):.4f}")
    
    # Distribución de niveles de alerta
    log.info(f"\n Distribución de niveles de alerta:")
    alertas_dist = predictions['nivel_alerta'].value_counts()
    for nivel in ['critico', 'moderado', 'bajo', 'normal']:
        count = alertas_dist.get(nivel, 0)
        porcentaje = (count / len(predictions) * 100) if len(predictions) > 0 else 0
        emoji = {'critico': 'ROJO', 'moderado': 'AMARILLO', 'bajo': 'VERDE', 'normal': 'GRIS'}.get(nivel, '?')
        log.info(f"   {emoji} {nivel.upper()}: {count:,} ({porcentaje:.2f}%)")
    
    # Top 10 máquinas con mayor score
    log.info(f"\n Top 10 máquinas con mayor score promedio:")
    top_maquinas = predictions.groupby('id_maquina_dfos')['score'].agg(['mean', 'max', 'count']).sort_values('mean', ascending=False).head(10)
    log.info(f"\n{top_maquinas}")
    
    # Alertas críticas
    alertas_criticas = predictions[predictions['nivel_alerta'] == 'critico']
    if len(alertas_criticas) > 0:
        log.info(f"\n ALERTAS CRÍTICAS DETECTADAS: {len(alertas_criticas)}")
        log.info("\nTop 10 alertas críticas más urgentes:")
        top_criticas = alertas_criticas.nlargest(10, 'score')[['timestamp_ventana', 'id_maquina_dfos', 'score', 'nivel_alerta']]
        log.info(f"\n{top_criticas}")
    else:
        log.info(f"\n No se detectaron alertas críticas")
    
    # Alertas moderadas
    alertas_moderadas = predictions[predictions['nivel_alerta'] == 'moderado']
    if len(alertas_moderadas) > 0:
        log.info(f"\n ALERTAS MODERADAS: {len(alertas_moderadas)}")
        log.info("Top 5 alertas moderadas:")
        top_moderadas = alertas_moderadas.nlargest(5, 'score')[['timestamp_ventana', 'id_maquina_dfos', 'score', 'nivel_alerta']]
        log.info(f"\n{top_moderadas}")
    
    # Guardar resultados
    output_file = BASE_DIR / f'test_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    predictions.to_csv(output_file, index=False)
    log.info(f"\n Predicciones guardadas en: {output_file}")
    
    # Resumen por timestamp
    log.info(f"\n Predicciones por hora:")
    predictions_por_hora = predictions.groupby('timestamp_ventana').agg({
        'score': ['count', 'mean', 'max'],
        'nivel_alerta': lambda x: (x == 'critico').sum()
    })
    predictions_por_hora.columns = ['total_ventanas', 'score_medio', 'score_max', 'alertas_criticas']
    log.info(f"\n{predictions_por_hora.tail(24)}")  # Últimas 24 horas
    
except Exception as e:
    log.error(f" Error en análisis de resultados: {e}")
    import traceback
    log.error(traceback.format_exc())

# ============================================================================
# FINALIZACIÓN
# ============================================================================

log.info("\n" + "=" * 80)
log.info(" TEST COMPLETADO EXITOSAMENTE")
log.info("=" * 80)
log.info(f"Timestamp final: {datetime.now()}")
log.info(f"Log guardado en: test_inference.log")
log.info(f"Predicciones guardadas en: {output_file}")

print("\n" + "=" * 80)
print(" TEST COMPLETADO - Revisa el archivo test_inference.log para más detalles")
print("=" * 80)

