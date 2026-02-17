"""
update_targets.py
=================
Script que actualiza los targets reales (target_real) después de 24h.
Se ejecuta 1 vez al día (23:00).
"""

import sys
import os
import logging
import yaml
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db_connector import DatabaseConnector
from src.utils import setup_logging


def load_config():
    """Carga configuración."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def update_targets(config):
    """
    Actualiza target_real para predicciones donde ya pasaron 24h.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(" ACTUALIZACIÓN DE TARGETS REALES")
    logger.info("=" * 80)
    
    # Conectar a bases de datos
    db = DatabaseConnector(config)
    db.connect_source()
    db.connect_predictions()
    
    # ==========================================================================
    # PASO 1: Obtener predicciones pendientes de actualizar
    # ==========================================================================
    logger.info("\n Obteniendo predicciones pendientes...")
    
    cutoff_time = datetime.now() - timedelta(hours=24)
    
    query_predictions = """
    SELECT 
        id,
        timestamp_ventana,
        id_maquina_dfos,
        score,
        nivel_alerta
    FROM bui_predictions_hourly
    WHERE timestamp_ventana < %s
      AND target_real IS NULL
    LIMIT 10000
    """
    
    predictions_df = pd.read_sql(
        query_predictions,
        db.prediction_engine,
        params=[cutoff_time]
    )
    
    logger.info(f"   Predicciones a actualizar: {len(predictions_df):,}")
    
    if len(predictions_df) == 0:
        logger.info(" No hay predicciones pendientes. Finalizando.")
        return
    
    # ==========================================================================
    # PASO 2: Para cada predicción, verificar si hubo falla real
    # ==========================================================================
    logger.info("\n Verificando fallas reales...")
    
    updated_count = 0
    
    for idx, row in predictions_df.iterrows():
        pred_id = row['id']
        timestamp_inicio = row['timestamp_ventana']
        timestamp_fin = timestamp_inicio + timedelta(hours=24)
        id_maquina = row['id_maquina_dfos']
        
        # Extraer id_linea desde id_maquina_dfos (formato: "23_L01-M001")
        id_linea = id_maquina.split('_')[1]
        
        # Buscar si hubo breakdown en las siguientes 24h
        query_fallas = """
        SELECT COUNT(*) as n_fallas
        FROM bui_perdida
        WHERE id_linea = %s
          AND timestamp_hora BETWEEN %s AND %s
          AND breakdown_equipment_failure_time_count > 0
        """
        
        result = pd.read_sql(
            query_fallas,
            db.source_engine,
            params=[id_linea, timestamp_inicio, timestamp_fin]
        )
        
        n_fallas = result['n_fallas'].iloc[0]
        target_real = 1 if n_fallas > 0 else 0
        
        # Determinar si fue acierto
        score = row['score']
        pred_binary = 1 if score >= 0.31 else 0
        acierto = 1 if pred_binary == target_real else 0
        
        # Actualizar en base de datos
        update_query = """
        UPDATE bui_predictions_hourly
        SET target_real = %s,
            acierto = %s
        WHERE id = %s
        """
        
        with db.prediction_engine.connect() as conn:
            conn.execute(update_query, [target_real, acierto, pred_id])
            conn.commit()
        
        updated_count += 1
        
        if updated_count % 1000 == 0:
            logger.info(f"   Actualizadas: {updated_count:,} / {len(predictions_df):,}")
    
    logger.info(f"\n Actualización completada: {updated_count:,} predicciones")
    
    # ==========================================================================
    # PASO 3: Calcular métricas del día
    # ==========================================================================
    logger.info("\n Calculando métricas del período actualizado...")
    
    query_metrics = """
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN target_real = 1 THEN 1 ELSE 0 END) as fallas_reales,
        SUM(CASE WHEN acierto = 1 THEN 1 ELSE 0 END) as aciertos,
        AVG(acierto) * 100 as accuracy
    FROM bui_predictions_hourly
    WHERE target_real IS NOT NULL
      AND DATE(timestamp_ventana) = CURDATE() - INTERVAL 1 DAY
    """
    
    metrics = pd.read_sql(query_metrics, db.prediction_engine)
    
    logger.info(f"   Total predicciones: {metrics['total'].iloc[0]:,}")
    logger.info(f"   Fallas reales: {metrics['fallas_reales'].iloc[0]:,}")
    logger.info(f"   Aciertos: {metrics['aciertos'].iloc[0]:,}")
    logger.info(f"   Accuracy: {metrics['accuracy'].iloc[0]:.2f}%")
    
    # Cerrar conexiones
    db.close_connections()
    
    logger.info("\n" + "=" * 80)
    logger.info(" ACTUALIZACIÓN DE TARGETS COMPLETADA")
    logger.info("=" * 80)


def main():
    """Función principal."""
    config = load_config()
    logger = setup_logging(config)
    
    try:
        update_targets(config)
        return 0
    except Exception as e:
        logger.error(f" ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)