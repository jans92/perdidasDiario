"""
update_targets.py
=================
Actualiza fl_target_real y fl_acierto para predicciones donde ya pasaron 24h.
Se ejecuta periodicamente (ej: 1 vez al dia a las 23:00).

ARQUITECTURA:
- LEE predicciones de: DEV (bui_predicciones_hora)
- LEE realidad de: PROD (bui_perdida, bui_pm_ewo)
- ESCRIBE targets en: DEV (bui_predicciones_hora)

Uso:
    python update_targets.py
"""

import sys
import os
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging

log = logging.getLogger(__name__)

UMBRAL_DURACION_CON_EWO = 10
UMBRAL_DURACION_LARGA = 60
UMBRAL_DURACION_TIPO_CRITICO = 20

TIPOS_CRITICOS = [
    'Breakdown - Mechanical',
    'Breakdown - Electrical',
    'Breakdown - Instrumentation & Control'
]

BATCH_LIMIT = 50000


def load_config():
    """Carga configuracion desde config.yaml"""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_db_connection(env_prefix):
    """
    Crea una conexion a base de datos basada en el prefijo de las variables de entorno.
    
    Args:
        env_prefix: Prefijo de variables de entorno ('DB_PROD' o 'DB_DEV')
    
    Returns:
        SQLAlchemy engine conectado a la base de datos
    """
    try:
        host = os.getenv(f'{env_prefix}_HOST')
        user = os.getenv(f'{env_prefix}_USER')
        password = os.getenv(f'{env_prefix}_PASSWORD')
        database = os.getenv(f'{env_prefix}_NAME')
        port = os.getenv(f'{env_prefix}_PORT', '3306')

        if not all([host, user, password, database]):
            raise ValueError(f"Faltan variables de entorno para {env_prefix}")

        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

        engine = create_engine(
            connection_string,
            connect_args={'ssl': {'fake_flag_to_enable_tls': True}},
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        log.info(f"Conexion exitosa a {env_prefix}: {database}@{host}")
        return engine

    except Exception as e:
        log.error(f"Error creando conexion para {env_prefix}: {e}")
        raise


def obtener_predicciones_pendientes(engine_dev, limit=BATCH_LIMIT):
    """
    Obtiene predicciones que necesitan actualizacion de target_real.
    
    Args:
        engine_dev: Motor de conexion a base de datos DEV
        limit: Numero maximo de predicciones a procesar
    
    Returns:
        DataFrame con predicciones pendientes
    """
    log.info("Obteniendo predicciones pendientes de actualizar...")
    
    query = text("""
        SELECT 
            id_prediccion,
            id_maquina_dfos,
            id_linea,
            fe_ventana,
            nm_score,
            fl_pred_modelo
        FROM bui_predicciones_hora
        WHERE fl_target_real IS NULL
          AND fe_ventana < NOW() - INTERVAL 2 HOUR
        ORDER BY fe_ventana ASC
        LIMIT :limit
    """)
    
    df = pd.read_sql(query, engine_dev, params={'limit': limit})
    
    log.info(f"Predicciones pendientes: {len(df):,}")
    if len(df) > 0:
        log.info(f"Rango de fechas: {df['fe_ventana'].min()} a {df['fe_ventana'].max()}")
        log.info(f"Maquinas unicas: {df['id_maquina_dfos'].nunique()}")
    
    return df


def cargar_perdidas_en_cache(engine_prod, fecha_inicio, fecha_fin):
    """
    Carga TODAS las perdidas del periodo en memoria para busquedas rapidas.
    """
    log.info("Cargando perdidas en cache desde PROD...")
    
    query = text("""
        SELECT 
            id_maquina_dfos,
            fe_inicio,
            fe_fin,
            de_perdida_2,
            de_perdida_3
        FROM bui_perdida
        WHERE de_perdida_2 = 'Breakdown & Equipment Failure Time'
          AND fe_inicio BETWEEN :fecha_inicio AND :fecha_fin
        ORDER BY id_maquina_dfos, fe_inicio
    """)
    
    df = pd.read_sql(query, engine_prod, params={
        'fecha_inicio': fecha_inicio - timedelta(days=1),
        'fecha_fin': fecha_fin + timedelta(days=1)
    })
    
    log.info(f"Perdidas en cache: {len(df):,}")
    
    return df


def obtener_ewos_validas(engine_prod, fecha_inicio, fecha_fin):
    """
    Obtiene los id_perdida de EWOs correctivas validas en un rango de fechas.
    
    Args:
        engine_prod: Motor de conexion a base de datos PROD
        fecha_inicio: Fecha inicial del periodo
        fecha_fin: Fecha final del periodo
    
    Returns:
        Set con los IDs de perdidas que tienen EWO valida
    """
    log.info("Obteniendo EWOs validas de PROD...")
    
    query = text("""
        SELECT DISTINCT id_perdida
        FROM bui_pm_ewo
        WHERE fl_borrador = 0
          AND id_perdida IS NOT NULL
          AND fe_inicio_averia BETWEEN :fecha_inicio AND :fecha_fin
          AND (
              fe_cerrar IS NOT NULL 
              OR fe_fin_mto IS NOT NULL
          )
    """)
    
    df = pd.read_sql(query, engine_prod, params={
        'fecha_inicio': fecha_inicio,
        'fecha_fin': fecha_fin
    })
    
    ewos_set = set(df['id_perdida'].tolist())
    log.info(f"EWOs validas encontradas: {len(ewos_set):,}")
    
    return ewos_set


def verificar_falla_grave_en_cache(df_perdidas, id_maquina, fe_ventana):
    """
    Verifica si hubo falla grave usando dataframe en cache (MUY RAPIDO).
    """
    fe_inicio_busqueda = fe_ventana - timedelta(minutes=15)
    fe_fin_busqueda = fe_ventana + timedelta(hours=1, minutes=15)
    
    perdidas_maquina = df_perdidas[df_perdidas['id_maquina_dfos'] == id_maquina]
    
    if len(perdidas_maquina) == 0:
        return 0
    
    fallas = perdidas_maquina[
        ((perdidas_maquina['fe_inicio'] >= fe_inicio_busqueda) & 
         (perdidas_maquina['fe_inicio'] <= fe_fin_busqueda)) |
        ((perdidas_maquina['fe_fin'] >= fe_inicio_busqueda) & 
         (perdidas_maquina['fe_fin'] <= fe_fin_busqueda)) |
        ((perdidas_maquina['fe_inicio'] <= fe_inicio_busqueda) & 
         (perdidas_maquina['fe_fin'] >= fe_fin_busqueda))
    ]
    
    if len(fallas) == 0:
        return 0
    
    return 1


def actualizar_prediccion(engine_dev, id_prediccion, target_real, pred_modelo):
    """
    Actualiza una prediccion con el target real y el acierto.
    
    Args:
        engine_dev: Motor de conexion a base de datos DEV
        id_prediccion: ID unico de la prediccion
        target_real: Valor real del target (0 o 1)
        pred_modelo: Prediccion del modelo (0 o 1)
    """
    acierto = 1 if pred_modelo == target_real else 0
    
    query = text("""
        UPDATE bui_predicciones_hora
        SET fl_target_real = :target_real,
            fl_acierto = :acierto
        WHERE id_prediccion = :id_prediccion
    """)
    
    with engine_dev.begin() as conn:
        conn.execute(query, {
            'target_real': target_real,
            'acierto': acierto,
            'id_prediccion': id_prediccion
        })


def update_targets(config):
    """
    Funcion principal que actualiza los targets reales.
    
    FLUJO:
    1. Conecta a DEV (predicciones) y PROD (realidad)
    2. Obtiene predicciones pendientes de DEV
    3. Carga TODAS las perdidas en cache (RAM)
    4. Para cada prediccion:
        a. Busca en cache (MUY RAPIDO)
        b. Calcula target_real segun criterios
        c. Actualiza prediccion en DEV
    5. Calcula y muestra metricas finales
    
    Args:
        config: Diccionario con configuracion del sistema
    
    Returns:
        Numero de predicciones actualizadas
    """
    log.info("=" * 80)
    log.info("ACTUALIZACION DE TARGETS REALES (CROSS-SCHEMA)")
    log.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 80)
    
    engine_dev = None
    engine_prod = None
    
    try:
        log.info("\nESTABLECIENDO CONEXIONES...")
        log.info("DEV (Escritura): Predicciones del modelo")
        engine_dev = get_db_connection('DB_DEV')
        
        log.info("PROD (Lectura): Datos reales de produccion")
        engine_prod = get_db_connection('DB_PROD')
        
        log.info("Ambas conexiones establecidas correctamente\n")
        
        predicciones = obtener_predicciones_pendientes(engine_dev)
        
        if len(predicciones) == 0:
            log.info("\nNo hay predicciones pendientes de actualizar")
            return 0
        
        fecha_min = predicciones['fe_ventana'].min()
        fecha_max = predicciones['fe_ventana'].max() + timedelta(hours=24)
        
        log.info(f"\nPeriodo a procesar:")
        log.info(f"Inicio: {fecha_min}")
        log.info(f"Fin: {fecha_max}\n")
        
        df_perdidas = cargar_perdidas_en_cache(engine_prod, fecha_min, fecha_max)
        ewos_validas = obtener_ewos_validas(engine_prod, fecha_min, fecha_max)
        
        log.info(f"\nVerificando fallas para {len(predicciones):,} predicciones...")
        log.info("[CACHE en RAM] → [ESCRIBE targets en DEV]\n")
        
        total_actualizadas = 0
        total_con_falla = 0
        total_sin_falla = 0
        total_aciertos = 0
        
        for id_maquina, grupo in tqdm(predicciones.groupby('id_maquina_dfos'),
                                      desc="Procesando maquinas",
                                      unit="maquina"):
            
            for _, row in grupo.iterrows():
                id_prediccion = row['id_prediccion']
                fe_ventana = row['fe_ventana']
                pred_modelo = row['fl_pred_modelo']
                
                target_real = verificar_falla_grave_en_cache(
                    df_perdidas, id_maquina, fe_ventana
                )
                
                actualizar_prediccion(engine_dev, id_prediccion, target_real, pred_modelo)
                
                total_actualizadas += 1
                if target_real == 1:
                    total_con_falla += 1
                else:
                    total_sin_falla += 1
                
                if pred_modelo == target_real:
                    total_aciertos += 1
        
        log.info("\n" + "=" * 80)
        log.info("RESUMEN DE ACTUALIZACION")
        log.info("=" * 80)
        
        log.info(f"\nPredicciones actualizadas: {total_actualizadas:,}")
        log.info(f"Con falla real (target=1): {total_con_falla:,} ({total_con_falla/total_actualizadas*100:.2f}%)")
        log.info(f"Sin falla real (target=0): {total_sin_falla:,} ({total_sin_falla/total_actualizadas*100:.2f}%)")
        log.info(f"Aciertos: {total_aciertos:,} ({total_aciertos/total_actualizadas*100:.2f}%)")
        
        if total_con_falla > 0:
            log.info("\nCalculando metricas de rendimiento...")
            
            query_metricas = text("""
                SELECT 
                    fl_pred_modelo,
                    fl_target_real,
                    COUNT(*) as cantidad
                FROM bui_predicciones_hora
                WHERE fl_target_real IS NOT NULL
                  AND fe_ventana >= :fecha_min
                  AND fe_ventana <= :fecha_max
                GROUP BY fl_pred_modelo, fl_target_real
            """)
            
            metricas_df = pd.read_sql(query_metricas, engine_dev, params={
                'fecha_min': fecha_min,
                'fecha_max': fecha_max - timedelta(hours=24)
            })
            
            tp = metricas_df[(metricas_df['fl_pred_modelo']==1) & (metricas_df['fl_target_real']==1)]['cantidad'].sum()
            fp = metricas_df[(metricas_df['fl_pred_modelo']==1) & (metricas_df['fl_target_real']==0)]['cantidad'].sum()
            fn = metricas_df[(metricas_df['fl_pred_modelo']==0) & (metricas_df['fl_target_real']==1)]['cantidad'].sum()
            tn = metricas_df[(metricas_df['fl_pred_modelo']==0) & (metricas_df['fl_target_real']==0)]['cantidad'].sum()
            
            total = tp + fp + fn + tn
            
            log.info(f"\nMatriz de Confusion (periodo actualizado):")
            log.info(f"+-------------+-------------+-------------+")
            log.info(f"|             |   Real = 1  |   Real = 0  |")
            log.info(f"+-------------+-------------+-------------+")
            log.info(f"|   Pred = 1  |  TP: {tp:>5}  |  FP: {fp:>5}  |")
            log.info(f"|   Pred = 0  |  FN: {fn:>5}  |  TN: {tn:>5}  |")
            log.info(f"+-------------+-------------+-------------+")
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
                log.info(f"\nPrecision: {precision:.2%}")
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
                log.info(f"Recall: {recall:.2%}")
            
            if tp + fp > 0 and tp + fn > 0:
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                log.info(f"F1-Score: {f1:.2%}")
            
            accuracy = (tp + tn) / total if total > 0 else 0
            log.info(f"Accuracy: {accuracy:.2%}")
        
        log.info("\n" + "=" * 80)
        log.info("ACTUALIZACION COMPLETADA EXITOSAMENTE")
        log.info("=" * 80)
        
        return total_actualizadas
    
    except Exception as e:
        log.error(f"Error durante actualizacion: {e}", exc_info=True)
        raise
    
    finally:
        if engine_dev:
            engine_dev.dispose()
            log.info("Conexion DEV cerrada")
        
        if engine_prod:
            engine_prod.dispose()
            log.info("Conexion PROD cerrada")


def main():
    """Funcion principal de entrada"""
    config = load_config()
    logger = setup_logging(config)
    
    try:
        registros = update_targets(config)
        log.info(f"\nProceso finalizado. {registros:,} predicciones actualizadas")
        return 0
    
    except Exception as e:
        log.error(f"\nERROR CRITICO: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
