"""
save_predictions.py
====================
Guarda las predicciones generadas por main.py en la tabla bui_predicciones_hora.
Se ejecuta inmediatamente después de main.py.

Uso:
    # Opción 1: Importar como módulo
    from save_predictions import save_predictions_to_db
    save_predictions_to_db(predictions_df, config)
    
    # Opción 2: Ejecutar standalone (lee predictions del CSV temporal)
    python save_predictions.py
"""

import sys
import os
import logging
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging

log = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE UMBRALES
# ============================================================================

UMBRAL_PRED_MODELO = 0.37  # score >= 0.37 → fl_pred_modelo = 1
UMBRALES_NIVEL = {
    'critico': 0.60,
    'moderado': 0.37,
    'bajo': 0.15
}


# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def load_config():
    """Carga configuración desde config.yaml"""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_db_engine(config):
    """Crea conexión a la base de datos"""
    
    # DETECTAR ESTRUCTURA DE CONFIGURACIÓN
    # Si existe 'target' dentro de 'database', usamos esa (estructura nueva)
    if 'target' in config['database']:
        db_config = config['database']['target']
        # Mapeamos nombres si es necesario (database -> bbdd)
        db_name = db_config.get('database', db_config.get('bbdd'))
    else:
        # Estructura antigua plana
        db_config = config['database']
        db_name = db_config.get('bbdd', db_config.get('database'))

    connection_string = (
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_name}"
    )
    
    engine = create_engine(
        connection_string,
        connect_args={'ssl': {'fake_flag_to_enable_tls': True}},
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    
    return engine


def clasificar_nivel_riesgo(score):
    """
    Clasifica el score en nivel de riesgo.
    
    Args:
        score: float entre 0 y 1
        
    Returns:
        str: 'critico', 'moderado', 'bajo', o 'normal'
    """
    if score >= UMBRALES_NIVEL['critico']:
        return 'critico'
    elif score >= UMBRALES_NIVEL['moderado']:
        return 'moderado'
    elif score >= UMBRALES_NIVEL['bajo']:
        return 'bajo'
    else:
        return 'normal'


def preparar_predicciones(predictions_df, de_modelo_version):
    """
    Prepara el DataFrame de predicciones para insertar en BD.
    """
    log.info("  Preparando predicciones para inserción...")
    
    # ============================================================
    # CREAR DATAFRAME LIMPIO DESDE CERO
    # ============================================================
    # Seleccionar solo columnas necesarias y crear DataFrame nuevo
    columnas_necesarias = ['id_maquina_dfos', 'id_linea', 'score', 'de_causas_raiz']
    
    # Detectar nombre de columna timestamp
    col_timestamp = None
    if 'timestamp_ventana' in predictions_df.columns:
        col_timestamp = 'timestamp_ventana'
    elif 'fe_ventana' in predictions_df.columns:
        col_timestamp = 'fe_ventana'
    else:
        raise ValueError("No se encuentra columna de timestamp (timestamp_ventana o fe_ventana)")
    
    columnas_necesarias.append(col_timestamp)
    
    # Crear DataFrame limpio
    df = predictions_df[columnas_necesarias].copy()
    
    # ELIMINAR DUPLICADOS INMEDIATAMENTE
    duplicados_antes = len(df)
    df = df.drop_duplicates(subset=['id_maquina_dfos', col_timestamp], keep='last')
    duplicados_eliminados = duplicados_antes - len(df)
    
    if duplicados_eliminados > 0:
        log.warning(f"  DUPLICADOS ELIMINADOS: {duplicados_eliminados} registros")
    
    # Renombrar timestamp a fe_ventana si es necesario
    if col_timestamp == 'timestamp_ventana':
        df = df.rename(columns={'timestamp_ventana': 'fe_ventana'})
    
    # Resetear índice para evitar problemas
    df = df.reset_index(drop=True)
    
    # AHORA SÍ convertir a datetime (sin duplicados)
    df['fe_ventana'] = pd.to_datetime(df['fe_ventana'], errors='coerce')
    
    # Eliminar filas con fechas inválidas
    df = df.dropna(subset=['fe_ventana'])
    
    # ============================================================
    # CALCULAR CAMPOS DERIVADOS
    # ============================================================
    
    # fl_pred_modelo (1 si score >= 0.31)
    df['fl_pred_modelo'] = (df['score'] >= UMBRAL_PRED_MODELO).astype(int)
    
    # de_nivel_riesgo
    df['de_nivel_riesgo'] = df['score'].apply(clasificar_nivel_riesgo)
    
    # nm_score (redondear a 4 decimales)
    df['nm_score'] = df['score'].round(4)
    
    # de_modelo_version
    df['de_modelo_version'] = de_modelo_version
    
    # ============================================================
    # SELECCIONAR COLUMNAS FINALES
    # ============================================================
    columnas_finales = [
        'id_maquina_dfos',
        'id_linea',
        'fe_ventana',
        'nm_score',
        'de_nivel_riesgo',
        'fl_pred_modelo',
        'de_modelo_version',
        'de_causas_raiz'
    ]
    
    df_final = df[columnas_finales].copy()
    
    # VERIFICACIÓN FINAL: No debe haber duplicados
    duplicados_final = df_final.duplicated(subset=['id_maquina_dfos', 'fe_ventana'], keep=False).sum()
    if duplicados_final > 0:
        log.error(f"  CRÍTICO: Aún hay {duplicados_final} duplicados después de limpieza")
        # Forzar eliminación
        df_final = df_final.drop_duplicates(subset=['id_maquina_dfos', 'fe_ventana'], keep='last')
    
    log.info(f"  Predicciones preparadas: {len(df_final):,} registros")
    log.info(f"  Máquinas únicas: {df_final['id_maquina_dfos'].nunique()}")
    log.info(f"  Rango temporal: {df_final['fe_ventana'].min()} a {df_final['fe_ventana'].max()}")
    
    return df_final


def save_predictions_to_db(predictions_df, config, de_modelo_version=None):
    """
    Guarda predicciones en la tabla bui_predicciones_hora.
    Versión optimizada: Borra la hora actual antes de insertar para tener siempre el dato más fresco.
    """
    log.info("=" * 80)
    log.info(" GUARDANDO PREDICCIONES EN BASE DE DATOS")
    log.info("=" * 80)
    
    if len(predictions_df) == 0:
        log.warning(" DataFrame vacío, nada que guardar")
        return 0
    
    # Obtener versión del modelo
    if de_modelo_version is None:
        de_modelo_version = config.get('model', {}).get('version', 'v1.0')
    
    # Preparar datos
    df_insert = preparar_predicciones(predictions_df, de_modelo_version)
    
    # Estadísticas antes de insertar
    log.info("\n Resumen de predicciones a guardar:")
    log.info(f"   Total: {len(df_insert):,}")
    
    dist_nivel = df_insert['de_nivel_riesgo'].value_counts()
    for nivel in ['critico', 'moderado', 'bajo', 'normal']:
        count = dist_nivel.get(nivel, 0)
        pct = count / len(df_insert) * 100
        emoji = {'critico': 'ROJO', 'moderado': 'AMARILLO', 'bajo': 'VERDE', 'normal': 'GRIS'}.get(nivel)
        log.info(f"   {emoji} {nivel.upper()}: {count:,} ({pct:.1f}%)")
    
    pred_positivas = df_insert['fl_pred_modelo'].sum()
    log.info(f"   Predicciones positivas (fl_pred_modelo=1): {pred_positivas:,} ({pred_positivas/len(df_insert)*100:.1f}%)")
    
    # Conectar a BD
    engine = get_db_engine(config)
    table_name = 'bui_predicciones_hora'
    
    try:
        # Verificar conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info(f" Conexión a BD establecida")
        
        # ==========================================================
        # NUEVO: LIMPIAR LA HORA ANTES DE INSERTAR (Evita duplicados)
        # ==========================================================
        with engine.begin() as conn:
            hora_exacta = df_insert['fe_ventana'].iloc[0]
            query = text(f"DELETE FROM {table_name} WHERE fe_ventana = :hora")
            conn.execute(query, {"hora": hora_exacta})
            log.info(f" Limpiados registros previos para la hora {hora_exacta}")
        # ==========================================================

        # Insertar usando REPLACE INTO (el método más seguro)
        log.info(f"\n Insertando nuevas predicciones para la hora {hora_exacta}...")
        
        registros_insertados = 0
        
        # Insertar en chunks para no saturar la memoria
        chunk_size = 1000
        total_chunks = (len(df_insert) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(df_insert), chunk_size):
            chunk = df_insert.iloc[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            
            with engine.connect() as conn:
                for _, row in chunk.iterrows():
                    try:
                        insert_sql = text(f"""
                            REPLACE INTO {table_name} 
                            (id_maquina_dfos, id_linea, fe_ventana, nm_score, 
                             de_nivel_riesgo, fl_pred_modelo, de_modelo_version,de_causas_raiz)
                            VALUES 
                            (:id_maquina_dfos, :id_linea, :fe_ventana, :nm_score,
                             :de_nivel_riesgo, :fl_pred_modelo, :de_modelo_version, :de_causas_raiz)
                        """)
                        
                        conn.execute(insert_sql, {
                            'id_maquina_dfos': row['id_maquina_dfos'],
                            'id_linea': int(row['id_linea']),
                            'fe_ventana': row['fe_ventana'],
                            'nm_score': float(row['nm_score']),
                            'de_nivel_riesgo': row['de_nivel_riesgo'],
                            'fl_pred_modelo': int(row['fl_pred_modelo']),
                            'de_modelo_version': row['de_modelo_version'],
                            'de_causas_raiz': row['de_causas_raiz']
                        })
                        registros_insertados += 1
                            
                    except Exception as e:
                        log.warning(f"   Error insertando fila para {row['id_maquina_dfos']}: {e}")
                
                conn.commit()
            
            if chunk_num % 10 == 0 or chunk_num == total_chunks:
                log.info(f"   Chunk {chunk_num}/{total_chunks} procesado")
        
        log.info(f"\n Inserción completada exitosamente:")
        log.info(f"   Registros frescos insertados: {registros_insertados:,}")
        
        return registros_insertados
        
    except Exception as e:
        log.error(f" Error guardando predicciones: {e}")
        raise
        
    finally:
        engine.dispose()

def save_predictions_batch(predictions_df, config, de_modelo_version=None):
    """
    Versión optimizada para inserción masiva usando pandas to_sql.
    Más rápida pero no maneja duplicados individualmente.
    
    Args:
        predictions_df: DataFrame con predicciones
        config: dict con configuración
        de_modelo_version: str con versión del modelo
        
    Returns:
        int: número de registros procesados
    """
    log.info("=" * 80)
    log.info(" GUARDANDO PREDICCIONES (MODO BATCH)")
    log.info("=" * 80)
    
    if len(predictions_df) == 0:
        log.warning(" DataFrame vacío, nada que guardar")
        return 0
    
    # Obtener versión del modelo
    if de_modelo_version is None:
        de_modelo_version = config.get('model', {}).get('version', 'v1.0')
    
    # Preparar datos
    df_insert = preparar_predicciones(predictions_df, de_modelo_version)
    
    # Conectar a BD
    engine = get_db_engine(config)
    table_name = 'bui_predicciones_hora'
    
    try:
        # Insertar usando pandas (más rápido)
        log.info(f" Insertando {len(df_insert):,} registros en '{table_name}'...")
        
        df_insert.to_sql(
            name=table_name,
            con=engine,
            if_exists='append',
            index=False,
            chunksize=1000,
            method='multi'
        )
        
        log.info(f" {len(df_insert):,} registros insertados")
        return len(df_insert)
        
    except Exception as e:
        if 'Duplicate entry' in str(e):
            log.warning(f" Algunos registros ya existían (duplicados)")
            # Fallback a inserción individual
            log.info("   Reintentando con INSERT IGNORE...")
            return save_predictions_to_db(predictions_df, config, de_modelo_version)
        else:
            log.error(f" Error guardando predicciones: {e}")
            raise
            
    finally:
        engine.dispose()


# ============================================================================
# EJECUCIÓN STANDALONE
# ============================================================================

def main():
    """
    Ejecución standalone - lee predicciones de CSV temporal.
    Útil para testing o re-procesamiento.
    """
    config = load_config()
    logger = setup_logging(config)
    
    log.info("=" * 80)
    log.info(" SAVE_PREDICTIONS.PY - MODO STANDALONE")
    log.info("=" * 80)
    
    # Buscar archivo de predicciones más reciente
    predictions_dir = Path(__file__).parent / 'outputs'
    predictions_files = list(predictions_dir.glob('predictions_*.csv'))
    
    if not predictions_files:
        log.error(" No se encontraron archivos de predicciones en outputs/")
        return 1
    
    # Usar el más reciente
    latest_file = max(predictions_files, key=lambda x: x.stat().st_mtime)
    log.info(f" Cargando predicciones desde: {latest_file}")
    
    predictions_df = pd.read_csv(latest_file)
    log.info(f"   Registros cargados: {len(predictions_df):,}")
    
    # Guardar en BD
    try:
        registros = save_predictions_to_db(predictions_df, config)
        log.info(f"\n COMPLETADO: {registros:,} predicciones guardadas")
        return 0
    except Exception as e:
        log.error(f"\n ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
