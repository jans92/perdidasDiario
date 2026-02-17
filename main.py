"""
main.py - Sistema de Mantenimiento Predictivo
==============================================

Orquestador principal del sistema de predicción.

Se ejecuta 1 vez al día a las 6:00 AM.
Genera predicciones para el próximo día completo (24 horas).

Nota: El score es único para cada máquina-hora.
"""

import sys
import os

if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        else:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception as e:
        print(f"Advertencia: No se pudo configurar UTF-8: {e}")

import logging
import yaml
import signal
import tempfile
import warnings
import pandas as pd
from pandas.errors import PerformanceWarning
from save_predictions import save_predictions_to_db
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv('.env')

try:
    import fcntl
except ImportError:
    fcntl = None

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.db_connector import DatabaseConnector
from src.feature_engineering import FeatureEngineer
from src.predictor import Predictor
from src.utils import setup_logging, send_email_alert
from src.config_loader import load_config


def setup_timeout(timeout_minutes):
    """Configura timeout para evitar ejecuciones infinitas (Solo Unix)."""
    if not hasattr(signal, 'SIGALRM'):
        logging.warning("[TIMEOUT] No soportado en Windows. Se omitirá el límite de tiempo.")
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Ejecución superó {timeout_minutes} minutos")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_minutes * 60)


def acquire_lock():
    """Adquiere lock para evitar ejecuciones simultáneas."""
    lockfile_path = os.path.join(tempfile.gettempdir(), 'predictive_maintenance.lock')
    lockfile = open(lockfile_path, 'w')

    if fcntl is None:
        logging.warning("[LOCK] No soportado en Windows. Se ejecutará sin exclusión mutua.")
        return lockfile

    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lockfile
    except IOError:
        logging.error("[LOCK] Otra instancia está corriendo. Saliendo...")
        sys.exit(0)


def release_lock(lockfile):
    """Libera el lock de forma segura."""
    if lockfile and not lockfile.closed:
        if fcntl:
            fcntl.flock(lockfile, fcntl.LOCK_UN)
        lockfile.close()


def obtener_maquinas_con_actividad(db, lookback_hours=168):
    """
    Obtiene lista de máquinas que tuvieron actividad en el período lookback.
    
    Args:
        db: DatabaseConnector conectado
        lookback_hours: Horas a mirar hacia atrás
    
    Returns:
        set con id_maquina_dfos únicas
    """
    query = f"""
    SELECT DISTINCT id_maquina_dfos
    FROM bui_perdida
    WHERE fe_fecha >= NOW() - INTERVAL {lookback_hours} HOUR
    """
    
    df = db.execute_query(query)
    maquinas = set(df['id_maquina_dfos'].unique())
    
    return maquinas


def generar_predicciones_dia_completo(predictor, feature_engineer, 
                                      bui_perdida, bui_pm_ewo, bui_line,
                                      fecha_inicio, logger):
    """
    Genera predicciones para todas las horas del día (00:00 a 23:00).
    
     FIX: Sin loop infinito + validación correcta
    """
    logger.info(f"\nGenerando predicciones para día completo: {fecha_inicio.date()}")
    
    todas_predicciones = []
    horas_sin_datos = 0
    horas_error = 0
    
    # PASO CRÍTICO: Feature Engineering SOLO UNA VEZ (datos históricos son estáticos)
    logger.info("[PRE-CÁLCULO] Matriz de características (datos históricos)...")
    try:
        X = feature_engineer.transform(bui_perdida, bui_pm_ewo, bui_line)
        logger.info(f"   Matriz X: {X.shape} ({X['id_maquina_dfos'].nunique()} máquinas)")
    except Exception as e:
        logger.error(f"   Error en feature engineering: {e}", exc_info=True)
        return pd.DataFrame()
    
    if len(X) == 0:
        logger.error("   Feature engineering devolvió matriz vacía")
        return pd.DataFrame()
    
    # BUCLE: Solo generar predicciones (sin re-calcular features)
    for hora_offset in range(2, 3):
        ventana_actual = fecha_inicio + timedelta(hours=hora_offset)
        
        logger.info(f"  Hora {hora_offset:02d}: {ventana_actual.strftime('%Y-%m-%d %H:00')}")
        
        try:
            #  Predecir con X ya calculado
            predictions = predictor.predict(X)
            
            if len(predictions) == 0:
                logger.warning(f"     Predictor devolvió resultado vacío")
                horas_sin_datos += 1
                continue
            
            #  Asignar timestamp correcto para esta hora
            predictions['timestamp_ventana'] = ventana_actual
            
            todas_predicciones.append(predictions)
            logger.info(f"     {len(predictions):,} predicciones generadas")
        
        except Exception as e:
            logger.error(f"     Error prediciendo hora {hora_offset}: {e}")
            horas_error += 1
            continue  # ← Ahora SÍ avanza a siguiente hora
    
    #  Validar resultado final
    if not todas_predicciones:
        logger.error(f" No se generaron predicciones (sin_datos={horas_sin_datos}, errores={horas_error})")
        return pd.DataFrame()
    
    predictions_df = pd.concat(todas_predicciones, ignore_index=True)
    
    logger.info(f"\n COMPLETADO:")
    logger.info(f"  - Total predicciones: {len(predictions_df):,}")
    logger.info(f"  - Máquinas cubiertas: {predictions_df['id_maquina_dfos'].nunique()}")
    logger.info(f"  - Horas sin datos: {horas_sin_datos}/24")
    logger.info(f"  - Horas con error: {horas_error}/24")
    
    return predictions_df


def main():
    """Función principal de ejecución."""
    
    try:
        config = load_config(str(BASE_DIR / 'config.yaml'))
    except Exception as e:
        print(f"[ERROR] Error crítico cargando config: {e}")
        sys.exit(1)
    
    logger = setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("[INICIO] Sistema de Mantenimiento Predictivo - Ejecución Diaria")
    logger.info("=" * 80)
    logger.info(f"[TIMESTAMP] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    lockfile = acquire_lock()
    
    timeout_minutes = config.get('execution', {}).get('timeout_minutes', 120)
    setup_timeout(timeout_minutes)
    
    db = None
    
    try:
        # =====================================================================
        # PASO 1: EXTRAER DATOS
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("[PASO 1] Extrayendo datos de base de datos...")
        logger.info("=" * 80)
        
        db = DatabaseConnector(config)
        db.connect_source()
        
        # Parámetros de extracción
        lookback_hours = config.get('execution', {}).get('lookback_hours', 168)
        lookback_days_maintenance = config.get('execution', {}).get('lookback_days_maintenance', 60)
        
        # Fecha de referencia (opcional, para testing)
        fecha_referencia = config.get('execution', {}).get('fecha_referencia', None)
        
        if fecha_referencia:
            logger.info(f"[MODO TEST] Fecha de referencia: {fecha_referencia}")
            fecha_hoy = pd.to_datetime(fecha_referencia).date()
        else:
            logger.info(f"[MODO PROD] Fecha actual: {datetime.now().strftime('%Y-%m-%d')}")
            fecha_hoy = datetime.now().date()
        
        # Extraer datos
        bui_perdida = db.extract_events_data(
            lookback_hours=lookback_hours,
            fecha_referencia=fecha_referencia
        )
        
        bui_pm_ewo = db.extract_maintenance_data(
            lookback_days=lookback_days_maintenance,
            fecha_referencia=fecha_referencia
        )
        
        bui_line = db.extract_line_metadata()
        # =====================================================================
        # NUEVO FILTRO: PROCESAR SOLO LA LÍNEA BALATON
        # =====================================================================
        ID_LINEA_BALATON = 52777
        logger.info(f"\n[FILTRO ACTIVO] Reduciendo datos solo a la línea {ID_LINEA_BALATON}...")
        
        if not bui_perdida.empty and 'id_linea' in bui_perdida.columns:
            bui_perdida = bui_perdida[bui_perdida['id_linea'] == ID_LINEA_BALATON]
            
        if not bui_pm_ewo.empty and 'id_linea' in bui_pm_ewo.columns:
            bui_pm_ewo = bui_pm_ewo[bui_pm_ewo['id_linea'] == ID_LINEA_BALATON]
            
        if not bui_line.empty and 'id_linea' in bui_line.columns:
            bui_line = bui_line[bui_line['id_linea'] == ID_LINEA_BALATON]
        # =====================================================================
        
        logger.info(f"\n[EXTRACCION] Datos extraídos:")
        logger.info(f" - bui_perdida: {len(bui_perdida):,} registros ({lookback_hours} horas)")
        logger.info(f" - bui_pm_ewo: {len(bui_pm_ewo):,} registros ({lookback_days_maintenance} días)")
        logger.info(f" - bui_line: {len(bui_line):,} registros")
        
        if not bui_perdida.empty:
            logger.info(f" - Máquinas únicas: {bui_perdida['id_maquina_dfos'].nunique()}")
        
        # =====================================================================
        # VALIDACIÓN: Verificar que hay datos para procesar
        # =====================================================================
        
        if len(bui_perdida) == 0:
            logger.warning("\n[SIN DATOS] No hay eventos de producción (bui_perdida)")
            logger.warning(" Posibles causas:")
            logger.warning(" - No hay actividad de producción en el período")
            logger.warning(" - Los datos aún no se han cargado en la BD")
            logger.warning(" - Problema con conexión o filtros de fecha")
            logger.info("\n[FIN] Ejecución finalizada sin predicciones (sin datos)")
            return 0
        
        # =====================================================================
        # PASO 2: FEATURE ENGINEERING
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("[PASO 2] Generando variables (Feature Engineering)...")
        logger.info("=" * 80)
        
        fe = FeatureEngineer(config)
        
        # Verificar inicialmente
        X = fe.transform(bui_perdida, bui_pm_ewo, bui_line)
        
        logger.info(f"\n[FEATURES] Matriz de características generada:")
        logger.info(f" - Dimensiones: {X.shape}")
        logger.info(f" - Máquinas procesadas: {X['id_maquina_dfos'].nunique()}")
        
        # =====================================================================
        # PASO 3: PREDICCIÓN CON MODELO
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("[PASO 3] Ejecutando modelo predictivo...")
        logger.info("=" * 80)
        
        predictor = Predictor(config)
        
        # =====================================================================
        # GENERACIÓN DE PREDICCIONES PARA DÍA COMPLETO
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("[PASO 4] Generando predicciones para el día completo...")
        logger.info("=" * 80)
        
        # Fecha inicial para predicciones (hoy a las 00:00)
        if fecha_referencia:
            fecha_predicciones = pd.to_datetime(fecha_referencia)
        else:
            fecha_predicciones = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        predictions_df = generar_predicciones_dia_completo(
            predictor, fe,
            bui_perdida, bui_pm_ewo, bui_line,
            fecha_predicciones,
            logger
        )
        
        if len(predictions_df) == 0:
            logger.error("[ERROR] No se generaron predicciones")
            return 1
        
        # =====================================================================
        # PASO 5: GUARDAR PREDICCIONES EN BASE DE DATOS
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("[PASO 5] Guardando predicciones en base de datos...")
        logger.info("=" * 80)
        
        registros_guardados = save_predictions_to_db(predictions_df, config)
        
        logger.info(f"\n[ÉXITO] {registros_guardados:,} predicciones guardadas en BD")
        
        # =====================================================================
        # RESUMEN FINAL
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("[RESUMEN] Ejecución completada exitosamente")
        logger.info("=" * 80)
        logger.info(f" Fecha procesada: {fecha_hoy}")
        logger.info(f" Predicciones para: próximas 24 horas")
        logger.info(f" Total predicciones: {registros_guardados:,}")
        logger.info(f" Máquinas cubiertas: {predictions_df['id_maquina_dfos'].nunique()}")
        logger.info(f" Duración: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
    
    except TimeoutError as e:
        logger.error(f"\n[TIMEOUT] {e}")
        return 1
    
    except Exception as e:
        logger.error(f"\n[ERROR CRÍTICO] {e}", exc_info=True)
        
        try:
            send_email_alert(
                config,
                subject="[ERROR] Mantenimiento Predictivo - Ejecución Fallida",
                body=f"Error en ejecución de predicciones:\n\n{str(e)}"
            )
        except:
            pass
        
        return 1
    
    finally:
        if db:
            db.close_connections()
            logger.info("\nConexión a BD cerrada")
        
        release_lock(lockfile)
        logger.info("[FIN] Proceso finalizado")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
