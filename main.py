"""
main.py - Orquestador del sistema de mantenimiento predictivo.
Fábrica: Veszprém (id_fabrica=21). Ejecución diaria a las 6:00 AM.
Genera predicciones hora a hora para las próximas 24h.
"""

import sys
import os
import signal
import tempfile
import logging
import warnings

import yaml
import pandas as pd
from pandas.errors import PerformanceWarning
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

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
        print(f"Advertencia: no se pudo configurar UTF-8: {e}")

try:
    import fcntl
except ImportError:
    fcntl = None

load_dotenv('.env')

dirBase = Path(__file__).parent
sys.path.insert(0, str(dirBase))

from src.db_connector import DatabaseConnector
from src.feature_engineering import FeatureEngineer
from src.predictor import Predictor
from src.utils import setup_logging, send_email_alert
from src.config_loader import load_config
from save_predictions import guardarPrediccionesEnBd


ID_FABRICA_VESZPREM = 21


def configurarTimeout(minutos):
    # TIMEOUT - solo sistemas UNIX
    if not hasattr(signal, 'SIGALRM'):
        logging.warning("[TIMEOUT] No disponible en Windows")
        return

    def _manejador(signum, frame):
        raise TimeoutError(f"Ejecución superó {minutos} minutos")

    signal.signal(signal.SIGALRM, _manejador)
    signal.alarm(minutos * 60)


def adquirirBloqueo():
    # BLOQUEO DE PROCESO - evita ejecuciones simultáneas
    rutaLock = os.path.join(tempfile.gettempdir(), 'predictive_maintenance.lock')
    archivoLock = open(rutaLock, 'w')

    if fcntl is None:
        logging.warning("[LOCK] No disponible en Windows")
        return archivoLock

    try:
        fcntl.flock(archivoLock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return archivoLock
    except IOError:
        logging.error("[LOCK] Otra instancia ya en ejecución")
        sys.exit(0)


def liberarBloqueo(archivoLock):
    if archivoLock and not archivoLock.closed:
        if fcntl:
            fcntl.flock(archivoLock, fcntl.LOCK_UN)
        archivoLock.close()


def generarPrediccionesDia(predictor, ingCaracteristicas, buiPerdida, buiPmEwo,
                            buiLine, fechaBase, logger):
    # PREDICCIÓN 24H - genera una fila por máquina para cada hora del día siguiente
    logger.info(f"\n[PASO 4] Generando predicciones: {fechaBase.date()}")

    logger.info("  Calculando matriz de características...")
    try:
        X = ingCaracteristicas.transform(buiPerdida, buiPmEwo, buiLine)
        logger.info(f"  Matriz X: {X.shape} | {X['id_maquina_dfos'].nunique()} máquinas")
    except Exception as e:
        logger.error(f"  Error en feature engineering: {e}", exc_info=True)
        return pd.DataFrame()

    if X.empty:
        logger.error("  Matriz de características vacía")
        return pd.DataFrame()

    acumulado = []
    horasSinDatos = 0
    horasConError = 0

    for offsetHora in range(1, 25):
        ventana = fechaBase + timedelta(hours=offsetHora)
        try:
            predicciones = predictor.predict(X)
            if predicciones.empty:
                horasSinDatos += 1
                continue
            predicciones['timestamp_ventana'] = ventana
            acumulado.append(predicciones)
        except Exception as e:
            logger.error(f"  Error hora +{offsetHora}h: {e}")
            horasConError += 1

    if horasSinDatos:
        logger.warning(f"  Horas sin datos: {horasSinDatos}/24")
    if horasConError:
        logger.warning(f"  Horas con error: {horasConError}/24")

    if not acumulado:
        logger.error("  No se generó ninguna predicción")
        return pd.DataFrame()

    return pd.concat(acumulado, ignore_index=True)


def main():
    try:
        config = load_config(str(dirBase / 'config.yaml'))
    except Exception as e:
        print(f"[ERROR] No se pudo cargar config.yaml: {e}")
        sys.exit(1)

    logger = setup_logging(config)
    logger.info("=" * 70)
    logger.info("[INICIO] Mantenimiento Predictivo — Fábrica VESZPRÉM")
    logger.info("=" * 70)

    archivoLock = adquirirBloqueo()
    minutosTimeout = config.get('execution', {}).get('timeout_minutes', 120)
    configurarTimeout(minutosTimeout)

    bd = None
    try:
        # PASO 1: EXTRACCIÓN DE DATOS
        bd = DatabaseConnector(config)
        bd.connect_source()

        ejec = config.get('execution', {})
        horasHistorial     = ejec.get('lookback_hours', 168)
        diasMantenimiento  = ejec.get('lookback_days_maintenance', 60)
        fechaRef           = ejec.get('fecha_referencia', None)
        fechaHoy           = pd.to_datetime(fechaRef).date() if fechaRef else datetime.now().date()

        buiPerdida = bd.extract_events_data(lookback_hours=horasHistorial, fecha_referencia=fechaRef)
        buiPmEwo   = bd.extract_maintenance_data(lookback_days=diasMantenimiento, fecha_referencia=fechaRef)
        buiLine    = bd.extract_line_metadata()

        # FILTRO VESZPRÉM - solo líneas de la fábrica objetivo
        buiLine = buiLine[buiLine['id_fabrica'] == ID_FABRICA_VESZPREM].copy()
        if buiLine.empty:
            logger.error("[FILTRO] Sin líneas activas para Veszprém")
            return 0

        idsLineas = buiLine['id_linea'].unique().tolist()
        logger.info(f"[FILTRO] {len(idsLineas)} líneas activas: {idsLineas}")

        buiPerdida = buiPerdida[buiPerdida['id_linea'].isin(idsLineas)].copy()
        buiPmEwo   = buiPmEwo[buiPmEwo['id_linea'].isin(idsLineas)].copy()

        if buiPerdida.empty:
            logger.warning("[DATOS] Sin eventos de pérdida para Veszprém")
            return 0

        logger.info(f"[DATOS] {buiPerdida['id_maquina_dfos'].nunique()} máquinas con eventos")

        # PASO 2: FEATURE ENGINEERING
        logger.info("\n[PASO 2] Feature engineering...")
        ingCaracteristicas = FeatureEngineer(config)
        ingCaracteristicas.transform(buiPerdida, buiPmEwo, buiLine)

        # PASO 3: CARGA DEL PREDICTOR
        logger.info("\n[PASO 3] Cargando modelo predictor...")
        predictor = Predictor(config)

        # PASO 4: PREDICCIONES 24H
        fechaBase = datetime.combine(fechaHoy, datetime.min.time())
        dfPredicciones = generarPrediccionesDia(
            predictor, ingCaracteristicas,
            buiPerdida, buiPmEwo, buiLine,
            fechaBase, logger
        )

        if dfPredicciones.empty:
            logger.error("[ERROR] No se generaron predicciones")
            return 1

        # PASO 5: PERSISTENCIA EN BASE DE DATOS
        logger.info("\n[PASO 5] Guardando predicciones en BD...")
        registrosGuardados = guardarPrediccionesEnBd(dfPredicciones, config)
        logger.info(f"\n[OK] {registrosGuardados:,} registros guardados")
        return 0

    except Exception as e:
        logger.error(f"\n[ERROR CRÍTICO] {e}", exc_info=True)
        try:
            send_email_alert(config, subject="[ERROR] Predicciones fallidas", body=str(e))
        except Exception:
            pass
        return 1

    finally:
        if bd:
            bd.close_connections()
        liberarBloqueo(archivoLock)
        logger.info("[FIN] Proceso finalizado")


if __name__ == "__main__":
    sys.exit(main())