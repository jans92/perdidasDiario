"""
update_targets.py - Actualización de targets reales en predicciones pasadas.
Lee predicciones de DEV, contrasta con eventos reales de PROD y actualiza
fl_target_real y fl_acierto. Ejecución diaria recomendada a las 23:00.
"""

import sys
import os
import logging

import yaml
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logging

log = logging.getLogger(__name__)

# UMBRAL MÍNIMO (minutos) para considerar una parada como falla grave
UMBRAL_DURACION_CON_EWO    = 10
UMBRAL_DURACION_LARGA      = 60
UMBRAL_DURACION_TIPO_CRITICO = 20

TIPOS_CRITICOS = [
    'Breakdown - Mechanical',
    'Breakdown - Electrical',
    'Breakdown - Instrumentation & Control',
]

LIMITE_BATCH = 50_000


def _cargarConfig():
    rutaConfig = Path(__file__).parent / 'config.yaml'
    with open(rutaConfig, 'r') as f:
        return yaml.safe_load(f)


def _crearConexion(prefijo):
    # CONEXIÓN BD - lee credenciales desde variables de entorno según prefijo (DB_PROD / DB_DEV)
    host     = os.getenv(f'{prefijo}_HOST')
    user     = os.getenv(f'{prefijo}_USER')
    password = os.getenv(f'{prefijo}_PASSWORD')
    database = os.getenv(f'{prefijo}_NAME')
    port     = os.getenv(f'{prefijo}_PORT', '3306')

    if not all([host, user, password, database]):
        raise ValueError(f"Variables de entorno incompletas para {prefijo}")

    cadena = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(
        cadena,
        connect_args={'ssl': {'fake_flag_to_enable_tls': True}},
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    log.info(f"Conexión establecida: {prefijo} → {database}@{host}")
    return engine


def _obtenerPrediccionesPendientes(engineDev, limite=LIMITE_BATCH):
    # LECTURA DEV - predicciones sin target real, con ventana ya cumplida (+5h de margen)
    log.info("Obteniendo predicciones pendientes de actualizar...")

    query = text("""
        SELECT id_prediccion, id_maquina_dfos, id_linea,
               fe_ventana, nm_score, fl_pred_modelo
        FROM bui_predicciones_hora_dia
        WHERE fl_target_real IS NULL
          AND fe_ventana < NOW() - INTERVAL 5 HOUR
        ORDER BY fe_ventana ASC
        LIMIT :limite
    """)

    df = pd.read_sql(query, engineDev, params={'limite': limite})
    log.info(f"Pendientes: {len(df):,} | máquinas: {df['id_maquina_dfos'].nunique()} "
             f"| rango: {df['fe_ventana'].min()} → {df['fe_ventana'].max()}")
    return df


def _cargarPerdidasEnCache(engineProd, fechaInicio, fechaFin):
    # CACHÉ PROD - carga todas las paradas del período en RAM para búsquedas O(1) por máquina
    log.info("Cargando pérdidas en caché desde PROD...")

    query = text("""
        SELECT id_maquina_dfos, fe_inicio, fe_fin,
               de_perdida_2, de_perdida_3
        FROM bui_perdida
        WHERE de_perdida_2 = 'Breakdown & Equipment Failure Time'
          AND fe_inicio BETWEEN :inicio AND :fin
        ORDER BY id_maquina_dfos, fe_inicio
    """)

    df = pd.read_sql(query, engineProd, params={
        'inicio': fechaInicio - timedelta(days=1),
        'fin':    fechaFin    + timedelta(days=1),
    })

    log.info(f"Pérdidas en caché: {len(df):,}")
    return df


def _obtenerEwosValidas(engineProd, fechaInicio, fechaFin):
    # EWOS VÁLIDAS - correctivas cerradas en el período, usadas para enriquecer el target
    log.info("Obteniendo EWOs válidas de PROD...")

    query = text("""
        SELECT DISTINCT id_perdida
        FROM bui_pm_ewo
        WHERE fl_borrador = 0
          AND id_perdida IS NOT NULL
          AND fe_inicio_averia BETWEEN :inicio AND :fin
          AND (fe_cerrar IS NOT NULL OR fe_fin_mto IS NOT NULL)
    """)

    df = pd.read_sql(query, engineProd, params={
        'inicio': fechaInicio,
        'fin':    fechaFin,
    })

    ewosSet = set(df['id_perdida'].tolist())
    log.info(f"EWOs válidas: {len(ewosSet):,}")
    return ewosSet


def _haFallaGraveEnVentana(dfPerdidas, idMaquina, feVentana):
    # DETECCIÓN DE FALLA - ventana ±75 min alrededor de la hora predicha
    inicioBusqueda = feVentana - timedelta(minutes=15)
    finBusqueda    = feVentana + timedelta(hours=1, minutes=15)

    perdidasMaquina = dfPerdidas[dfPerdidas['id_maquina_dfos'] == idMaquina]
    if perdidasMaquina.empty:
        return 0

    solapadas = perdidasMaquina[
        ((perdidasMaquina['fe_inicio'] >= inicioBusqueda) & (perdidasMaquina['fe_inicio'] <= finBusqueda)) |
        ((perdidasMaquina['fe_fin']    >= inicioBusqueda) & (perdidasMaquina['fe_fin']    <= finBusqueda)) |
        ((perdidasMaquina['fe_inicio'] <= inicioBusqueda) & (perdidasMaquina['fe_fin']    >= finBusqueda))
    ]

    return int(not solapadas.empty)


def _actualizarPrediccion(engineDev, idPrediccion, targetReal, predModelo):
    acierto = int(predModelo == targetReal)
    query = text("""
        UPDATE bui_predicciones_hora_dia
        SET fl_target_real = :targetReal,
            fl_acierto     = :acierto
        WHERE id_prediccion = :idPrediccion
    """)
    with engineDev.begin() as conn:
        conn.execute(query, {
            'targetReal':    targetReal,
            'acierto':       acierto,
            'idPrediccion':  idPrediccion,
        })


def _imprimirMatrizConfusion(engineDev, fechaMin, fechaMax):
    # MÉTRICAS DEL PERÍODO - precision, recall, F1 y accuracy sobre predicciones actualizadas
    queryMetricas = text("""
        SELECT fl_pred_modelo, fl_target_real, COUNT(*) AS cantidad
        FROM bui_predicciones_hora_dia
        WHERE fl_target_real IS NOT NULL
          AND fe_ventana BETWEEN :inicio AND :fin
        GROUP BY fl_pred_modelo, fl_target_real
    """)

    dfMetricas = pd.read_sql(queryMetricas, engineDev, params={
        'inicio': fechaMin,
        'fin':    fechaMax - timedelta(hours=24),
    })

    tp = dfMetricas[(dfMetricas['fl_pred_modelo']==1) & (dfMetricas['fl_target_real']==1)]['cantidad'].sum()
    fp = dfMetricas[(dfMetricas['fl_pred_modelo']==1) & (dfMetricas['fl_target_real']==0)]['cantidad'].sum()
    fn = dfMetricas[(dfMetricas['fl_pred_modelo']==0) & (dfMetricas['fl_target_real']==1)]['cantidad'].sum()
    tn = dfMetricas[(dfMetricas['fl_pred_modelo']==0) & (dfMetricas['fl_target_real']==0)]['cantidad'].sum()
    total = tp + fp + fn + tn

    log.info("\nMatriz de Confusión (período actualizado):")
    log.info(f"+-------------+-------------+-------------+")
    log.info(f"|             |   Real = 1  |   Real = 0  |")
    log.info(f"+-------------+-------------+-------------+")
    log.info(f"|   Pred = 1  |  TP: {tp:>5}  |  FP: {fp:>5}  |")
    log.info(f"|   Pred = 0  |  FN: {fn:>5}  |  TN: {tn:>5}  |")
    log.info(f"+-------------+-------------+-------------+")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        log.info(f"Precisión : {precision:.2%}")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        log.info(f"Recall    : {recall:.2%}")
    if tp + fp > 0 and tp + fn > 0 and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
        log.info(f"F1-Score  : {f1:.2%}")
    if total > 0:
        log.info(f"Accuracy  : {(tp + tn) / total:.2%}")


def actualizarTargets(config):
    """
    Flujo principal:
      1. Conecta a DEV (predicciones) y PROD (eventos reales)
      2. Carga predicciones pendientes de DEV
      3. Carga pérdidas del período en RAM (caché)
      4. Para cada predicción: evalúa target real y actualiza en DEV
      5. Calcula y registra métricas del período
    """
    log.info("=" * 70)
    log.info("ACTUALIZACIÓN DE TARGETS REALES")
    log.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    engineDev  = None
    engineProd = None

    try:
        log.info("\nConectando a DEV (escritura) y PROD (lectura)...")
        engineDev  = _crearConexion('DB_DEV')
        engineProd = _crearConexion('DB_PROD')

        predicciones = _obtenerPrediccionesPendientes(engineDev)
        if predicciones.empty:
            log.info("No hay predicciones pendientes")
            return 0

        fechaMin = predicciones['fe_ventana'].min()
        fechaMax = predicciones['fe_ventana'].max() + timedelta(hours=24)
        log.info(f"\nPeríodo: {fechaMin} → {fechaMax}")

        dfPerdidas = _cargarPerdidasEnCache(engineProd, fechaMin, fechaMax)
        _obtenerEwosValidas(engineProd, fechaMin, fechaMax)  # reservado para lógica futura

        # BUCLE PRINCIPAL - itera por máquina para aprovechar localidad de datos en caché
        totalActualizadas = conFalla = sinFalla = aciertos = 0

        for idMaquina, grupo in tqdm(predicciones.groupby('id_maquina_dfos'),
                                     desc="Procesando máquinas", unit="máq"):
            for _, fila in grupo.iterrows():
                targetReal = _haFallaGraveEnVentana(dfPerdidas, idMaquina, fila['fe_ventana'])
                _actualizarPrediccion(engineDev, fila['id_prediccion'], targetReal, fila['fl_pred_modelo'])

                totalActualizadas += 1
                conFalla  += targetReal
                sinFalla  += 1 - targetReal
                aciertos  += int(fila['fl_pred_modelo'] == targetReal)

        log.info("\n" + "=" * 70)
        log.info("RESUMEN")
        log.info("=" * 70)
        log.info(f"Actualizadas : {totalActualizadas:,}")
        log.info(f"Con falla    : {conFalla:,}  ({conFalla / totalActualizadas * 100:.2f}%)")
        log.info(f"Sin falla    : {sinFalla:,}  ({sinFalla / totalActualizadas * 100:.2f}%)")
        log.info(f"Aciertos     : {aciertos:,}  ({aciertos / totalActualizadas * 100:.2f}%)")

        if conFalla > 0:
            _imprimirMatrizConfusion(engineDev, fechaMin, fechaMax)

        log.info("\nACTUALIZACIÓN COMPLETADA")
        return totalActualizadas

    except Exception as e:
        log.error(f"Error durante actualización: {e}", exc_info=True)
        raise

    finally:
        if engineDev:
            engineDev.dispose()
            log.info("Conexión DEV cerrada")
        if engineProd:
            engineProd.dispose()
            log.info("Conexión PROD cerrada")


def main():
    config = _cargarConfig()
    setup_logging(config)

    try:
        n = actualizarTargets(config)
        log.info(f"Proceso finalizado: {n:,} predicciones actualizadas")
        return 0
    except Exception as e:
        log.error(f"Error crítico: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())