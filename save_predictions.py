"""
save_predictions.py - Persistencia de predicciones en BD.
Escribe en bui_predicciones_hora_dia usando INSERT IGNORE para tolerar duplicados.
Uso como módulo: from save_predictions import guardarPrediccionesEnBd
"""

import sys
import os
import logging

import yaml
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logging

log = logging.getLogger(__name__)

# UMBRALES DE CLASIFICACIÓN DE RIESGO
UMBRAL_MODELO   = 0.3724709494525155
UMBRALES_NIVEL  = {'critico': 0.60, 'moderado': 0.37, 'bajo': 0.15}
TABLA_DESTINO   = 'bui_predicciones_hora_dia'
CHUNK_INSERCION = 1000


def _cargarConfig():
    rutaConfig = Path(__file__).parent / 'config.yaml'
    with open(rutaConfig, 'r') as f:
        return yaml.safe_load(f)


def _crearEngine(config):
    # CONEXIÓN BD - compatible con estructura plana y anidada (database.target)
    dbConfig = config['database'].get('target', config['database'])
    nombreBd = dbConfig.get('database') or dbConfig.get('bbdd')

    cadena = (
        f"mysql+pymysql://{dbConfig['user']}:{dbConfig['password']}"
        f"@{dbConfig['host']}:{dbConfig['port']}/{nombreBd}"
    )
    return create_engine(
        cadena,
        connect_args={'ssl': {'fake_flag_to_enable_tls': True}},
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )


def _clasificarNivelRiesgo(score):
    if score >= UMBRALES_NIVEL['critico']:  return 'critico'
    if score >= UMBRALES_NIVEL['moderado']: return 'moderado'
    if score >= UMBRALES_NIVEL['bajo']:     return 'bajo'
    return 'normal'


def _prepararDataFrame(dfRaw, versionModelo):
    # NORMALIZACIÓN - unifica nombre de columna de timestamp y elimina duplicados
    colTimestamp = next(
        (c for c in ('timestamp_ventana', 'fe_ventana') if c in dfRaw.columns), None
    )
    if colTimestamp is None:
        raise ValueError("No se encontró columna de timestamp (timestamp_ventana / fe_ventana)")

    df = dfRaw[['id_maquina_dfos', 'id_linea', 'score', colTimestamp]].copy()

    nAntes = len(df)
    df = df.drop_duplicates(subset=['id_maquina_dfos', colTimestamp], keep='last')
    if (eliminados := nAntes - len(df)):
        log.warning(f"  Duplicados eliminados: {eliminados}")

    if colTimestamp != 'fe_ventana':
        df = df.rename(columns={colTimestamp: 'fe_ventana'})

    df = df.reset_index(drop=True)
    df['fe_ventana'] = pd.to_datetime(df['fe_ventana'], errors='coerce')
    df = df.dropna(subset=['fe_ventana'])

    # CAMPOS DERIVADOS
    df['fl_pred_modelo']    = (df['score'] >= UMBRAL_MODELO).astype(int)
    df['de_nivel_riesgo']   = df['score'].apply(_clasificarNivelRiesgo)
    df['nm_score']          = df['score'].round(4)
    df['de_modelo_version'] = versionModelo

    columnasFin = [
        'id_maquina_dfos', 'id_linea', 'fe_ventana',
        'nm_score', 'de_nivel_riesgo', 'fl_pred_modelo', 'de_modelo_version'
    ]
    df = df[columnasFin].copy()

    # VERIFICACIÓN FINAL de integridad
    dupsFin = df.duplicated(subset=['id_maquina_dfos', 'fe_ventana'], keep=False).sum()
    if dupsFin:
        log.error(f"  Duplicados residuales tras limpieza: {dupsFin} — forzando eliminación")
        df = df.drop_duplicates(subset=['id_maquina_dfos', 'fe_ventana'], keep='last')

    log.info(f"  {len(df):,} registros listos | {df['id_maquina_dfos'].nunique()} máquinas "
             f"| {df['fe_ventana'].min()} → {df['fe_ventana'].max()}")
    return df


def guardarPrediccionesEnBd(dfPredicciones, config, versionModelo=None):
    """
    Inserta predicciones en bui_predicciones_hora_dia usando INSERT IGNORE.
    Retorna el número de filas nuevas insertadas.
    """
    log.info("=" * 70)
    log.info("GUARDANDO PREDICCIONES EN BASE DE DATOS")
    log.info("=" * 70)

    if dfPredicciones.empty:
        log.warning("DataFrame vacío, nada que guardar")
        return 0

    if versionModelo is None:
        versionModelo = config.get('model', {}).get('version', 'v1.0')

    dfInsertar = _prepararDataFrame(dfPredicciones, versionModelo)

    # RESUMEN PRE-INSERCIÓN
    distNivel = dfInsertar['de_nivel_riesgo'].value_counts()
    etiquetas = {'critico': 'CRÍTICO', 'moderado': 'MODERADO', 'bajo': 'BAJO', 'normal': 'NORMAL'}
    for nivel, etiqueta in etiquetas.items():
        n = distNivel.get(nivel, 0)
        log.info(f"  {etiqueta}: {n:,} ({n / len(dfInsertar) * 100:.1f}%)")

    nPositivas = dfInsertar['fl_pred_modelo'].sum()
    log.info(f"  Predicciones positivas: {nPositivas:,} ({nPositivas / len(dfInsertar) * 100:.1f}%)")

    engine = _crearEngine(config)
    insertados  = 0
    duplicados  = 0
    sqlInsertar = text(f"""
        INSERT IGNORE INTO {TABLA_DESTINO}
        (id_maquina_dfos, id_linea, fe_ventana, nm_score,
         de_nivel_riesgo, fl_pred_modelo, de_modelo_version)
        VALUES
        (:id_maquina_dfos, :id_linea, :fe_ventana, :nm_score,
         :de_nivel_riesgo, :fl_pred_modelo, :de_modelo_version)
    """)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Conexión a BD verificada")

        totalChunks = (len(dfInsertar) + CHUNK_INSERCION - 1) // CHUNK_INSERCION

        for i in range(0, len(dfInsertar), CHUNK_INSERCION):
            chunk     = dfInsertar.iloc[i:i + CHUNK_INSERCION]
            numChunk  = i // CHUNK_INSERCION + 1

            try:
                with engine.connect() as conn:
                    for _, fila in chunk.iterrows():
                        try:
                            resultado = conn.execute(sqlInsertar, {
                                'id_maquina_dfos':   fila['id_maquina_dfos'],
                                'id_linea':          int(fila['id_linea']),
                                'fe_ventana':        fila['fe_ventana'],
                                'nm_score':          float(fila['nm_score']),
                                'de_nivel_riesgo':   fila['de_nivel_riesgo'],
                                'fl_pred_modelo':    int(fila['fl_pred_modelo']),
                                'de_modelo_version': fila['de_modelo_version'],
                            })
                            if resultado.rowcount > 0:
                                insertados += 1
                            else:
                                duplicados += 1
                        except Exception as e:
                            log.warning(f"  Fila omitida: {e}")
                            duplicados += 1
                    conn.commit()

                if numChunk % 10 == 0 or numChunk == totalChunks:
                    log.info(f"  Chunk {numChunk}/{totalChunks} procesado")

            except Exception as e:
                log.error(f"  Error en chunk {numChunk}: {e}")
                raise

        log.info(f"\n  Insertados: {insertados:,} | Duplicados omitidos: {duplicados:,}")
        return insertados

    except Exception as e:
        log.error(f"Error guardando predicciones: {e}")
        raise
    finally:
        engine.dispose()


def guardarPrediccionesBatch(dfPredicciones, config, versionModelo=None):
    """
    Inserción masiva vía pandas to_sql. Más rápida pero sin control fino de duplicados.
    Si detecta duplicados, hace fallback a guardarPrediccionesEnBd.
    """
    log.info("GUARDANDO PREDICCIONES EN BASE DE DATOS (MODO BATCH)")

    if dfPredicciones.empty:
        log.warning("DataFrame vacío, nada que guardar")
        return 0

    if versionModelo is None:
        versionModelo = config.get('model', {}).get('version', 'v1.0')

    dfInsertar = _prepararDataFrame(dfPredicciones, versionModelo)
    engine = _crearEngine(config)

    try:
        log.info(f"Insertando {len(dfInsertar):,} registros en '{TABLA_DESTINO}'...")
        dfInsertar.to_sql(
            name=TABLA_DESTINO, con=engine,
            if_exists='append', index=False,
            chunksize=CHUNK_INSERCION, method='multi'
        )
        log.info(f"{len(dfInsertar):,} registros insertados")
        return len(dfInsertar)

    except Exception as e:
        if 'Duplicate entry' in str(e):
            log.warning("Duplicados detectados — reintentando con INSERT IGNORE...")
            return guardarPrediccionesEnBd(dfPredicciones, config, versionModelo)
        log.error(f"Error guardando predicciones: {e}")
        raise
    finally:
        engine.dispose()


def main():
    # MODO STANDALONE - reprocesa el CSV de predicciones más reciente en outputs/
    config = _cargarConfig()
    setup_logging(config)

    log.info("SAVE_PREDICTIONS — MODO STANDALONE")

    dirOutputs = Path(__file__).parent / 'outputs'
    archivos   = list(dirOutputs.glob('predictions_*.csv'))

    if not archivos:
        log.error("No se encontraron archivos predictions_*.csv en outputs/")
        return 1

    archivoReciente = max(archivos, key=lambda x: x.stat().st_mtime)
    log.info(f"Cargando: {archivoReciente}")

    dfPred = pd.read_csv(archivoReciente)
    log.info(f"{len(dfPred):,} registros cargados")

    try:
        n = guardarPrediccionesEnBd(dfPred, config)
        log.info(f"Completado: {n:,} predicciones guardadas")
        return 0
    except Exception as e:
        log.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())