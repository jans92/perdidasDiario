"""
db_connector.py - Conexiones y extracción de datos desde MySQL.
Source (PROD/lectura): bui_perdida, bui_pm_ewo, bui_line.
Target (DEV/escritura): bui_predicciones_hora_dia.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)


class DatabaseConnector:

    def __init__(self, config):
        self.config           = config
        self.engine           = None   # SOURCE - lectura (PROD)
        self.enginePredicciones = None # TARGET - escritura (DEV)

    # CONEXIONES ──────────────────────────────────────────────────────────────

    @staticmethod
    def _crearEngine(dbConfig):
        cadena = (
            f"mysql+pymysql://{dbConfig['user']}:{dbConfig['password']}"
            f"@{dbConfig['host']}:{dbConfig['port']}/{dbConfig['database']}"
        )
        return create_engine(
            cadena,
            connect_args={'ssl': {'fake_flag_to_enable_tls': True}},
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )

    def connect_source(self):
        """Abre conexión a SOURCE (PROD). Llamada automática si no está abierta."""
        try:
            dbConfig = self.config['database']['source']
            log.info("Conectando a SOURCE (PROD)...")
            self.engine = self._crearEngine(dbConfig)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info(f"  SOURCE conectado: {dbConfig['database']}@{dbConfig['host']}")
        except Exception as e:
            log.error(f"Error conectando a SOURCE: {e}")
            raise

    def _conectarTarget(self):
        """Abre conexión a TARGET (DEV). Fallback a SOURCE si no hay config separada."""
        try:
            if 'target' not in self.config['database']:
                log.warning("Sin config 'target' — usando SOURCE como destino")
                self.enginePredicciones = self.engine
                return
            dbConfig = self.config['database']['target']
            log.info("Conectando a TARGET (DEV)...")
            self.enginePredicciones = self._crearEngine(dbConfig)
            log.info(f"  TARGET conectado: {dbConfig['database']}@{dbConfig['host']}")
        except Exception as e:
            log.error(f"Error conectando a TARGET: {e}")
            raise

    # EXTRACCIÓN ──────────────────────────────────────────────────────────────

    def _resolverFechaFin(self, fechaReferencia):
        """Convierte fechaReferencia (str, date o None) a datetime."""
        if fechaReferencia is None:
            return datetime.now()
        if isinstance(fechaReferencia, str):
            return datetime.strptime(fechaReferencia, '%Y-%m-%d')
        return fechaReferencia

    def extract_events_data(self, lookback_hours: int = 168, fecha_referencia=None) -> pd.DataFrame:
        """
        Extrae bui_perdida en el rango [fechaFin - lookback_hours, fechaFin].
        Incluye duración calculada en minutos.
        """
        if not self.engine:
            self.connect_source()

        fechaFin   = self._resolverFechaFin(fecha_referencia)
        fechaInicio = fechaFin - timedelta(hours=lookback_hours)
        log.info(f"Extrayendo bui_perdida ({lookback_hours}h): {fechaInicio} → {fechaFin}")

        query = f"""
            SELECT
                id_perdida, id_linea, id_fabrica, id_maquina_dfos,
                de_perdida_1, de_perdida_2, de_perdida_3,
                fe_inicio, fe_fin,
                TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin) AS duracion_minutos
            FROM bui_perdida
            WHERE fe_inicio BETWEEN '{fechaInicio:%Y-%m-%d %H:%M:%S}'
                                 AND '{fechaFin:%Y-%m-%d %H:%M:%S}'
            ORDER BY id_maquina_dfos, fe_inicio
        """
        try:
            df = pd.read_sql(query, self.engine)
            df['fe_inicio'] = pd.to_datetime(df['fe_inicio'])
            df['fe_fin']    = pd.to_datetime(df['fe_fin'])
            log.info(f"  bui_perdida: {len(df):,} registros")
            return df
        except Exception as e:
            log.error(f"Error extrayendo bui_perdida: {e}")
            raise

    def extract_maintenance_data(self, lookback_days: int = 60, fecha_referencia=None) -> pd.DataFrame:
        """
        Extrae bui_pm_ewo en el rango [fechaFin - lookback_days, fechaFin].
        Calcula duracion_reparacion_min y estado_ewo en la propia query/post-proceso.
        """
        if not self.engine:
            self.connect_source()

        fechaFin    = self._resolverFechaFin(fecha_referencia)
        fechaInicio = fechaFin - timedelta(days=lookback_days)
        log.info(f"Extrayendo bui_pm_ewo ({lookback_days}d): {fechaInicio} → {fechaFin}")

        query = f"""
            SELECT
                id_ewo, id_linea, id_perdida, fl_mantenimiento,
                fe_inicio_averia, fe_inicio_prod, fl_borrador,
                fe_cerrar, fe_fin_mto, fe_fin_reparacion,
                fe_fin_diagnostico, fe_llegada_mto, fe_aviso_mto,
                TIMESTAMPDIFF(MINUTE, fe_inicio_averia, fe_inicio_prod) AS duracion_reparacion_min
            FROM bui_pm_ewo
            WHERE fe_inicio_averia BETWEEN '{fechaInicio:%Y-%m-%d %H:%M:%S}'
                                       AND '{fechaFin:%Y-%m-%d %H:%M:%S}'
            ORDER BY id_linea, fe_inicio_averia
        """
        try:
            df = pd.read_sql(query, self.engine)
            df['fe_inicio_averia'] = pd.to_datetime(df['fe_inicio_averia'])
            df['fe_inicio_prod']   = pd.to_datetime(df['fe_inicio_prod'])
            df['estado_ewo']       = df.apply(self._estadoEwo, axis=1)
            log.info(f"  bui_pm_ewo: {len(df):,} registros")
            return df
        except Exception as e:
            log.error(f"Error extrayendo bui_pm_ewo: {e}")
            raise

    @staticmethod
    def _estadoEwo(row) -> str:
        # ESTADO EWO - derivado de las fechas de ciclo de vida de la orden
        if pd.notna(row.get('fe_cerrar')):          return 'CERRADA'
        if pd.notna(row.get('fe_fin_mto')):         return 'FINALIZADA_PENDIENTE_CIERRE'
        if pd.notna(row.get('fe_fin_reparacion')):  return 'REPARADA'
        if pd.notna(row.get('fe_fin_diagnostico')): return 'DIAGNOSTICADA'
        if pd.notna(row.get('fe_llegada_mto')):     return 'EN_ATENCION'
        if pd.notna(row.get('fe_aviso_mto')):       return 'AVISADA'
        return 'CREADA_SIN_AVISO'

    def extract_line_metadata(self) -> pd.DataFrame:
        """Extrae id_linea, id_subcategoria, id_fabrica_area e id_fabrica de bui_line."""
        if not self.engine:
            self.connect_source()
        try:
            df = pd.read_sql(
                "SELECT id_linea, id_subcategoria, id_fabrica_area, id_fabrica FROM bui_line",
                self.engine
            )
            log.info(f"  bui_line: {len(df):,} líneas")
            return df
        except Exception as e:
            log.error(f"Error extrayendo bui_line: {e}")
            raise

    # ESCRITURA ───────────────────────────────────────────────────────────────

    def save_predictions(self, dfPredicciones: pd.DataFrame, versionModelo: str) -> int:
        """Inserta predicciones en la tabla destino. Retorna número de filas escritas."""
        if self.enginePredicciones is None:
            self._conectarTarget()

        tabla = self.config.get('predictions', {}).get('table_name', 'bui_predicciones_hora_dia')
        log.info(f"Guardando {len(dfPredicciones):,} predicciones en '{tabla}'...")

        try:
            df = dfPredicciones.copy()
            df['de_modelo_version'] = versionModelo
            df['fe_ventana']        = datetime.now()

            filas = df.to_sql(
                name=tabla, con=self.enginePredicciones,
                if_exists='append', index=False,
                chunksize=1000, method='multi'
            )
            log.info("Guardado exitoso")
            return filas
        except Exception as e:
            log.error(f"Error guardando predicciones: {e}")
            raise

    # CICLO DE VIDA ───────────────────────────────────────────────────────────

    def close_connections(self):
        if self.engine:
            self.engine.dispose()
        if self.enginePredicciones and self.enginePredicciones is not self.engine:
            self.enginePredicciones.dispose()

    def __enter__(self):
        self.connect_source()
        return self

    def __exit__(self, *_):
        self.close_connections()