"""
db_connector.py - Versión Corregida
===================================
Combina la lógica original (nombres de columnas correctos) 
con la configuración de entornos separada (Source/Target).
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pymysql

log = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Clase para manejar conexiones y operaciones con MySQL
    """
    
    def __init__(self, config):
        self.config = config
        self.engine = None              # Motor de LECTURA (Source/PROD)
        self.predictions_engine = None  # Motor de ESCRITURA (Target/DEV)
    
    # ========================================================================
    # MÉTODOS DE CONEXIÓN
    # ========================================================================
    
    def connect_source(self):
        """Conecta a la base de datos fuente (PROD/Lectura)"""
        try:
            # Accedemos a la sección 'source' del config.yaml
            db_config = self.config['database']['source']
            
            log.info("🔌 Conectando a base de datos fuente (PROD)...")
            
            # Construcción de string de conexión
            connection_string = (
                f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            self.engine = create_engine(
                connection_string,
                connect_args={'ssl': {'fake_flag_to_enable_tls': True}},
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Probar conexión
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            log.info(" Conexión a SOURCE establecida")
            
        except Exception as e:
            log.error(f" Error conectando a SOURCE: {str(e)}")
            raise
    
    def connect_predictions(self):
        """Conecta a la base de datos destino (DEV/Escritura)"""
        try:
            # Accedemos a la sección 'target' del config.yaml
            if 'target' in self.config['database']:
                db_config = self.config['database']['target']
                log.info("🔌 Conectando a base de datos destino (DEV)...")
            else:
                # Fallback si no existe target (mismo que source)
                log.warning(" No hay config 'target', usando source.")
                self.predictions_engine = self.engine
                return

            connection_string = (
                f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            self.predictions_engine = create_engine(
                connection_string,
                connect_args={'ssl': {'fake_flag_to_enable_tls': True}},
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            log.info(" Conexión a TARGET establecida")
            
        except Exception as e:
            log.error(f" Error conectando a TARGET: {str(e)}")
            raise

    # ========================================================================
    # MÉTODOS DE EXTRACCIÓN (Queries Originales)
    # ========================================================================
    
    def extract_events_data(self, lookback_hours=168, fecha_referencia=None):
        """Extrae datos de bui_perdida usando fe_inicio (QUERY ORIGINAL)"""
        if not self.engine:
            self.connect_source()

        try:
            if fecha_referencia is None:
                fecha_fin = datetime.now()
            elif isinstance(fecha_referencia, str):
                fecha_fin = datetime.strptime(fecha_referencia, '%Y-%m-%d')
            else:
                fecha_fin = fecha_referencia
            
            fecha_inicio = fecha_fin - timedelta(hours=lookback_hours)
            
            log.info(f" Extrayendo bui_perdida ({lookback_hours}h)...")
            
            # QUERY ORIGINAL DEL USUARIO
            query = f"""
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
            
            df = pd.read_sql(query, self.engine)
            
            # Convertir fechas
            df['fe_inicio'] = pd.to_datetime(df['fe_inicio'])
            df['fe_fin'] = pd.to_datetime(df['fe_fin'])
            
            log.info(f" bui_perdida: {len(df):,} registros")
            return df
            
        except Exception as e:
            log.error(f" Error extrayendo eventos: {str(e)}")
            raise

    def extract_maintenance_data(self, lookback_days=60, fecha_referencia=None):
        """Extrae datos de bui_pm_ewo usando fe_inicio_averia (QUERY ORIGINAL)"""
        if not self.engine:
            self.connect_source()

        try:
            if fecha_referencia is None:
                fecha_fin = datetime.now()
            elif isinstance(fecha_referencia, str):
                fecha_fin = datetime.strptime(fecha_referencia, '%Y-%m-%d')
            else:
                fecha_fin = fecha_referencia
            
            fecha_inicio = fecha_fin - timedelta(days=lookback_days)
            
            log.info(f" Extrayendo bui_pm_ewo ({lookback_days}d)...")
            
            # QUERY ORIGINAL DEL USUARIO
            query = f"""
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
            WHERE fe_inicio_averia >= '{fecha_inicio.strftime('%Y-%m-%d %H:%M:%S')}'
              AND fe_inicio_averia <= '{fecha_fin.strftime('%Y-%m-%d %H:%M:%S')}'
            ORDER BY id_linea, fe_inicio_averia
            """
            
            df = pd.read_sql(query, self.engine)
            
            df['fe_inicio_averia'] = pd.to_datetime(df['fe_inicio_averia'])
            df['fe_inicio_prod'] = pd.to_datetime(df['fe_inicio_prod'])
            
            log.info("   Calculando estado_ewo...")
            df['estado_ewo'] = df.apply(self._calcular_estado_ewo, axis=1)
            
            log.info(f" bui_pm_ewo: {len(df):,} registros")
            return df
            
        except Exception as e:
            log.error(f" Error extrayendo mantenimientos: {str(e)}")
            raise

    def _calcular_estado_ewo(self, row):
        """Lógica original de estado"""
        if pd.notna(row.get('fe_cerrar')): return 'CERRADA'
        elif pd.notna(row.get('fe_fin_mto')): return 'FINALIZADA_PENDIENTE_CIERRE'
        elif pd.notna(row.get('fe_fin_reparacion')): return 'REPARADA'
        elif pd.notna(row.get('fe_fin_diagnostico')): return 'DIAGNOSTICADA'
        elif pd.notna(row.get('fe_llegada_mto')): return 'EN_ATENCION'
        elif pd.notna(row.get('fe_aviso_mto')): return 'AVISADA'
        else: return 'CREADA_SIN_AVISO'

    def extract_line_metadata(self):
        """Extrae metadata de líneas (QUERY ORIGINAL)"""
        if not self.engine:
            self.connect_source()
        try:
            log.info(" Extrayendo bui_line...")
            query = "SELECT id_linea, id_subcategoria, id_fabrica_area, id_fabrica FROM bui_line"
            df = pd.read_sql(query, self.engine)
            log.info(f" bui_line: {len(df):,} registros")
            return df
        except Exception as e:
            log.error(f" Error extrayendo metadata: {str(e)}")
            raise

    # ========================================================================
    # GUARDADO
    # ========================================================================

    def save_predictions(self, predictions_df, de_modelo_version):
        """Guarda en la base de datos de DESTINO"""
        try:
            if self.predictions_engine is None:
                self.connect_predictions()
            
            table_name = self.config.get('predictions', {}).get('table_name', 'bui_predicciones_hora')
            
            log.info(f" Guardando {len(predictions_df)} predicciones en '{table_name}'...")
            
            # Copia para no alterar el original
            df_to_save = predictions_df.copy()
            df_to_save['de_modelo_version'] = de_modelo_version
            df_to_save['fe_ventana'] = datetime.now()
            
            rows = df_to_save.to_sql(
                name=table_name,
                con=self.predictions_engine,
                if_exists='append',
                index=False,
                chunksize=1000,
                method='multi'
            )
            log.info(" Guardado exitoso.")
            return rows
            
        except Exception as e:
            log.error(f" Error guardando predicciones: {str(e)}")
            raise

    def close_connections(self):
        if self.engine:
            self.engine.dispose()
        if self.predictions_engine and self.predictions_engine != self.engine:
            self.predictions_engine.dispose()

    def __enter__(self):
        self.connect_source()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connections()
