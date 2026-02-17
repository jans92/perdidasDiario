"""
Feature Engineering - PARTE 1: Funciones Auxiliares
Sistema de Mantenimiento Predictivo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import gc
import logging



log = logging.getLogger(__name__)

# ============================================================================
# FUNCIONES AUXILIARES PARA PIVOTEO
# ============================================================================

def crear_features_ventana(df_ventana):
    """
    Agrega eventos de una ventana de 1 hora en features
    Basado exactamente en el script de entrenamiento
    
    Args:
        df_ventana: DataFrame con eventos de 1 hora para 1 máquina
        
    Returns:
        dict con features calculadas
    """
    features = {}

    # ========== FEATURES DE ESTADO MACRO (de_perdida_1) ==========
    for estado in ['RUN', 'MPL', 'UCL', 'PDL', 'DESCONOCIDO']:
        mask = df_ventana['de_perdida_1'] == estado
        features[f'{estado.lower()}_time'] = df_ventana.loc[mask, 'duracion_minutos'].sum()
        features[f'{estado.lower()}_count'] = mask.sum()

    # ========== FEATURES DE EVENTOS NIVEL 2 (de_perdida_2) ==========
    eventos_nivel2 = [
        'Minor Stoppages',
        'Speed Loss',
        'Measurement & Adjustment',
        'Process Failure Time',
        'Idle Time',
        'Material Availability at Line-Side Loss',
        'Quality Defect Time Loss',
        'Maintenance Time',
        'Cleaning & Sanitation Time',
        'Changeover Time',
        'Cutting Blade Change',
        'Breakdown & Equipment Failure Time',
        'Preparatory & Close Out Time Losses',
        'Planned Stoppage Time'
    ]

    for evento in eventos_nivel2:
        mask = df_ventana['de_perdida_2'] == evento
        # Nombre de columna limpio
        col_name = evento.lower().replace(' ', '_').replace('&', 'and').replace('/', '_')
        features[f'{col_name}_duration'] = df_ventana.loc[mask, 'duracion_minutos'].sum()
        features[f'{col_name}_count'] = mask.sum()

    # ========== FEATURES DE BREAKDOWN DETALLADO (de_perdida_3) ==========
    breakdowns_tipo = {
        'breakdown_mechanical': 'Breakdown - Mechanical',
        'breakdown_electrical': 'Breakdown - Electrical',
        'breakdown_ic': 'Breakdown - Instrumentation & Control'
    }

    for col_name, evento in breakdowns_tipo.items():
        mask = df_ventana['de_perdida_3'] == evento
        features[f'{col_name}_count'] = mask.sum()
        features[f'{col_name}_duration'] = df_ventana.loc[mask, 'duracion_minutos'].sum()

    # Otros eventos importantes de nivel 3
    otros_eventos_nivel3 = [
        'Adjustment',
        'Speed Loss',
        'Minor Stoppages',
        'Idle Time',
        'Product Changeover',
        'Changing Supplies',
        'Changing Cutting Elements',
        'Lack of Product or WIP from Previous Process',
        'Measurement',
        'Human Error',
        'Material Issue'
    ]

    for evento in otros_eventos_nivel3:
        mask = df_ventana['de_perdida_3'] == evento
        col_name = evento.lower().replace(' ', '_').replace('/', '_')
        features[f'{col_name}_nivel3_count'] = mask.sum()
        features[f'{col_name}_nivel3_duration'] = df_ventana.loc[mask, 'duracion_minutos'].sum()

    # ========== FEATURES ESTADÍSTICAS GENERALES ==========
    features['total_events'] = len(df_ventana)
    features['total_duration'] = df_ventana['duracion_minutos'].sum()
    features['avg_event_duration'] = df_ventana['duracion_minutos'].mean() if len(df_ventana) > 0 else 0
    features['max_event_duration'] = df_ventana['duracion_minutos'].max() if len(df_ventana) > 0 else 0

    return features


def pivotar_eventos_por_maquina_hora(perdida_clean):
    """
    Pivotea eventos por máquina y ventana horaria
    
    Args:
        perdida_clean: DataFrame con datos de bui_perdida
        
    Returns:
        DataFrame pivoteado con features por ventana horaria
    """
    log.info("\n Pivoteando eventos por máquina/hora...")
    
    # Crear ventana horaria para cada registro
    perdida_clean['timestamp_hora'] = perdida_clean['fe_inicio'].dt.floor('H')

    # Lista de todas las máquinas
    maquinas_unicas = perdida_clean['id_maquina_dfos'].unique()
    log.info(f"Total máquinas a procesar: {len(maquinas_unicas)}")

    # Procesar por chunks de máquinas
    chunk_size = 100
    chunks_procesados = []

    for i in tqdm(range(0, len(maquinas_unicas), chunk_size), desc="Chunks de máquinas"):
        chunk_maquinas = maquinas_unicas[i:i+chunk_size]

        # Filtrar datos del chunk
        df_chunk = perdida_clean[perdida_clean['id_maquina_dfos'].isin(chunk_maquinas)].copy()

        # Agrupar por máquina y ventana horaria
        resultados_chunk = []

        for maquina in chunk_maquinas:
            df_maquina = df_chunk[df_chunk['id_maquina_dfos'] == maquina]

            if len(df_maquina) == 0:
                continue

            # Obtener info de contexto (línea, fábrica)
            id_linea = df_maquina['id_linea'].iloc[0]
            id_fabrica = df_maquina['id_fabrica'].iloc[0]

            # Agrupar por ventana horaria
            for timestamp_hora, df_ventana in df_maquina.groupby('timestamp_hora'):
                features = crear_features_ventana(df_ventana)

                # Agregar identificadores
                features['id_maquina_dfos'] = maquina
                features['id_linea'] = id_linea
                features['id_fabrica'] = id_fabrica
                features['timestamp_hora'] = timestamp_hora

                resultados_chunk.append(features)

        # Convertir a DataFrame
        df_chunk_pivot = pd.DataFrame(resultados_chunk)
        chunks_procesados.append(df_chunk_pivot)

        # Liberar memoria
        del df_chunk, df_maquina, resultados_chunk
        gc.collect()

    # Concatenar todos los chunks
    log.info(" Concatenando resultados...")
    df_pivoteado = pd.concat(chunks_procesados, ignore_index=True)

    # Llenar NaN con 0
    df_pivoteado = df_pivoteado.fillna(0)

    log.info(f" Pivoteo completado: {df_pivoteado.shape}")
    
    # Liberar memoria
    del chunks_procesados
    gc.collect()
    
    return df_pivoteado


# ============================================================================
# FUNCIONES PARA FEATURES DE MANTENIMIENTO
# ============================================================================

def calcular_features_mantenimiento(df_pivot, ewos_df, tipo_mant, ventanas_temporales):
    """
    Calcula features de mantenimiento para cada ventana
    
    Args:
        df_pivot: DataFrame pivoteado
        ewos_df: DataFrame de EWOs (preventivos o correctivos)
        tipo_mant: 'pm' o 'cm'
        ventanas_temporales: dict con ventanas (ej: {'24h': 24, '7d': 168})
    """
    log.info(f"\n Procesando mantenimientos {tipo_mant.upper()}...")

    # Preparar EWOs con timestamp
    ewos_df = ewos_df.copy()
    ewos_df['timestamp_hora'] = pd.to_datetime(ewos_df['fe_inicio_averia']).dt.floor('H')

    # Inicializar columnas con 0
    for nombre_ventana in ventanas_temporales.keys():
        df_pivot[f'{tipo_mant}_count_{nombre_ventana}'] = 0
        df_pivot[f'{tipo_mant}_duration_total_{nombre_ventana}'] = 0.0

        if nombre_ventana == '24h':
            df_pivot[f'{tipo_mant}_duration_max_{nombre_ventana}'] = 0.0

    df_pivot[f'horas_desde_ultimo_{tipo_mant}'] = 9999.0
    df_pivot[f'tiene_historial_{tipo_mant}'] = 0

    # Agrupar por línea
    lineas_unicas = df_pivot['id_linea'].unique()

    for linea in tqdm(lineas_unicas, desc=f"Líneas {tipo_mant}"):
        # Filtrar EWOs de esta línea
        ewos_linea = ewos_df[ewos_df['id_linea'] == linea].sort_values('timestamp_hora')

        if len(ewos_linea) == 0:
            continue

        # Línea con historial
        mask_linea = df_pivot['id_linea'] == linea
        df_pivot.loc[mask_linea, f'tiene_historial_{tipo_mant}'] = 1

        # Para cada ventana temporal en el dataset
        for idx in df_pivot[mask_linea].index:
            timestamp_actual = df_pivot.loc[idx, 'timestamp_hora']

            # Calcular features para cada ventana temporal
            for nombre_ventana, horas in ventanas_temporales.items():
                timestamp_inicio = timestamp_actual - pd.Timedelta(hours=horas)

                # Filtrar EWOs en ventana
                ewos_ventana = ewos_linea[
                    (ewos_linea['timestamp_hora'] >= timestamp_inicio) &
                    (ewos_linea['timestamp_hora'] < timestamp_actual)
                ]

                # Count y duration
                df_pivot.loc[idx, f'{tipo_mant}_count_{nombre_ventana}'] = len(ewos_ventana)
                df_pivot.loc[idx, f'{tipo_mant}_duration_total_{nombre_ventana}'] = ewos_ventana['duracion_reparacion_min'].sum()

                # Duration max solo para 24h
                if nombre_ventana == '24h' and len(ewos_ventana) > 0:
                    df_pivot.loc[idx, f'{tipo_mant}_duration_max_{nombre_ventana}'] = ewos_ventana['duracion_reparacion_min'].max()

            # Tiempo desde último mantenimiento
            ewos_previos = ewos_linea[ewos_linea['timestamp_hora'] < timestamp_actual]
            if len(ewos_previos) > 0:
                ultimo_mant = ewos_previos['timestamp_hora'].max()
                horas_desde = (timestamp_actual - ultimo_mant).total_seconds() / 3600
                df_pivot.loc[idx, f'horas_desde_ultimo_{tipo_mant}'] = horas_desde


def agregar_features_mantenimiento(df_pivoteado, bui_pm_ewo):
    """
    Agrega todas las features de mantenimiento al dataset pivoteado
    ...
    """
    log.info("\n Agregando features de mantenimiento...")
    
    #  AGREGAR: Verificar que estado_ewo existe, si no, calcularlo
    if 'estado_ewo' not in bui_pm_ewo.columns:
        log.info(" estado_ewo no encontrado, calculando...")
        
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
        log.info(f" estado_ewo calculado")
    
    # Calcular duración de reparación
    bui_pm_ewo['duracion_reparacion_min'] = (
        (pd.to_datetime(bui_pm_ewo['fe_inicio_prod']) - pd.to_datetime(bui_pm_ewo['fe_inicio_averia']))
        .dt.total_seconds() / 60
    )

    # Filtrar EWOs válidas (ahora estado_ewo existe)
    ewos_para_features = bui_pm_ewo[
        (bui_pm_ewo['fl_borrador'] == 0) &
        (bui_pm_ewo['estado_ewo'].isin(['CERRADA', 'FINALIZADA_PENDIENTE_CIERRE'])) &
        (bui_pm_ewo['duracion_reparacion_min'].notna()) &
        (bui_pm_ewo['duracion_reparacion_min'] >= 0) &
        (bui_pm_ewo['duracion_reparacion_min'] <= 4320)  # Max 3 días
    ].copy()

    log.info(f" EWOs válidas: {len(ewos_para_features):,}")

    # Separar preventivos y correctivos
    ewos_preventivos = ewos_para_features[ewos_para_features['fl_mantenimiento'] == 1].copy()
    ewos_correctivos = ewos_para_features[ewos_para_features['fl_mantenimiento'] == 0].copy()

    log.info(f"   - Preventivos: {len(ewos_preventivos):,}")
    log.info(f"   - Correctivos: {len(ewos_correctivos):,}")

    # Definir ventanas temporales
    ventanas = {
        '24h': 24,
        '7d': 24 * 7,
        '30d': 24 * 30
    }

    # Calcular features para preventivos
    calcular_features_mantenimiento(df_pivoteado, ewos_preventivos, 'pm', ventanas)

    # Calcular features para correctivos
    calcular_features_mantenimiento(df_pivoteado, ewos_correctivos, 'cm', ventanas)

    # Llenar NaN con 0
    df_pivoteado = df_pivoteado.fillna(0)
    
    log.info(f" Features de mantenimiento agregadas")
    
    return df_pivoteado


# ============================================================================
# FUNCIONES PARA ENRIQUECIMIENTO CON METADATA
# ============================================================================

def enriquecer_con_metadata(df_pivoteado, bui_line):
    """
    Enriquece el dataset con metadata de líneas
    
    Args:
        df_pivoteado: DataFrame pivoteado
        bui_line: DataFrame con metadata de líneas
        
    Returns:
        DataFrame enriquecido
    """
    log.info("\n Enriqueciendo con metadata de líneas...")
    
    # Sincronizar tipos de datos
    df_pivoteado['id_linea'] = df_pivoteado['id_linea'].astype('int64')
    bui_line['id_linea'] = bui_line['id_linea'].astype('int64')
    
    # Merge
    df_pivoteado = df_pivoteado.merge(
        bui_line[['id_linea', 'id_subcategoria', 'id_fabrica_area']],
        on='id_linea',
        how='left',
        validate='m:1'
    )
    
    # Imputar valores faltantes
    df_pivoteado['id_subcategoria'] = df_pivoteado['id_subcategoria'].fillna(0).astype('int32')
    df_pivoteado['id_fabrica_area'] = df_pivoteado['id_fabrica_area'].fillna(0).astype('int32')
    
    log.info(f" Metadata agregada")
    
    return df_pivoteado


# ============================================================================
# FUNCIONES PARA LIMPIEZA Y VALIDACIÓN
# ============================================================================

def limpiar_datos_anomalos(df):
    """
    Limpia valores anómalos en el dataset
    """
    log.info("\n Limpiando datos anómalos...")
    
    # Clipear valores fuera de rango razonable
    df['run_time'] = df['run_time'].clip(lower=0, upper=120)
    df['total_duration'] = df['total_duration'].clip(lower=0.1, upper=120)
    
    # Si total_duration es 0, usar 60 como default
    df.loc[df['total_duration'] == 0, 'total_duration'] = 60.0
    
    log.info(" Datos limpios")
    
    return df


"""
Feature Engineering - PARTE 2: Clase Principal
Sistema de Mantenimiento Predictivo
"""

from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CLASE PRINCIPAL: FEATURE ENGINEER
# ============================================================================

class FeatureEngineer:
    """
    Clase para generar las 63 features del modelo de mantenimiento predictivo
    Basada exactamente en los scripts de entrenamiento
    """
    
    def __init__(self, config):
        """
        Inicializa el Feature Engineer
        
        Args:
            config: dict con configuración del sistema
        """
        self.config = config
        self.label_encoders = {}
        
        log.info(" Feature Engineer inicializado")
    
    
    def transform(self, bui_perdida, bui_pm_ewo, bui_line):
        """
        Método principal: transforma datos raw en features listos para predicción
        
        Args:
            bui_perdida: DataFrame con eventos de pérdidas
            bui_pm_ewo: DataFrame con mantenimientos
            bui_line: DataFrame con metadata de líneas
            
        Returns:
            DataFrame con todas las features calculadas
        """
        log.info("=" * 80)
        log.info(" INICIANDO FEATURE ENGINEERING")
        log.info("=" * 80)
        
        # PASO 1: Pivotar eventos por máquina/hora
        df = pivotar_eventos_por_maquina_hora(bui_perdida)
        
        # PASO 2: Agregar features de mantenimiento
        df = agregar_features_mantenimiento(df, bui_pm_ewo)
        
        # PASO 3: Enriquecer con metadata
        df = enriquecer_con_metadata(df, bui_line)
        
        # PASO 4: Limpieza inicial
        df = limpiar_datos_anomalos(df)
        
        # PASO 5: Features detalladas por evento
        df = self._crear_features_eventos_detallados(df)
        
        # PASO 6: Métricas agregadas
        df = self._crear_metricas_agregadas(df)
        
        # PASO 7: Features de mantenimiento mejoradas
        df = self._crear_features_mantenimiento_mejoradas(df)
        
        # PASO 8: Rolling windows
        df = self._crear_rolling_windows(df)
        
        # PASO 9: Tendencias
        df = self._crear_tendencias(df)
        
        # PASO 10: Ratios
        df = self._crear_ratios(df)
        
        # PASO 11: Features temporales
        df = self._crear_features_temporales(df)
        
        # PASO 12: Variabilidad
        df = self._crear_variabilidad(df)
        
        # PASO 13: Encoding de categorías
        df = self._encoding_categorias(df)
        
        # PASO 14: Features de anomalías
        df = self._crear_features_anomalias(df)
        
        # PASO 15: Comparación con pares
        df = self._comparacion_pares(df)
        
        # PASO 16: Limpieza final
        df = self._limpieza_final(df)
        
        # PASO 17: Filtrar solo últimas 24h (para predicción del día siguiente)
        df = self._filtrar_ventanas_prediccion(df)
        
        log.info("\n" + "=" * 80)
        log.info(" FEATURE ENGINEERING COMPLETADO")
        log.info("=" * 80)
        log.info(f"Shape final: {df.shape}")
        
        return df
    
    
    def _crear_features_eventos_detallados(self, df):
        """
        PASO 5: Features detalladas por evento
        """
        log.info("\n Paso 5: Features detalladas por evento...")
        
        eventos_detallados = {
            'speed_loss': ('speed_loss_duration', 'speed_loss_count'),
            'minor_stoppages': ('minor_stoppages_duration', 'minor_stoppages_count'),
            'process_failure': ('process_failure_time_duration', 'process_failure_time_count'),
            'measurement_adjustment': ('measurement_and_adjustment_duration', 'measurement_and_adjustment_count'),
            'quality_defects': ('quality_defect_time_loss_duration', 'quality_defect_time_loss_count'),
            'material_loss': ('material_availability_at_line-side_loss_duration', 'material_availability_at_line-side_loss_count'),
            'idle_time': ('idle_time_duration', 'idle_time_count'),
            'breakdown_mechanical': ('breakdown_mechanical_duration', 'breakdown_mechanical_count'),
            'breakdown_electrical': ('breakdown_electrical_duration', 'breakdown_electrical_count'),
            'breakdown_ic': ('breakdown_ic_duration', 'breakdown_ic_count')
        }
        
        for nombre_evento, (col_duracion, col_count) in eventos_detallados.items():
            # Duración promedio por evento
            df[f'{nombre_evento}_duracion_avg'] = np.where(
                df[col_count] > 0,
                df[col_duracion] / df[col_count],
                0
            )
            
            # Porcentaje del tiempo total
            df[f'{nombre_evento}_tiempo_pct'] = (
                df[col_duracion] / df['total_duration'] * 100
            ).clip(upper=100)
            
            # Frecuencia por hora
            df[f'{nombre_evento}_frecuencia_por_hora'] = np.where(
                df['total_duration'] > 0,
                df[col_count] / (df['total_duration'] / 60),
                0
            )
            
            # Severity score
            df[f'{nombre_evento}_severity_score'] = (
                df[col_count] * df[f'{nombre_evento}_duracion_avg']
            )
        
        # Feature especial para minor_stoppages
        df['minor_stoppages_freq_normalizada'] = np.where(
            df['run_time'] > 0,
            df['minor_stoppages_count'] / (df['run_time'] / 60),
            0
        )
        
        log.info(f" {len(eventos_detallados) * 4 + 1} features creadas")
        return df
    
    
    def _crear_metricas_agregadas(self, df):
        """
        PASO 6: Métricas agregadas
        """
        log.info("\n Paso 6: Métricas agregadas...")
        
        # Total de pérdidas
        perdidas_cols = [col for col in df.columns if '_duration' in col and
                         any(x in col for x in ['loss', 'defect', 'failure', 'stoppage', 'idle'])]
        
        df['total_perdidas_duracion'] = df[perdidas_cols].sum(axis=1)
        df['total_perdidas_tiempo_pct'] = (
            df['total_perdidas_duracion'] / df['total_duration'] * 100
        ).clip(upper=100)
        
        # Diversidad de eventos
        count_cols = [col for col in df.columns if col.endswith('_count') and
                      not col.startswith(('run', 'mpl', 'ucl', 'pdl', 'desconocido', 'total'))]
        
        df['diversidad_eventos'] = (df[count_cols] > 0).sum(axis=1)
        
        # Evento más largo
        duracion_cols = [col for col in df.columns if col.endswith('_duration') and
                         not col.startswith(('total', 'run', 'mpl', 'ucl', 'pdl'))]
        
        df['evento_mas_largo_duracion'] = df[duracion_cols].max(axis=1)
        df['evento_mas_frecuente_count'] = df[count_cols].max(axis=1)
        
        # Concentración en eventos críticos
        eventos_criticos_dur = [
            'breakdown_mechanical_duration',
            'breakdown_electrical_duration',
            'breakdown_ic_duration',
            'process_failure_time_duration'
        ]
        df['duracion_eventos_criticos'] = df[eventos_criticos_dur].sum(axis=1)
        df['ratio_eventos_criticos_total'] = np.where(
            df['total_perdidas_duracion'] > 0,
            df['duracion_eventos_criticos'] / df['total_perdidas_duracion'],
            0
        )
        
        log.info(" 7 features creadas")
        return df
    
    
    def _crear_features_mantenimiento_mejoradas(self, df):
        """
        PASO 7: Features de mantenimiento mejoradas
        """
        log.info("\n Paso 7: Features de mantenimiento mejoradas...")
        
        # Convertir horas a días
        df['dias_desde_ultimo_pm'] = df['horas_desde_ultimo_pm'] / 24
        df['dias_desde_ultimo_cm'] = df['horas_desde_ultimo_cm'] / 24
        
        # Frecuencias mensuales
        df['pm_frecuencia_mensual'] = df['pm_count_30d'] / 30
        df['cm_tasa_fallas_mensual'] = df['cm_count_30d'] / 30
        
        # Duración total últimos 7 días
        df['mantenimiento_total_duracion_7d'] = (
            df['pm_duration_total_7d'] + df['cm_duration_total_7d']
        )
        
        log.info(" 5 features creadas")
        return df
    
    
    def _crear_rolling_windows(self, df):
        """
        PASO 8: Rolling windows (2h, 6h, 12h, 24h)
        """
        log.info("\n Paso 8: Rolling windows...")
        
        # Ordenar por máquina y tiempo
        df = df.sort_values(['id_maquina_dfos', 'timestamp_hora']).reset_index(drop=True)
        
        eventos_rolling = [
            'minor_stoppages_count',
            'speed_loss_duration',
            'process_failure_time_count',
            'measurement_and_adjustment_count',
            'breakdown_and_equipment_failure_time_count',
            'quality_defect_time_loss_duration'
        ]
        
        ventanas = {
            '2h': 2,
            '6h': 6,
            '12h': 12,
            '24h': 24
        }
        
        for nombre_ventana, horas in ventanas.items():
            for evento in eventos_rolling:
                # Nombre corto
                evento_corto = evento.replace('_duration', '').replace('_count', '').replace('_time', '').replace('_and', '')
                evento_corto = evento_corto.replace('__', '_')
                
                # Rolling sum
                col_rolling = f'{evento_corto}_rolling_{nombre_ventana}'
                df[col_rolling] = df.groupby('id_maquina_dfos')[evento].transform(
                    lambda x: x.rolling(window=horas, min_periods=1).sum()
                )
                
                # Rolling mean
                col_avg = f'{evento_corto}_rolling_{nombre_ventana}_avg'
                df[col_avg] = df[col_rolling] / horas
            
            gc.collect()
        
        log.info(f" {len(eventos_rolling) * len(ventanas) * 2} features creadas")
        return df
    
    
    def _crear_tendencias(self, df):
        """
        PASO 9: Tendencias y aceleraciones
        """
        log.info("\n Paso 9: Tendencias...")
        
        eventos_tendencia = [
            'minor_stoppages',
            'speed_loss',
            'process_failure',
            'breakdown_equipment_failure'
        ]
        
        for evento in eventos_tendencia:
            col_12h = f'{evento}_rolling_12h'
            
            if col_12h not in df.columns:
                log.warning(f" Columna {col_12h} no encontrada")
                continue
            
            # Rolling 12h previo
            col_12h_prev = df.groupby('id_maquina_dfos')[col_12h].shift(12)
            
            # Tendencia
            df[f'{evento}_trend_24h'] = df[col_12h] - col_12h_prev.fillna(0)
            
            # Aceleración
            df[f'{evento}_accel_24h'] = np.where(
                col_12h_prev > 0,
                ((df[col_12h] - col_12h_prev) / col_12h_prev * 100),
                0
            ).clip(-500, 500)
        
        log.info(f" {len(eventos_tendencia) * 2} features creadas")
        return df
    
    
    def _crear_ratios(self, df):
        """
        PASO 10: Ratios y proporciones
        """
        log.info("\n Paso 10: Ratios...")
        
        # Eficiencia operacional
        tiempo_total_operativo = df['run_time'] + df['mpl_time'] + df['ucl_time'] + df['pdl_time']
        tiempo_total_operativo = tiempo_total_operativo.clip(lower=0.1)
        
        df['ratio_run_total'] = (df['run_time'] / tiempo_total_operativo * 100).clip(0, 100)
        df['ratio_mpl_total'] = (df['mpl_time'] / tiempo_total_operativo * 100).clip(0, 100)
        df['ratio_ucl_total'] = (df['ucl_time'] / tiempo_total_operativo * 100).clip(0, 100)
        df['ratio_unplanned_total'] = ((df['mpl_time'] + df['ucl_time']) / tiempo_total_operativo * 100).clip(0, 100)
        
        # Ratios de eventos
        df['ratio_minor_vs_critical'] = np.where(
            df['breakdown_and_equipment_failure_time_count'] > 0,
            df['minor_stoppages_count'] / df['breakdown_and_equipment_failure_time_count'],
            0
        )
        
        df['ratio_quality_vs_performance'] = np.where(
            df['speed_loss_duration'] > 0,
            df['quality_defect_time_loss_duration'] / df['speed_loss_duration'],
            0
        )
        
        # Intensidad
        df['events_per_hour'] = df['total_events'] / (df['total_duration'] / 60)
        df['avg_duration_per_event'] = np.where(
            df['total_events'] > 0,
            df['total_duration'] / df['total_events'],
            0
        )
        
        log.info(" 8 features creadas")
        return df
    
    
    def _crear_features_temporales(self, df):
        """
        PASO 11: Features temporales
        """
        log.info("\n Paso 11: Features temporales...")
        
        # Componentes temporales
        df['hora_del_dia'] = df['timestamp_hora'].dt.hour
        df['dia_semana'] = df['timestamp_hora'].dt.dayofweek
        df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
        
        # Turnos
        df['es_turno_noche'] = ((df['hora_del_dia'] >= 22) | (df['hora_del_dia'] < 6)).astype(int)
        df['es_turno_tarde'] = ((df['hora_del_dia'] >= 14) & (df['hora_del_dia'] < 22)).astype(int)
        df['es_turno_mañana'] = ((df['hora_del_dia'] >= 6) & (df['hora_del_dia'] < 14)).astype(int)
        
        # Inicio y fin de turno
        df['es_inicio_turno'] = df['hora_del_dia'].isin([6, 14, 22]).astype(int)
        df['es_fin_turno'] = df['hora_del_dia'].isin([5, 13, 21]).astype(int)
        
        # Encoding circular
        df['hora_sin'] = np.sin(2 * np.pi * df['hora_del_dia'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora_del_dia'] / 24)
        df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        
        log.info(" 13 features creadas")
        return df
    
    
    def _crear_variabilidad(self, df):
        """
        PASO 12: Variabilidad y estabilidad (7 días)
        """
        log.info("\n Paso 12: Variabilidad...")
        
        eventos_variabilidad = [
            'minor_stoppages_count',
            'speed_loss_duration',
            'breakdown_and_equipment_failure_time_count'
        ]
        
        for evento in eventos_variabilidad:
            evento_corto = evento.replace('_duration', '').replace('_count', '').replace('_time', '').replace('_and', '')
            evento_corto = evento_corto.replace('__', '_')
            
            # Desviación estándar 7 días
            df[f'{evento_corto}_std_7d'] = df.groupby('id_maquina_dfos')[evento].transform(
                lambda x: x.rolling(window=168, min_periods=24).std()
            ).fillna(0)
            
            # Coeficiente de variación
            rolling_mean = df.groupby('id_maquina_dfos')[evento].transform(
                lambda x: x.rolling(window=168, min_periods=24).mean()
            )
            df[f'{evento_corto}_cv_7d'] = np.where(
                rolling_mean > 0,
                (df[f'{evento_corto}_std_7d'] / rolling_mean),
                0
            ).clip(0, 10)
        
        # Días desde último breakdown
        log.info("Calculando días desde último breakdown...")
        df['horas_desde_ultimo_breakdown'] = np.nan
        mask_breakdown = df['breakdown_and_equipment_failure_time_count'] > 0
        
        for maquina in tqdm(df['id_maquina_dfos'].unique(), desc="Máquinas"):
            mask_maquina = df['id_maquina_dfos'] == maquina
            timestamps_breakdown = df.loc[mask_maquina & mask_breakdown, 'timestamp_hora'].values
            
            if len(timestamps_breakdown) == 0:
                df.loc[mask_maquina, 'horas_desde_ultimo_breakdown'] = 9999
                continue
            
            for idx in df[mask_maquina].index:
                timestamp_actual = df.loc[idx, 'timestamp_hora']
                previos = timestamps_breakdown[timestamps_breakdown < timestamp_actual]
                
                if len(previos) > 0:
                    ultimo = previos.max()
                    horas = (timestamp_actual - ultimo).total_seconds() / 3600
                    df.loc[idx, 'horas_desde_ultimo_breakdown'] = horas
                else:
                    df.loc[idx, 'horas_desde_ultimo_breakdown'] = 9999
        
        df['horas_desde_ultimo_breakdown'].fillna(9999, inplace=True)
        df['dias_desde_ultimo_breakdown'] = df['horas_desde_ultimo_breakdown'] / 24
        
        log.info(f" {len(eventos_variabilidad) * 2 + 2} features creadas")
        return df
    
    
    def _encoding_categorias(self, df):
        """
        PASO 13: Encoding de categorías
        """
        log.info("\n Paso 13: Encoding de categorías...")
        
        # One-hot encoding para id_subcategoria
        subcategoria_dummies = pd.get_dummies(df['id_subcategoria'], prefix='subcategoria', dtype=int)
        df = pd.concat([df, subcategoria_dummies], axis=1)
        
        # Label encoding para id_fabrica_area y id_fabrica
        le_area = LabelEncoder()
        le_fabrica = LabelEncoder()
        
        df['id_fabrica_area_encoded'] = le_area.fit_transform(df['id_fabrica_area'].astype(str))
        df['id_fabrica_encoded'] = le_fabrica.fit_transform(df['id_fabrica'].astype(str))
        
        # Guardar encoders para futuras predicciones
        self.label_encoders['area'] = le_area
        self.label_encoders['fabrica'] = le_fabrica
        
        log.info(f" {len(subcategoria_dummies.columns) + 2} features creadas")
        return df
    
    
    def _crear_features_anomalias(self, df):
        """
        PASO 14: Features de anomalías
        """
        log.info("\n Paso 14: Features de anomalías...")
        
        eventos_anomalia = [
            'minor_stoppages_count',
            'speed_loss_duration',
            'breakdown_and_equipment_failure_time_count',
            'process_failure_time_count',
            'quality_defect_time_loss_duration'
        ]
        
        for evento in eventos_anomalia:
            evento_corto = evento.replace('_duration', '').replace('_count', '').replace('_time', '').replace('_and', '')
            evento_corto = evento_corto.replace('__', '_')
            
            # Z-score por máquina
            df[f'{evento_corto}_zscore'] = df.groupby('id_maquina_dfos')[evento].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            ).fillna(0).clip(-10, 10)
            
            # Es outlier
            df[f'{evento_corto}_is_outlier'] = (np.abs(df[f'{evento_corto}_zscore']) > 3).astype(int)
        
        # Score compuesto
        outlier_cols = [col for col in df.columns if col.endswith('_is_outlier')]
        df['anomaly_score_composite'] = df[outlier_cols].sum(axis=1)
        
        log.info(f" {len(eventos_anomalia) * 2 + 1} features creadas")
        return df
    
    
    def _comparacion_pares(self, df):
        """
        PASO 15: Comparación con pares
        """
        log.info("\n Paso 15: Comparación con pares...")
        
        # Por subcategoría
        for evento in ['breakdown_and_equipment_failure_time_count', 'minor_stoppages_count']:
            evento_corto = evento.replace('_count', '').replace('_time', '').replace('_and', '')
            evento_corto = evento_corto.replace('__', '_')
            
            df[f'avg_{evento_corto}_subcategoria'] = df.groupby('id_subcategoria')[evento].transform(
                lambda x: x.rolling(window=720, min_periods=24).mean()
            ).fillna(0)
            
            df[f'diff_{evento_corto}_vs_subcategoria'] = (
                df[evento] - df[f'avg_{evento_corto}_subcategoria']
            )
        
        # Por área
        evento = 'breakdown_and_equipment_failure_time_count'
        evento_corto = 'breakdown_equipment_failure'
        
        df[f'concurrent_{evento_corto}_area_24h'] = df.groupby('id_fabrica_area')[evento].transform(
            lambda x: x.rolling(window=24, min_periods=1).sum()
        )
        
        df[f'area_stress_level_24h'] = df.groupby('id_fabrica_area')['total_events'].transform(
            lambda x: x.rolling(window=24, min_periods=1).sum()
        )
        
        # Flag alto riesgo
        breakdown_rate_subcat = df.groupby('id_subcategoria')['id_subcategoria'].transform('count')
        threshold_high_risk = breakdown_rate_subcat.quantile(0.75)
        
        df['is_high_risk_subcategoria'] = (breakdown_rate_subcat > threshold_high_risk).astype(int)
        
        log.info(" 7 features creadas")
        return df
    
    
    def _limpieza_final(self, df):
        """
        PASO 16: Limpieza final y optimización
        """
        log.info("\n Paso 16: Limpieza final...")
        
        # Llenar NaN
        df = df.fillna(0)
        
        # Optimizar tipos
        binary_cols = [col for col in df.columns if col.startswith(('es_', 'is_', 'tiene_', 'subcategoria_'))]
        for col in binary_cols:
            df[col] = df[col].astype('int8')
        
        count_cols = [col for col in df.columns if '_count' in col and df[col].dtype == 'float64']
        for col in count_cols:
            df[col] = df[col].astype('int32')
        
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype('float32')
        
        # Eliminar columnas redundantes
        columnas_eliminar = ['id_subcategoria', 'id_fabrica_area', 'id_fabrica']
        columnas_a_eliminar_final = [col for col in columnas_eliminar if col in df.columns]
        df = df.drop(columns=columnas_a_eliminar_final)
        
        log.info(" Limpieza completada")
        return df
    
    
    def _filtrar_ventanas_prediccion(self, df):
        """
        PASO 17: Filtrar solo últimas 24 horas para predicción
        """
        log.info("\n Paso 17: Filtrando ventanas para predicción...")
        
        fecha_max = df['timestamp_hora'].max()
        fecha_min_prediccion = fecha_max - pd.Timedelta(hours=24)
        
        df_prediccion = df[df['timestamp_hora'] > fecha_min_prediccion].copy()
        
        log.info(f" Ventanas filtradas:")
        log.info(f"   Total original: {len(df):,}")
        log.info(f"   Para predicción (últimas 24h): {len(df_prediccion):,}")
        log.info(f"   Rango: {df_prediccion['timestamp_hora'].min()} a {df_prediccion['timestamp_hora'].max()}")
        
        return df_prediccion
    
    
