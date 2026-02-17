"""
================================================================================
Feature Engineering V2 - CON ARTIFACTS DE PREPROCESAMIENTO
================================================================================
Sistema de Mantenimiento Predictivo
Versión corregida que garantiza consistencia entre entrenamiento e inferencia

CAMBIOS PRINCIPALES:
1. Carga preprocessing_artifacts.pkl al inicializar
2. Usa LabelEncoders guardados (transform, no fit_transform)
3. Crea columnas one-hot fijas (las esperadas por el modelo)
4. Usa threshold guardado para is_high_risk_subcategoria
5. Usa estadísticas guardadas para Z-scores
6. Maneja datos nuevos con valores por defecto
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import gc
import logging
import pickle
from pathlib import Path

log = logging.getLogger(__name__)

# ============================================================================
# FUNCIONES AUXILIARES PARA PIVOTEO (SIN CAMBIOS)
# ============================================================================

def crear_features_ventana(df_ventana):
    """
    Agrega eventos de una ventana de 1 hora en features
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
        'Adjustment', 'Speed Loss', 'Minor Stoppages', 'Idle Time',
        'Product Changeover', 'Changing Supplies', 'Changing Cutting Elements',
        'Lack of Product or WIP from Previous Process', 'Measurement',
        'Human Error', 'Material Issue'
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
    """
    log.info("\n Pivoteando eventos por máquina/hora...")
    

    # Validar que hay datos
    if perdida_clean is None or len(perdida_clean) == 0:
        log.warning(" No hay eventos para procesar (DataFrame vacío)")
        raise ValueError("No hay datos de eventos (bui_perdida) para procesar. "
                        "Verifique que existen registros en el período especificado.")

    perdida_clean['timestamp_hora'] = perdida_clean['fe_inicio'].dt.floor('h')
    maquinas_unicas = perdida_clean['id_maquina_dfos'].unique()
    log.info(f"Total máquinas a procesar: {len(maquinas_unicas)}")

    chunk_size = 100
    chunks_procesados = []

    for i in tqdm(range(0, len(maquinas_unicas), chunk_size), desc="Chunks de máquinas"):
        chunk_maquinas = maquinas_unicas[i:i+chunk_size]
        df_chunk = perdida_clean[perdida_clean['id_maquina_dfos'].isin(chunk_maquinas)].copy()
        resultados_chunk = []

        for maquina in chunk_maquinas:
            df_maquina = df_chunk[df_chunk['id_maquina_dfos'] == maquina]
            if len(df_maquina) == 0:
                continue

            id_linea = df_maquina['id_linea'].iloc[0]
            id_fabrica = df_maquina['id_fabrica'].iloc[0]

            for timestamp_hora, df_ventana in df_maquina.groupby('timestamp_hora'):
                features = crear_features_ventana(df_ventana)
                features['id_maquina_dfos'] = maquina
                features['id_linea'] = id_linea
                features['id_fabrica'] = id_fabrica
                features['timestamp_hora'] = timestamp_hora
                resultados_chunk.append(features)

        df_chunk_pivot = pd.DataFrame(resultados_chunk)
        chunks_procesados.append(df_chunk_pivot)
        del df_chunk, resultados_chunk
        gc.collect()

    log.info(" Concatenando resultados...")
    df_pivoteado = pd.concat(chunks_procesados, ignore_index=True)
    df_pivoteado = df_pivoteado.fillna(0)
    log.info(f" Pivoteo completado: {df_pivoteado.shape}")
    
    del chunks_procesados
    gc.collect()
    
    return df_pivoteado


# ============================================================================
# FUNCIONES PARA FEATURES DE MANTENIMIENTO (SIN CAMBIOS MAYORES)
# ============================================================================

def calcular_features_mantenimiento(df_pivot, ewos_df, tipo_mant, ventanas_temporales):
    """
    Calcula features de mantenimiento para cada ventana
    """
    log.info(f"\n Procesando mantenimientos {tipo_mant.upper()}...")

    ewos_df = ewos_df.copy()
    ewos_df['timestamp_hora'] = pd.to_datetime(ewos_df['fe_inicio_averia']).dt.floor('H')

    for nombre_ventana in ventanas_temporales.keys():
        df_pivot[f'{tipo_mant}_count_{nombre_ventana}'] = 0
        df_pivot[f'{tipo_mant}_duration_total_{nombre_ventana}'] = 0.0
        if nombre_ventana == '24h':
            df_pivot[f'{tipo_mant}_duration_max_{nombre_ventana}'] = 0.0

    df_pivot[f'horas_desde_ultimo_{tipo_mant}'] = 9999.0
    df_pivot[f'tiene_historial_{tipo_mant}'] = 0

    lineas_unicas = df_pivot['id_linea'].unique()

    for linea in tqdm(lineas_unicas, desc=f"Líneas {tipo_mant}"):
        ewos_linea = ewos_df[ewos_df['id_linea'] == linea].sort_values('timestamp_hora')
        if len(ewos_linea) == 0:
            continue

        mask_linea = df_pivot['id_linea'] == linea
        df_pivot.loc[mask_linea, f'tiene_historial_{tipo_mant}'] = 1

        for idx in df_pivot[mask_linea].index:
            timestamp_actual = df_pivot.loc[idx, 'timestamp_hora']

            for nombre_ventana, horas in ventanas_temporales.items():
                timestamp_inicio = timestamp_actual - pd.Timedelta(hours=horas)
                ewos_ventana = ewos_linea[
                    (ewos_linea['timestamp_hora'] >= timestamp_inicio) &
                    (ewos_linea['timestamp_hora'] < timestamp_actual)
                ]

                df_pivot.loc[idx, f'{tipo_mant}_count_{nombre_ventana}'] = len(ewos_ventana)
                df_pivot.loc[idx, f'{tipo_mant}_duration_total_{nombre_ventana}'] = ewos_ventana['duracion_reparacion_min'].sum()

                if nombre_ventana == '24h' and len(ewos_ventana) > 0:
                    df_pivot.loc[idx, f'{tipo_mant}_duration_max_{nombre_ventana}'] = ewos_ventana['duracion_reparacion_min'].max()

            ewos_previos = ewos_linea[ewos_linea['timestamp_hora'] < timestamp_actual]
            if len(ewos_previos) > 0:
                ultimo_mant = ewos_previos['timestamp_hora'].max()
                horas_desde = (timestamp_actual - ultimo_mant).total_seconds() / 3600
                df_pivot.loc[idx, f'horas_desde_ultimo_{tipo_mant}'] = horas_desde


def agregar_features_mantenimiento(df_pivoteado, bui_pm_ewo):
    """
    Agrega todas las features de mantenimiento al dataset pivoteado
    """
    log.info("\n Agregando features de mantenimiento...")
    
    if 'estado_ewo' not in bui_pm_ewo.columns:
        log.info(" estado_ewo no encontrado, calculando...")
        
        def calcular_estado_ewo(row):
            if pd.notna(row.get('fe_cerrar')):
                return 'CERRADA'
            elif pd.notna(row.get('fe_fin_mto')):
                return 'FINALIZADA_PENDIENTE_CIERRE'
            elif pd.notna(row.get('fe_fin_reparacion')):
                return 'REPARADA'
            elif pd.notna(row.get('fe_fin_diagnostico')):
                return 'DIAGNOSTICADA'
            elif pd.notna(row.get('fe_llegada_mto')):
                return 'EN_ATENCION'
            elif pd.notna(row.get('fe_aviso_mto')):
                return 'AVISADA'
            else:
                return 'CREADA_SIN_AVISO'
        
        bui_pm_ewo['estado_ewo'] = bui_pm_ewo.apply(calcular_estado_ewo, axis=1)

    bui_pm_ewo['duracion_reparacion_min'] = (
        (pd.to_datetime(bui_pm_ewo['fe_inicio_prod']) - pd.to_datetime(bui_pm_ewo['fe_inicio_averia']))
        .dt.total_seconds() / 60
    )

    ewos_para_features = bui_pm_ewo[
        (bui_pm_ewo['fl_borrador'] == 0) &
        (bui_pm_ewo['estado_ewo'].isin(['CERRADA', 'FINALIZADA_PENDIENTE_CIERRE'])) &
        (bui_pm_ewo['duracion_reparacion_min'].notna()) &
        (bui_pm_ewo['duracion_reparacion_min'] >= 0) &
        (bui_pm_ewo['duracion_reparacion_min'] <= 4320)
    ].copy()

    log.info(f" EWOs válidas: {len(ewos_para_features):,}")

    ewos_preventivos = ewos_para_features[ewos_para_features['fl_mantenimiento'] == 1].copy()
    ewos_correctivos = ewos_para_features[ewos_para_features['fl_mantenimiento'] == 0].copy()

    log.info(f"   - Preventivos: {len(ewos_preventivos):,}")
    log.info(f"   - Correctivos: {len(ewos_correctivos):,}")

    ventanas = {'24h': 24, '7d': 24 * 7, '30d': 24 * 30}

    calcular_features_mantenimiento(df_pivoteado, ewos_preventivos, 'pm', ventanas)
    calcular_features_mantenimiento(df_pivoteado, ewos_correctivos, 'cm', ventanas)

    df_pivoteado = df_pivoteado.fillna(0)
    log.info(f" Features de mantenimiento agregadas")
    
    return df_pivoteado


def enriquecer_con_metadata(df_pivoteado, bui_line):
    """
    Enriquece el dataset con metadata de líneas
    """
    log.info("\n Enriqueciendo con metadata de líneas...")
    
    df_pivoteado['id_linea'] = df_pivoteado['id_linea'].astype('int64')
    bui_line['id_linea'] = bui_line['id_linea'].astype('int64')
    
    df_pivoteado = df_pivoteado.merge(
        bui_line[['id_linea', 'id_subcategoria', 'id_fabrica_area']],
        on='id_linea',
        how='left',
        validate='m:1'
    )
    
    df_pivoteado['id_subcategoria'] = df_pivoteado['id_subcategoria'].fillna(0).astype('int32')
    df_pivoteado['id_fabrica_area'] = df_pivoteado['id_fabrica_area'].fillna(0).astype('int32')
    
    log.info(f" Metadata agregada")
    
    return df_pivoteado


def limpiar_datos_anomalos(df):
    """
    Limpia valores anómalos en el dataset
    """
    log.info("\n Limpiando datos anómalos...")
    
    df['run_time'] = df['run_time'].clip(lower=0, upper=120)
    df['total_duration'] = df['total_duration'].clip(lower=0.1, upper=120)
    df.loc[df['total_duration'] == 0, 'total_duration'] = 60.0
    
    log.info(" Datos limpios")
    
    return df


# ============================================================================
# CLASE PRINCIPAL: FEATURE ENGINEER V2 (CON ARTIFACTS)
# ============================================================================

class FeatureEngineer:
    """
    Clase para generar features del modelo de mantenimiento predictivo
    VERSIÓN 2: Usa preprocessing_artifacts.pkl para consistencia
    """
    
    def __init__(self, config):
        """
        Inicializa el Feature Engineer y carga artifacts
        """
        self.config = config
        self.artifacts = None
        self.label_encoders = {}
        
        # Cargar artifacts de preprocesamiento
        self._load_artifacts()
        
        log.info(" Feature Engineer V2 inicializado (con artifacts)")
    
    
    def _load_artifacts(self):
        """
        Carga los artifacts de preprocesamiento (VERSIÓN ESTRICTA PARA PRODUCCIÓN)
        
        IMPORTANTE: Este método FALLA si:
        - No encuentra el archivo preprocessing_artifacts.pkl
        - El archivo está incompleto
        - Faltan encoders requeridos
        
        Esto es por diseño: mejor fallar explícitamente que generar 
        predicciones incorrectas por training-serving skew.
        """
        # Buscar archivo en varias ubicaciones
        posibles_rutas = [
            Path(self.config.get('model', {}).get('path', '.')).parent / 'preprocessing_artifacts.pkl',
            Path('.') / 'preprocessing_artifacts.pkl',
            Path(__file__).parent.parent / 'models' / 'preprocessing_artifacts.pkl',
            Path(__file__).parent.parent / 'models' / 'inference_export' / 'preprocessing_artifacts.pkl',
        ]
        
        artifacts_path = None
        for ruta in posibles_rutas:
            if ruta.exists():
                artifacts_path = ruta
                break
        
        # FALLAR SI NO SE ENCUENTRA
        if artifacts_path is None:
            rutas_str = "\n".join(f"     - {r}" for r in posibles_rutas)
            raise FileNotFoundError(
                f"\n{'='*80}\n"
                f" ERROR CRÍTICO: preprocessing_artifacts.pkl NO ENCONTRADO\n"
                f"{'='*80}\n\n"
                f"   Este archivo es OBLIGATORIO para inferencia en producción.\n"
                f"   Sin él, hay riesgo de training-serving skew.\n\n"
                f"   Rutas buscadas:\n{rutas_str}\n\n"
                f"   Solución:\n"
                f"   1. Verifica que el archivo existe en una de las rutas arriba\n"
                f"   2. O actualiza la configuración 'model.path' en config.yaml\n"
                f"{'='*80}\n"
            )
        
        log.info(f" Cargando artifacts desde: {artifacts_path}")
        
        # Cargar archivo
        with open(artifacts_path, 'rb') as f:
            self.artifacts = pickle.load(f)
        
        # VALIDAR COMPLETITUD DE ARTIFACTS
        componentes_requeridos = {
            'version': 'Versión del artifact',
            'fecha_generacion': 'Fecha de generación',
            'label_encoders': 'Encoders de categorías',
            'onehot_columns': 'Columnas one-hot de subcategorías',
            'defaults': 'Valores por defecto para datos desconocidos',
            'stats_por_grupo': 'Estadísticas por grupo (Z-scores, pares)',
            'thresholds': 'Umbrales de clasificación'
        }
        
        faltantes = [nombre for nombre in componentes_requeridos.keys() 
                    if nombre not in self.artifacts]
        
        if faltantes:
            faltantes_str = "\n".join(f"     - {f}: {componentes_requeridos[f]}" 
                                    for f in faltantes)
            raise ValueError(
                f"\n{'='*80}\n"
                f" ERROR: Artifacts INCOMPLETO\n"
                f"{'='*80}\n\n"
                f"   Faltan los siguientes componentes requeridos:\n\n{faltantes_str}\n\n"
                f"   Solución: Re-generar preprocessing_artifacts.pkl desde entrenamiento\n"
                f"{'='*80}\n"
            )
        
        # VALIDAR ENCODERS ESPECÍFICOS
        encoders_requeridos = ['area', 'fabrica']
        encoders_faltantes = [e for e in encoders_requeridos 
                            if e not in self.artifacts['label_encoders']]
        
        if encoders_faltantes:
            raise ValueError(
                f"\n{'='*80}\n"
                f" ERROR: Faltan LabelEncoders requeridos\n"
                f"{'='*80}\n\n"
                f"   Encoders faltantes: {encoders_faltantes}\n"
                f"   Encoders disponibles: {list(self.artifacts['label_encoders'].keys())}\n\n"
                f"   Solución: Re-generar preprocessing_artifacts.pkl con todos los encoders\n"
                f"{'='*80}\n"
            )
        
        # VALIDAR ESTRUCTURA DE ONEHOT_COLUMNS
        if 'subcategorias' not in self.artifacts['onehot_columns']:
            raise ValueError(
                " ERROR: artifacts['onehot_columns'] no contiene 'subcategorias'"
            )
        
        if 'subcategoria_columns' not in self.artifacts['onehot_columns']:
            raise ValueError(
                " ERROR: artifacts['onehot_columns'] no contiene 'subcategoria_columns'"
            )
        
        # TODO: Agregar validación de versión compatible con modelo
        
        # Logging de confirmación
        log.info(f" Artifacts validado y cargado correctamente:")
        log.info(f"   • Versión: {self.artifacts['version']}")
        log.info(f"   • Fecha generación: {self.artifacts['fecha_generacion']}")
        log.info(f"   • LabelEncoders: {list(self.artifacts['label_encoders'].keys())}")
        
        # Detalles de encoders
        le_area = self.artifacts['label_encoders']['area']
        le_fabrica = self.artifacts['label_encoders']['fabrica']
        log.info(f"   • Encoder área: {len(le_area.classes_)} clases conocidas")
        log.info(f"   • Encoder fábrica: {len(le_fabrica.classes_)} clases conocidas")
        
        # Detalles de one-hot
        n_subcats = len(self.artifacts['onehot_columns']['subcategorias'])
        log.info(f"   • Subcategorías one-hot: {n_subcats} categorías")
        
        # Detalles de defaults
        log.info(f"   • Default área: {self.artifacts['defaults'].get('area_default_code', 'N/A')}")
        log.info(f"   • Default fábrica: {self.artifacts['defaults'].get('fabrica_default_code', 'N/A')}")
        
        # Extraer label encoders para uso rápido
        self.label_encoders = self.artifacts['label_encoders']

    
    def transform(self, bui_perdida, bui_pm_ewo, bui_line):
        """
        Método principal: transforma datos raw en features listos para predicción
        """
        log.info("=" * 80)
        log.info(" INICIANDO FEATURE ENGINEERING V2 (CON ARTIFACTS)")
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
        
        # PASO 13: Encoding de categorías (CORREGIDO - usa artifacts)
        df = self._encoding_categorias_con_artifacts(df)
        
        # PASO 14: Features de anomalías (CORREGIDO - usa artifacts)
        df = self._crear_features_anomalias_con_artifacts(df)
        
        # PASO 15: Comparación con pares (CORREGIDO - usa artifacts)
        df = self._comparacion_pares_con_artifacts(df)
        
        # PASO 16: Limpieza final
        df = self._limpieza_final(df)
        
        # PASO 17: Filtrar solo últimas 24h
        df = self._filtrar_ventanas_prediccion(df)
        
        log.info("\n" + "=" * 80)
        log.info(" FEATURE ENGINEERING V2 COMPLETADO")
        log.info("=" * 80)
        log.info(f"Shape final: {df.shape}")
        
        return df
    
    
    # ========================================================================
    # PASOS 5-12: SIN CAMBIOS (copiar del original)
    # ========================================================================
    
    def _crear_features_eventos_detallados(self, df):
        """PASO 5: Features detalladas por evento"""
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
            df[f'{nombre_evento}_duracion_avg'] = np.where(
                df[col_count] > 0, df[col_duracion] / df[col_count], 0
            )
            df[f'{nombre_evento}_tiempo_pct'] = (
                df[col_duracion] / df['total_duration'] * 100
            ).clip(upper=100)
            df[f'{nombre_evento}_frecuencia_por_hora'] = np.where(
                df['total_duration'] > 0,
                df[col_count] / (df['total_duration'] / 60), 0
            )
            df[f'{nombre_evento}_severity_score'] = (
                df[col_count] * df[f'{nombre_evento}_duracion_avg']
            )
        
        df['minor_stoppages_freq_normalizada'] = np.where(
            df['run_time'] > 0,
            df['minor_stoppages_count'] / (df['run_time'] / 60), 0
        )
        
        log.info(f" {len(eventos_detallados) * 4 + 1} features creadas")
        return df
    
    
    def _crear_metricas_agregadas(self, df):
        """PASO 6: Métricas agregadas"""
        log.info("\n Paso 6: Métricas agregadas...")
        
        perdidas_cols = [col for col in df.columns if '_duration' in col and
                         any(x in col for x in ['loss', 'defect', 'failure', 'stoppage', 'idle'])]
        
        df['total_perdidas_duracion'] = df[perdidas_cols].sum(axis=1)
        df['total_perdidas_tiempo_pct'] = (
            df['total_perdidas_duracion'] / df['total_duration'] * 100
        ).clip(upper=100)
        
        count_cols = [col for col in df.columns if col.endswith('_count') and
                      not col.startswith(('run', 'mpl', 'ucl', 'pdl', 'desconocido', 'total'))]
        
        df['diversidad_eventos'] = (df[count_cols] > 0).sum(axis=1)
        
        duracion_cols = [col for col in df.columns if col.endswith('_duration') and
                         not col.startswith(('total', 'run', 'mpl', 'ucl', 'pdl'))]
        
        df['evento_mas_largo_duracion'] = df[duracion_cols].max(axis=1)
        df['evento_mas_frecuente_count'] = df[count_cols].max(axis=1)
        
        eventos_criticos_dur = [
            'breakdown_mechanical_duration', 'breakdown_electrical_duration',
            'breakdown_ic_duration', 'process_failure_time_duration'
        ]
        df['duracion_eventos_criticos'] = df[eventos_criticos_dur].sum(axis=1)
        df['ratio_eventos_criticos_total'] = np.where(
            df['total_perdidas_duracion'] > 0,
            df['duracion_eventos_criticos'] / df['total_perdidas_duracion'], 0
        )
        
        log.info(" 7 features creadas")
        return df
    
    
    def _crear_features_mantenimiento_mejoradas(self, df):
        """PASO 7: Features de mantenimiento mejoradas"""
        log.info("\n Paso 7: Features de mantenimiento mejoradas...")
        
        df['dias_desde_ultimo_pm'] = df['horas_desde_ultimo_pm'] / 24
        df['dias_desde_ultimo_cm'] = df['horas_desde_ultimo_cm'] / 24
        df['pm_frecuencia_mensual'] = df['pm_count_30d'] / 30
        df['cm_tasa_fallas_mensual'] = df['cm_count_30d'] / 30
        df['mantenimiento_total_duracion_7d'] = (
            df['pm_duration_total_7d'] + df['cm_duration_total_7d']
        )
        
        log.info(" 5 features creadas")
        return df
    
    
    def _crear_rolling_windows(self, df):
        """PASO 8: Rolling windows (2h, 6h, 12h, 24h)"""
        log.info("\n Paso 8: Rolling windows...")
        
        df = df.sort_values(['id_maquina_dfos', 'timestamp_hora']).reset_index(drop=True)
        
        eventos_rolling = [
            'minor_stoppages_count', 'speed_loss_duration',
            'process_failure_time_count', 'measurement_and_adjustment_count',
            'breakdown_and_equipment_failure_time_count', 'quality_defect_time_loss_duration'
        ]
        
        ventanas = {'2h': 2, '6h': 6, '12h': 12, '24h': 24}
        
        for nombre_ventana, horas in ventanas.items():
            for evento in eventos_rolling:
                evento_corto = evento.replace('_duration', '').replace('_count', '').replace('_time', '').replace('_and', '')
                evento_corto = evento_corto.replace('__', '_')
                
                col_rolling = f'{evento_corto}_rolling_{nombre_ventana}'
                df[col_rolling] = df.groupby('id_maquina_dfos')[evento].transform(
                    lambda x: x.rolling(window=horas, min_periods=1).sum()
                )
                
                col_avg = f'{evento_corto}_rolling_{nombre_ventana}_avg'
                df[col_avg] = df[col_rolling] / horas
            
            gc.collect()
        
        log.info(f" {len(eventos_rolling) * len(ventanas) * 2} features creadas")
        return df
    
    
    def _crear_tendencias(self, df):
        """PASO 9: Tendencias y aceleraciones"""
        log.info("\n Paso 9: Tendencias...")
        
        eventos_tendencia = [
            'minor_stoppages', 'speed_loss', 'process_failure', 'breakdown_equipment_failure'
        ]
        
        for evento in eventos_tendencia:
            col_12h = f'{evento}_rolling_12h'
            
            if col_12h not in df.columns:
                log.warning(f" Columna {col_12h} no encontrada")
                continue
            
            col_12h_prev = df.groupby('id_maquina_dfos')[col_12h].shift(12)
            
            df[f'{evento}_trend_24h'] = df[col_12h] - col_12h_prev.fillna(0)
            df[f'{evento}_accel_24h'] = np.where(
                col_12h_prev > 0,
                ((df[col_12h] - col_12h_prev) / col_12h_prev * 100), 0
            ).clip(-500, 500)
        
        log.info(f" {len(eventos_tendencia) * 2} features creadas")
        return df
    
    
    def _crear_ratios(self, df):
        """PASO 10: Ratios y proporciones"""
        log.info("\n Paso 10: Ratios...")
        
        tiempo_total_operativo = df['run_time'] + df['mpl_time'] + df['ucl_time'] + df['pdl_time']
        tiempo_total_operativo = tiempo_total_operativo.clip(lower=0.1)
        
        df['ratio_run_total'] = (df['run_time'] / tiempo_total_operativo * 100).clip(0, 100)
        df['ratio_mpl_total'] = (df['mpl_time'] / tiempo_total_operativo * 100).clip(0, 100)
        df['ratio_ucl_total'] = (df['ucl_time'] / tiempo_total_operativo * 100).clip(0, 100)
        df['ratio_unplanned_total'] = ((df['mpl_time'] + df['ucl_time']) / tiempo_total_operativo * 100).clip(0, 100)
        
        df['ratio_minor_vs_critical'] = np.where(
            df['breakdown_and_equipment_failure_time_count'] > 0,
            df['minor_stoppages_count'] / df['breakdown_and_equipment_failure_time_count'], 0
        )
        df['ratio_quality_vs_performance'] = np.where(
            df['speed_loss_duration'] > 0,
            df['quality_defect_time_loss_duration'] / df['speed_loss_duration'], 0
        )
        df['events_per_hour'] = df['total_events'] / (df['total_duration'] / 60)
        df['avg_duration_per_event'] = np.where(
            df['total_events'] > 0, df['total_duration'] / df['total_events'], 0
        )
        
        log.info(" 8 features creadas")
        return df
    
    
    def _crear_features_temporales(self, df):
        """PASO 11: Features temporales"""
        log.info("\n Paso 11: Features temporales...")
        
        df['hora_del_dia'] = df['timestamp_hora'].dt.hour
        df['dia_semana'] = df['timestamp_hora'].dt.dayofweek
        df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
        
        df['es_turno_noche'] = ((df['hora_del_dia'] >= 22) | (df['hora_del_dia'] < 6)).astype(int)
        df['es_turno_tarde'] = ((df['hora_del_dia'] >= 14) & (df['hora_del_dia'] < 22)).astype(int)
        df['es_turno_mañana'] = ((df['hora_del_dia'] >= 6) & (df['hora_del_dia'] < 14)).astype(int)
        
        df['es_inicio_turno'] = df['hora_del_dia'].isin([6, 14, 22]).astype(int)
        df['es_fin_turno'] = df['hora_del_dia'].isin([5, 13, 21]).astype(int)
        
        df['hora_sin'] = np.sin(2 * np.pi * df['hora_del_dia'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora_del_dia'] / 24)
        df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        
        log.info(" 13 features creadas")
        return df
    
    
    def _crear_variabilidad(self, df):
        """PASO 12: Variabilidad y estabilidad (7 días)"""
        log.info("\n Paso 12: Variabilidad...")
        
        eventos_variabilidad = [
            'minor_stoppages_count', 'speed_loss_duration',
            'breakdown_and_equipment_failure_time_count'
        ]
        
        for evento in eventos_variabilidad:
            evento_corto = evento.replace('_duration', '').replace('_count', '').replace('_time', '').replace('_and', '')
            evento_corto = evento_corto.replace('__', '_')
            
            df[f'{evento_corto}_std_7d'] = df.groupby('id_maquina_dfos')[evento].transform(
                lambda x: x.rolling(window=168, min_periods=24).std()
            ).fillna(0)
            
            rolling_mean = df.groupby('id_maquina_dfos')[evento].transform(
                lambda x: x.rolling(window=168, min_periods=24).mean()
            )
            df[f'{evento_corto}_cv_7d'] = np.where(
                rolling_mean > 0, (df[f'{evento_corto}_std_7d'] / rolling_mean), 0
            ).clip(0, 10)
        
        # Días desde último breakdown
        log.info("Calculando días desde último breakdown...")
        df['horas_desde_ultimo_breakdown'] = 9999.0
        mask_breakdown = df['breakdown_and_equipment_failure_time_count'] > 0
        
        for maquina in tqdm(df['id_maquina_dfos'].unique(), desc="Máquinas"):
            mask_maquina = df['id_maquina_dfos'] == maquina
            timestamps_breakdown = df.loc[mask_maquina & mask_breakdown, 'timestamp_hora'].values
            
            if len(timestamps_breakdown) == 0:
                continue
            
            for idx in df[mask_maquina].index:
                timestamp_actual = df.loc[idx, 'timestamp_hora']
                previos = timestamps_breakdown[timestamps_breakdown < timestamp_actual]
                
                if len(previos) > 0:
                    ultimo = previos.max()
                    horas = (timestamp_actual - ultimo).total_seconds() / 3600
                    df.loc[idx, 'horas_desde_ultimo_breakdown'] = horas
        
        df['dias_desde_ultimo_breakdown'] = df['horas_desde_ultimo_breakdown'] / 24
        
        log.info(f" {len(eventos_variabilidad) * 2 + 2} features creadas")
        return df
    
    
    # ========================================================================
    # PASOS 13-15: CORREGIDOS PARA USAR ARTIFACTS
    # ========================================================================
    
    def _encoding_categorias_con_artifacts(self, df):
        """
        PASO 13: Encoding de categorías usando artifacts (SIN FALLBACKS PELIGROSOS)
        
        IMPORTANTE: Este método asume que self.artifacts fue validado en __init__.
        No hay fallbacks con fit_transform para evitar training-serving skew.
        """
        log.info("\n Paso 13: Encoding de categorías (con artifacts)...")
        
        # ============================================================
        # ONE-HOT ENCODING FIJO PARA SUBCATEGORÍAS
        # ============================================================
        
        subcategorias_esperadas = self.artifacts['onehot_columns']['subcategorias']
        log.info(f"    Creando {len(subcategorias_esperadas)} columnas one-hot...")
        
        # Identificar subcategorías presentes en datos de inferencia
        subcategorias_en_datos = set(df['id_subcategoria'].unique())
        subcategorias_nuevas = subcategorias_en_datos - set(subcategorias_esperadas)
        subcategorias_faltantes = set(subcategorias_esperadas) - subcategorias_en_datos
        
        if subcategorias_nuevas:
            log.warning(f"    Subcategorías NUEVAS (no vistas en entrenamiento): {len(subcategorias_nuevas)}")
            log.warning(f"      IDs: {sorted(list(subcategorias_nuevas))[:10]}")
            log.warning(f"      → Estas se ignorarán (todas sus columnas serán 0)")
        
        if subcategorias_faltantes:
            log.info(f"   ℹ️ Subcategorías del entrenamiento NO presentes: {len(subcategorias_faltantes)}")
            log.info(f"      → Sus columnas se crearán con valor 0")
        
        # Crear TODAS las columnas esperadas (inicializar con 0)
        for subcat in subcategorias_esperadas:
            col_name = f'subcategoria_{int(subcat)}'
            df[col_name] = 0
        
        # Activar (1) solo para subcategorías presentes Y conocidas
        for subcat in subcategorias_en_datos:
            if subcat in subcategorias_esperadas:
                col_name = f'subcategoria_{int(subcat)}'
                df.loc[df['id_subcategoria'] == subcat, col_name] = 1
        
        # Validar exclusividad
        subcategoria_cols = [f'subcategoria_{int(s)}' for s in subcategorias_esperadas]
        sum_onehot = df[subcategoria_cols].sum(axis=1)
        filas_con_1 = (sum_onehot == 1).sum()
        filas_con_0 = (sum_onehot == 0).sum()
        filas_con_multiples = (sum_onehot > 1).sum()
        
        log.info(f"    {len(subcategorias_esperadas)} columnas one-hot creadas")
        log.info(f"      Filas con 1 activo: {filas_con_1:,} ({filas_con_1/len(df)*100:.1f}%)")
        log.info(f"      Filas con 0 activos: {filas_con_0:,} ({filas_con_0/len(df)*100:.1f}%) ← subcats nuevas")
        
        if filas_con_multiples > 0:
            raise ValueError(
                f" ERROR en one-hot encoding: {filas_con_multiples:,} filas tienen múltiples subcategorías activas"
            )
        
        # ============================================================
        # LABEL ENCODING - ÁREA (sin fallback)
        # ============================================================
        
        le_area = self.label_encoders['area']
        default_area = self.artifacts['defaults']['area_default_code']
        
        area_values = df['id_fabrica_area'].astype(str).values
        known_areas = set(le_area.classes_)
        
        # Identificar valores no vistos
        valores_no_vistos = set(area_values) - known_areas
        if valores_no_vistos:
            log.warning(f"    Áreas NO VISTAS en entrenamiento: {len(valores_no_vistos)}")
            if len(valores_no_vistos) <= 10:
                log.warning(f"      IDs: {sorted(list(valores_no_vistos))}")
            log.warning(f"      → Usando código por defecto: {default_area}")
        
        # Aplicar encoding
        encoded_areas = [
            le_area.transform([area])[0] if area in known_areas else default_area
            for area in area_values
        ]
        df['id_fabrica_area_encoded'] = encoded_areas
        
        # Estadísticas
        n_conocidas = sum(1 for enc in encoded_areas if enc != default_area)
        pct_conocidas = n_conocidas / len(encoded_areas) * 100
        n_unicas = df['id_fabrica_area_encoded'].nunique()
        
        log.info(f"    id_fabrica_area_encoded creada")
        log.info(f"      Valores conocidos: {n_conocidas:,} ({pct_conocidas:.1f}%)")
        log.info(f"      Valores con default: {len(df) - n_conocidas:,} ({100-pct_conocidas:.1f}%)")
        log.info(f"      Valores únicos: {n_unicas}")
        
        # ============================================================
        # LABEL ENCODING - FÁBRICA (sin fallback)
        # ============================================================
        
        le_fabrica = self.label_encoders['fabrica']
        default_fabrica = self.artifacts['defaults']['fabrica_default_code']
        
        fabrica_values = df['id_fabrica'].astype(str).values
        known_fabricas = set(le_fabrica.classes_)
        
        # Identificar valores no vistos
        valores_no_vistos_fab = set(fabrica_values) - known_fabricas
        if valores_no_vistos_fab:
            log.warning(f"    Fábricas NO VISTAS en entrenamiento: {len(valores_no_vistos_fab)}")
            if len(valores_no_vistos_fab) <= 10:
                log.warning(f"      IDs: {sorted(list(valores_no_vistos_fab))}")
            log.warning(f"      → Usando código por defecto: {default_fabrica}")
        
        # Aplicar encoding
        encoded_fabricas = [
            le_fabrica.transform([fab])[0] if fab in known_fabricas else default_fabrica
            for fab in fabrica_values
        ]
        df['id_fabrica_encoded'] = encoded_fabricas
        
        # Estadísticas
        n_conocidas_fab = sum(1 for enc in encoded_fabricas if enc != default_fabrica)
        pct_conocidas_fab = n_conocidas_fab / len(encoded_fabricas) * 100
        n_unicas_fab = df['id_fabrica_encoded'].nunique()
        
        log.info(f"    id_fabrica_encoded creada")
        log.info(f"      Valores conocidos: {n_conocidas_fab:,} ({pct_conocidas_fab:.1f}%)")
        log.info(f"      Valores con default: {len(df) - n_conocidas_fab:,} ({100-pct_conocidas_fab:.1f}%)")
        log.info(f"      Valores únicos: {n_unicas_fab}")
        
        log.info(f" Encoding completado exitosamente")
        return df

    
    
    def _crear_features_anomalias_con_artifacts(self, df):
        """
        PASO 14: Features de anomalías (CORREGIDO - usa estadísticas guardadas)
        """
        log.info("\n Paso 14: Features de anomalías (con artifacts)...")
        
        eventos_anomalia = {
            'minor_stoppages_count': 'minor_stoppages',
            'speed_loss_duration': 'speed_loss',
            'breakdown_and_equipment_failure_time_count': 'breakdown_equipment_failure',
            'process_failure_time_count': 'process_failure',
            'quality_defect_time_loss_duration': 'quality_defect_loss'
        }
        
        # Usar estadísticas guardadas (ya validadas en __init__)
        stats_zscore = self.artifacts['stats_por_grupo']['zscore']
        log.info("   Usando estadísticas de entrenamiento para Z-scores")

        
        for evento_col, evento_corto in eventos_anomalia.items():
            if evento_col not in df.columns:
                continue
            
            if evento_corto in stats_zscore:

                # Usar estadísticas guardadas
                stats = stats_zscore[evento_corto]
                stats_por_maquina = stats['por_maquina']
                global_mean = stats['global_mean']
                global_std = stats['global_std']
                
                # Calcular Z-score usando estadísticas del entrenamiento
                zscores = []
                for idx, row in df.iterrows():
                    maquina = row['id_maquina_dfos']
                    valor = row[evento_col]
                    
                    if maquina in stats_por_maquina:
                        mean = stats_por_maquina[maquina]['mean']
                        std = stats_por_maquina[maquina]['std']
                    else:
                        # Máquina nueva: usar estadísticas globales
                        mean = global_mean
                        std = global_std
                    
                    if std > 0:
                        zscore = (valor - mean) / std
                    else:
                        zscore = 0
                    
                    zscores.append(np.clip(zscore, -10, 10))
                
                df[f'{evento_corto}_zscore'] = zscores
                
            
            
            # Es outlier
            df[f'{evento_corto}_is_outlier'] = (np.abs(df[f'{evento_corto}_zscore']) > 3).astype(int)
        
        # Score compuesto
        outlier_cols = [col for col in df.columns if col.endswith('_is_outlier')]
        df['anomaly_score_composite'] = df[outlier_cols].sum(axis=1)
        
        log.info(f" {len(eventos_anomalia) * 2 + 1} features creadas")
        return df
    
    
    def _comparacion_pares_con_artifacts(self, df):
        """
        PASO 15: Comparación con pares (CORREGIDO - usa estadísticas guardadas)
        """
        log.info("\n Paso 15: Comparación con pares (con artifacts)...")
        
        # Usar estadísticas guardadas (ya validadas en __init__)
        stats_pares = self.artifacts['stats_por_grupo']['pares']
        log.info("   Usando estadísticas de entrenamiento para comparación con pares")

        eventos_pares = {
            'breakdown_and_equipment_failure_time_count': 'breakdown_equipment_failure',
            'minor_stoppages_count': 'minor_stoppages'
        }

        
        for evento_col, evento_corto in eventos_pares.items():
            if evento_col not in df.columns:
                continue
            
            if evento_corto in stats_pares:

                # Usar medias guardadas
                media_por_subcat = stats_pares[evento_corto]['media_por_subcategoria']
                media_global = stats_pares[evento_corto]['media_global']
                
                # Mapear subcategoría a media
                df[f'avg_{evento_corto}_subcategoria'] = df['id_subcategoria'].map(
                    lambda x: media_por_subcat.get(x, media_global)
                )
                
            
            
            df[f'diff_{evento_corto}_vs_subcategoria'] = (
                df[evento_col] - df[f'avg_{evento_corto}_subcategoria']
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
        
        # ============================================================
        # FLAG ALTO RIESGO - USAR THRESHOLD GUARDADO
        # ============================================================
        
        # Usar threshold guardado (ya validado en __init__)

        threshold = self.artifacts['thresholds'].get('high_risk_threshold', 0.1)
        breakdown_rates = self.artifacts['thresholds'].get('breakdown_rate_by_subcategoria', {})
            
        log.info(f"   Usando threshold guardado: {threshold:.6f}")
            
        # Mapear subcategoría a su tasa de fallas
        df['_temp_breakdown_rate'] = df['id_subcategoria'].map(
                lambda x: breakdown_rates.get(x, 0)
            )
            
        df['is_high_risk_subcategoria'] = (df['_temp_breakdown_rate'] > threshold).astype(int)
            
            # Limpiar columna temporal
        df.drop(columns=['_temp_breakdown_rate'], inplace=True)
            
       
        
        log.info(" 7 features creadas")
        return df
    
    
    def _limpieza_final(self, df):
        """PASO 16: Limpieza final y optimización"""
        log.info("\n Paso 16: Limpieza final...")
        
        df = df.fillna(0)
        
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
        """PASO 17: Filtrar solo últimas 24 horas para predicción"""
        log.info("\n Paso 17: Filtrando ventanas para predicción...")
        
        fecha_max = df['timestamp_hora'].max()
        fecha_min_prediccion = fecha_max - pd.Timedelta(hours=24)
        
        df_prediccion = df[df['timestamp_hora'] > fecha_min_prediccion].copy()
        
        log.info(f" Ventanas filtradas:")
        log.info(f"   Total original: {len(df):,}")
        log.info(f"   Para predicción (últimas 24h): {len(df_prediccion):,}")
        log.info(f"   Rango: {df_prediccion['timestamp_hora'].min()} a {df_prediccion['timestamp_hora'].max()}")
        
        return df_prediccion
