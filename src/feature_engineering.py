"""
feature_engineering.py - Generación de features para inferencia.
Usa preprocessing_artifacts.pkl para garantizar consistencia con el entrenamiento
(LabelEncoders, columnas one-hot, estadísticas Z-score, umbrales de riesgo).
"""

import gc
import pickle
import logging

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

log = logging.getLogger(__name__)


# DEFINICIÓN DE EVENTOS ───────────────────────────────────────────────────────

_ESTADOS_MACRO = ['RUN', 'MPL', 'UCL', 'PDL', 'DESCONOCIDO']

_EVENTOS_NIVEL2 = [
    'Minor Stoppages', 'Speed Loss', 'Measurement & Adjustment',
    'Process Failure Time', 'Idle Time',
    'Material Availability at Line-Side Loss', 'Quality Defect Time Loss',
    'Maintenance Time', 'Cleaning & Sanitation Time', 'Changeover Time',
    'Cutting Blade Change', 'Breakdown & Equipment Failure Time',
    'Preparatory & Close Out Time Losses', 'Planned Stoppage Time',
]

_BREAKDOWNS_TIPO = {
    'breakdown_mechanical': 'Breakdown - Mechanical',
    'breakdown_electrical': 'Breakdown - Electrical',
    'breakdown_ic':         'Breakdown - Instrumentation & Control',
}

_OTROS_EVENTOS_N3 = [
    'Adjustment', 'Speed Loss', 'Minor Stoppages', 'Idle Time',
    'Product Changeover', 'Changing Supplies', 'Changing Cutting Elements',
    'Lack of Product or WIP from Previous Process', 'Measurement',
    'Human Error', 'Material Issue',
]

_VENTANAS_MANT = {'24h': 24, '7d': 24 * 7, '30d': 24 * 30}

_EVENTOS_ROLLING = [
    'minor_stoppages_count', 'speed_loss_duration',
    'process_failure_time_count', 'measurement_and_adjustment_count',
    'breakdown_and_equipment_failure_time_count', 'quality_defect_time_loss_duration',
]

_EVENTOS_DETALLADOS = {
    'speed_loss':              ('speed_loss_duration',                      'speed_loss_count'),
    'minor_stoppages':         ('minor_stoppages_duration',                 'minor_stoppages_count'),
    'process_failure':         ('process_failure_time_duration',            'process_failure_time_count'),
    'measurement_adjustment':  ('measurement_and_adjustment_duration',      'measurement_and_adjustment_count'),
    'quality_defects':         ('quality_defect_time_loss_duration',        'quality_defect_time_loss_count'),
    'material_loss':           ('material_availability_at_line-side_loss_duration',
                                'material_availability_at_line-side_loss_count'),
    'idle_time':               ('idle_time_duration',                       'idle_time_count'),
    'breakdown_mechanical':    ('breakdown_mechanical_duration',            'breakdown_mechanical_count'),
    'breakdown_electrical':    ('breakdown_electrical_duration',            'breakdown_electrical_count'),
    'breakdown_ic':            ('breakdown_ic_duration',                    'breakdown_ic_count'),
}

_EVENTOS_VARIABILIDAD = [
    'minor_stoppages_count',
    'speed_loss_duration',
    'breakdown_and_equipment_failure_time_count',
]

_EVENTOS_ZSCORE = {
    'minor_stoppages_count':                          'minor_stoppages',
    'speed_loss_duration':                            'speed_loss',
    'breakdown_and_equipment_failure_time_count':     'breakdown_equipment_failure',
    'process_failure_time_count':                     'process_failure',
    'quality_defect_time_loss_duration':              'quality_defect_loss',
}

_EVENTOS_PARES = {
    'breakdown_and_equipment_failure_time_count': 'breakdown_equipment_failure',
    'minor_stoppages_count':                      'minor_stoppages',
}


# FUNCIONES DE PIVOTEO ────────────────────────────────────────────────────────

def _crearFeaturesVentana(dfVentana):
    """Agrega los eventos de una ventana de 1h en un diccionario de features."""
    f = {}

    for estado in _ESTADOS_MACRO:
        mask = dfVentana['de_perdida_1'] == estado
        f[f'{estado.lower()}_time']  = dfVentana.loc[mask, 'duracion_minutos'].sum()
        f[f'{estado.lower()}_count'] = mask.sum()

    for evento in _EVENTOS_NIVEL2:
        mask    = dfVentana['de_perdida_2'] == evento
        colBase = evento.lower().replace(' ', '_').replace('&', 'and').replace('/', '_')
        f[f'{colBase}_duration'] = dfVentana.loc[mask, 'duracion_minutos'].sum()
        f[f'{colBase}_count']    = mask.sum()

    for colName, evento in _BREAKDOWNS_TIPO.items():
        mask = dfVentana['de_perdida_3'] == evento
        f[f'{colName}_count']    = mask.sum()
        f[f'{colName}_duration'] = dfVentana.loc[mask, 'duracion_minutos'].sum()

    for evento in _OTROS_EVENTOS_N3:
        mask    = dfVentana['de_perdida_3'] == evento
        colBase = evento.lower().replace(' ', '_').replace('/', '_')
        f[f'{colBase}_nivel3_count']    = mask.sum()
        f[f'{colBase}_nivel3_duration'] = dfVentana.loc[mask, 'duracion_minutos'].sum()

    n = len(dfVentana)
    f['total_events']        = n
    f['total_duration']      = dfVentana['duracion_minutos'].sum()
    f['avg_event_duration']  = dfVentana['duracion_minutos'].mean() if n else 0
    f['max_event_duration']  = dfVentana['duracion_minutos'].max()  if n else 0

    return f


def _pivotarEventosPorMaquinaHora(perdidaClean):
    """Pivota bui_perdida en una fila por (máquina, hora), en chunks de 100 máquinas."""
    if perdidaClean is None or perdidaClean.empty:
        raise ValueError(
            "Sin datos de eventos (bui_perdida). "
            "Verifica que existen registros en el período especificado."
        )

    log.info("Pivotando eventos por máquina/hora...")
    perdidaClean['timestamp_hora'] = perdidaClean['fe_inicio'].dt.floor('h')
    maquinas = perdidaClean['id_maquina_dfos'].unique()
    log.info(f"  {len(maquinas)} máquinas")

    chunks = []
    for i in tqdm(range(0, len(maquinas), 100), desc="Chunks máquinas"):
        grupoMaquinas = maquinas[i:i + 100]
        dfChunk       = perdidaClean[perdidaClean['id_maquina_dfos'].isin(grupoMaquinas)].copy()
        filas         = []

        for maquina in grupoMaquinas:
            dfMaq = dfChunk[dfChunk['id_maquina_dfos'] == maquina]
            if dfMaq.empty:
                continue
            idLinea   = dfMaq['id_linea'].iloc[0]
            idFabrica = dfMaq['id_fabrica'].iloc[0]

            for tsHora, dfVentana in dfMaq.groupby('timestamp_hora'):
                fila = _crearFeaturesVentana(dfVentana)
                fila.update({
                    'id_maquina_dfos': maquina,
                    'id_linea':        idLinea,
                    'id_fabrica':      idFabrica,
                    'timestamp_hora':  tsHora,
                })
                filas.append(fila)

        chunks.append(pd.DataFrame(filas))
        del dfChunk, filas
        gc.collect()

    dfPivot = pd.concat(chunks, ignore_index=True).fillna(0)
    log.info(f"  Pivoteo completado: {dfPivot.shape}")
    del chunks
    gc.collect()
    return dfPivot


# FEATURES DE MANTENIMIENTO ───────────────────────────────────────────────────

def _calcularFeaturesMantenimiento(dfPivot, ewosDF, tipoMant, ventanas):
    """Calcula conteos, duraciones y tiempo desde último EWO por tipo de mantenimiento."""
    ewosDF = ewosDF.copy()
    ewosDF['timestamp_hora'] = pd.to_datetime(ewosDF['fe_inicio_averia']).dt.floor('H')

    for nombreV in ventanas:
        dfPivot[f'{tipoMant}_count_{nombreV}']          = 0
        dfPivot[f'{tipoMant}_duration_total_{nombreV}'] = 0.0
        if nombreV == '24h':
            dfPivot[f'{tipoMant}_duration_max_{nombreV}'] = 0.0

    dfPivot[f'horas_desde_ultimo_{tipoMant}'] = 9999.0
    dfPivot[f'tiene_historial_{tipoMant}']    = 0

    for linea in tqdm(dfPivot['id_linea'].unique(), desc=f"Líneas {tipoMant}"):
        ewosLinea = ewosDF[ewosDF['id_linea'] == linea].sort_values('timestamp_hora')
        if ewosLinea.empty:
            continue

        maskLinea = dfPivot['id_linea'] == linea
        dfPivot.loc[maskLinea, f'tiene_historial_{tipoMant}'] = 1

        for idx in dfPivot[maskLinea].index:
            tsActual = dfPivot.loc[idx, 'timestamp_hora']

            for nombreV, horas in ventanas.items():
                ewosVentana = ewosLinea[
                    (ewosLinea['timestamp_hora'] >= tsActual - pd.Timedelta(hours=horas)) &
                    (ewosLinea['timestamp_hora'] <  tsActual)
                ]
                dfPivot.loc[idx, f'{tipoMant}_count_{nombreV}']          = len(ewosVentana)
                dfPivot.loc[idx, f'{tipoMant}_duration_total_{nombreV}'] = ewosVentana['duracion_reparacion_min'].sum()
                if nombreV == '24h' and not ewosVentana.empty:
                    dfPivot.loc[idx, f'{tipoMant}_duration_max_{nombreV}'] = ewosVentana['duracion_reparacion_min'].max()

            ewsPrevios = ewosLinea[ewosLinea['timestamp_hora'] < tsActual]
            if not ewsPrevios.empty:
                dfPivot.loc[idx, f'horas_desde_ultimo_{tipoMant}'] = (
                    tsActual - ewsPrevios['timestamp_hora'].max()
                ).total_seconds() / 3600


def _agregarFeaturesMantenimiento(dfPivot, buiPmEwo):
    """Calcula estado EWO, filtra válidas y delega en _calcularFeaturesMantenimiento."""
    log.info("Agregando features de mantenimiento...")

    if 'estado_ewo' not in buiPmEwo.columns:
        def _estadoEwo(row):
            if pd.notna(row.get('fe_cerrar')):           return 'CERRADA'
            if pd.notna(row.get('fe_fin_mto')):          return 'FINALIZADA_PENDIENTE_CIERRE'
            if pd.notna(row.get('fe_fin_reparacion')):   return 'REPARADA'
            if pd.notna(row.get('fe_fin_diagnostico')):  return 'DIAGNOSTICADA'
            if pd.notna(row.get('fe_llegada_mto')):      return 'EN_ATENCION'
            if pd.notna(row.get('fe_aviso_mto')):        return 'AVISADA'
            return 'CREADA_SIN_AVISO'
        buiPmEwo['estado_ewo'] = buiPmEwo.apply(_estadoEwo, axis=1)

    buiPmEwo['duracion_reparacion_min'] = (
        (pd.to_datetime(buiPmEwo['fe_inicio_prod']) - pd.to_datetime(buiPmEwo['fe_inicio_averia']))
        .dt.total_seconds() / 60
    )

    ewosValidas = buiPmEwo[
        (buiPmEwo['fl_borrador'] == 0) &
        (buiPmEwo['estado_ewo'].isin(['CERRADA', 'FINALIZADA_PENDIENTE_CIERRE'])) &
        (buiPmEwo['duracion_reparacion_min'].between(0, 4320, inclusive='both'))
    ].copy()

    log.info(f"  EWOs válidas: {len(ewosValidas):,}")
    ewosPreventivas = ewosValidas[ewosValidas['fl_mantenimiento'] == 1].copy()
    ewosCorrectivas = ewosValidas[ewosValidas['fl_mantenimiento'] == 0].copy()
    log.info(f"    preventivas: {len(ewosPreventivas):,} | correctivas: {len(ewosCorrectivas):,}")

    _calcularFeaturesMantenimiento(dfPivot, ewosPreventivas, 'pm', _VENTANAS_MANT)
    _calcularFeaturesMantenimiento(dfPivot, ewosCorrectivas, 'cm', _VENTANAS_MANT)

    return dfPivot.fillna(0)


def _enriquecerConMetadata(dfPivot, buiLine):
    """Merge con bui_line para añadir id_subcategoria e id_fabrica_area."""
    log.info("Enriqueciendo con metadata de líneas...")
    dfPivot['id_linea'] = dfPivot['id_linea'].astype('int64')
    buiLine['id_linea'] = buiLine['id_linea'].astype('int64')

    dfPivot = dfPivot.merge(
        buiLine[['id_linea', 'id_subcategoria', 'id_fabrica_area']],
        on='id_linea', how='left', validate='m:1'
    )
    dfPivot['id_subcategoria'] = dfPivot['id_subcategoria'].fillna(0).astype('int32')
    dfPivot['id_fabrica_area'] = dfPivot['id_fabrica_area'].fillna(0).astype('int32')
    return dfPivot


def _limpiarDatosAnomalos(df):
    """Recorta valores fisiológicamente imposibles en run_time y total_duration."""
    df['run_time']       = df['run_time'].clip(lower=0, upper=120)
    df['total_duration'] = df['total_duration'].clip(lower=0.1, upper=120)
    df.loc[df['total_duration'] == 0, 'total_duration'] = 60.0
    return df


# CLASE PRINCIPAL ─────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Transforma datos raw de planta en la matriz de features que espera el modelo.
    Requiere preprocessing_artifacts.pkl generado en el entrenamiento para evitar
    training-serving skew en encoders, Z-scores y umbrales de riesgo.
    """

    def __init__(self, config):
        self.config        = config
        self.artifacts     = None
        self.labelEncoders = {}
        self._cargarArtifacts()
        log.info("FeatureEngineer inicializado")

    # CARGA Y VALIDACIÓN DE ARTIFACTS ─────────────────────────────────────────

    def _cargarArtifacts(self):
        # ARTIFACTS OBLIGATORIOS - falla explícitamente si no se encuentran o están incompletos
        rutaBase = Path(self.config.get('model', {}).get('path', '.'))
        candidatos = [
            rutaBase.parent / 'preprocessing_artifacts.pkl',
            Path('.') / 'preprocessing_artifacts.pkl',
            Path(__file__).parent.parent / 'models' / 'preprocessing_artifacts.pkl',
            Path(__file__).parent.parent / 'models' / 'inference_export' / 'preprocessing_artifacts.pkl',
        ]

        rutaArt = next((r for r in candidatos if r.exists()), None)
        if rutaArt is None:
            listaRutas = '\n'.join(f'  - {r}' for r in candidatos)
            raise FileNotFoundError(
                f"preprocessing_artifacts.pkl no encontrado.\n"
                f"Sin él hay riesgo de training-serving skew.\n"
                f"Rutas buscadas:\n{listaRutas}\n"
                f"Actualiza 'model.path' en config.yaml o coloca el archivo en alguna de esas rutas."
            )

        log.info(f"Cargando artifacts: {rutaArt}")
        with open(rutaArt, 'rb') as f:
            self.artifacts = pickle.load(f)

        # VALIDACIÓN DE COMPLETITUD
        requeridos = {
            'version', 'fecha_generacion', 'label_encoders',
            'onehot_columns', 'defaults', 'stats_por_grupo', 'thresholds',
        }
        faltantes = requeridos - set(self.artifacts)
        if faltantes:
            raise ValueError(f"Artifacts incompleto — componentes faltantes: {faltantes}")

        encodersFaltantes = [e for e in ('area', 'fabrica') if e not in self.artifacts['label_encoders']]
        if encodersFaltantes:
            raise ValueError(
                f"LabelEncoders faltantes: {encodersFaltantes}. "
                f"Disponibles: {list(self.artifacts['label_encoders'].keys())}"
            )

        for clave in ('subcategorias', 'subcategoria_columns'):
            if clave not in self.artifacts['onehot_columns']:
                raise ValueError(f"artifacts['onehot_columns'] no contiene '{clave}'")

        leArea    = self.artifacts['label_encoders']['area']
        leFabrica = self.artifacts['label_encoders']['fabrica']
        log.info(
            f"  v{self.artifacts['version']} | {self.artifacts['fecha_generacion']} | "
            f"encoder área: {len(leArea.classes_)} clases | "
            f"encoder fábrica: {len(leFabrica.classes_)} clases | "
            f"subcategorías one-hot: {len(self.artifacts['onehot_columns']['subcategorias'])}"
        )

        self.labelEncoders = self.artifacts['label_encoders']

    # PIPELINE PRINCIPAL ──────────────────────────────────────────────────────

    def transform(self, buiPerdida, buiPmEwo, buiLine):
        """
        Ejecuta el pipeline completo (17 pasos) y retorna la matriz X
        con las últimas 24h de ventanas horarias lista para predict().
        """
        log.info("FEATURE ENGINEERING — INICIO")

        df = _pivotarEventosPorMaquinaHora(buiPerdida)           # 1
        df = _agregarFeaturesMantenimiento(df, buiPmEwo)          # 2
        df = _enriquecerConMetadata(df, buiLine)                  # 3
        df = _limpiarDatosAnomalos(df)                            # 4
        df = self._featuresEventosDetallados(df)                  # 5
        df = self._metricasAgregadas(df)                          # 6
        df = self._featuresMantenimientoMejoradas(df)             # 7
        df = self._rollingWindows(df)                             # 8
        df = self._tendencias(df)                                 # 9
        df = self._ratios(df)                                     # 10
        df = self._featuresTemporales(df)                         # 11
        df = self._variabilidad(df)                               # 12
        df = self._encodingCategorias(df)                         # 13
        df = self._featuresAnomalias(df)                          # 14
        df = self._comparacionPares(df)                           # 15
        df = self._limpiezaFinal(df)                              # 16
        df = self._filtrarUltimas24h(df)                          # 17

        log.info(f"FEATURE ENGINEERING — FIN | shape={df.shape}")
        return df

    # PASOS 5-12 ──────────────────────────────────────────────────────────────

    def _featuresEventosDetallados(self, df):
        log.info("[5] Features detalladas por evento...")
        for nombreEvt, (colDur, colCnt) in _EVENTOS_DETALLADOS.items():
            df[f'{nombreEvt}_duracion_avg']          = np.where(df[colCnt] > 0, df[colDur] / df[colCnt], 0)
            df[f'{nombreEvt}_tiempo_pct']            = (df[colDur] / df['total_duration'] * 100).clip(upper=100)
            df[f'{nombreEvt}_frecuencia_por_hora']   = np.where(df['total_duration'] > 0,
                                                                 df[colCnt] / (df['total_duration'] / 60), 0)
            df[f'{nombreEvt}_severity_score']        = df[colCnt] * df[f'{nombreEvt}_duracion_avg']

        df['minor_stoppages_freq_normalizada'] = np.where(
            df['run_time'] > 0, df['minor_stoppages_count'] / (df['run_time'] / 60), 0
        )
        return df

    def _metricasAgregadas(self, df):
        log.info("[6] Métricas agregadas...")
        colsDurPerdida = [c for c in df.columns if '_duration' in c and
                          any(x in c for x in ('loss', 'defect', 'failure', 'stoppage', 'idle'))]
        df['total_perdidas_duracion']    = df[colsDurPerdida].sum(axis=1)
        df['total_perdidas_tiempo_pct']  = (df['total_perdidas_duracion'] / df['total_duration'] * 100).clip(upper=100)

        colsCount = [c for c in df.columns if c.endswith('_count') and
                     not c.startswith(('run', 'mpl', 'ucl', 'pdl', 'desconocido', 'total'))]
        df['diversidad_eventos'] = (df[colsCount] > 0).sum(axis=1)

        colsDur = [c for c in df.columns if c.endswith('_duration') and
                   not c.startswith(('total', 'run', 'mpl', 'ucl', 'pdl'))]
        df['evento_mas_largo_duracion']   = df[colsDur].max(axis=1)
        df['evento_mas_frecuente_count']  = df[colsCount].max(axis=1)

        colsCriticos = [
            'breakdown_mechanical_duration', 'breakdown_electrical_duration',
            'breakdown_ic_duration', 'process_failure_time_duration',
        ]
        df['duracion_eventos_criticos']        = df[colsCriticos].sum(axis=1)
        df['ratio_eventos_criticos_total']     = np.where(
            df['total_perdidas_duracion'] > 0,
            df['duracion_eventos_criticos'] / df['total_perdidas_duracion'], 0
        )
        return df

    def _featuresMantenimientoMejoradas(self, df):
        log.info("[7] Features de mantenimiento mejoradas...")
        df['dias_desde_ultimo_pm']          = df['horas_desde_ultimo_pm'] / 24
        df['dias_desde_ultimo_cm']          = df['horas_desde_ultimo_cm'] / 24
        df['pm_frecuencia_mensual']         = df['pm_count_30d'] / 30
        df['cm_tasa_fallas_mensual']        = df['cm_count_30d'] / 30
        df['mantenimiento_total_duracion_7d'] = df['pm_duration_total_7d'] + df['cm_duration_total_7d']
        return df

    def _rollingWindows(self, df):
        # ROLLING WINDOWS - acumula eventos en ventanas de 2/6/12/24h por máquina
        log.info("[8] Rolling windows (2h, 6h, 12h, 24h)...")
        df = df.sort_values(['id_maquina_dfos', 'timestamp_hora']).reset_index(drop=True)

        for nombreV, horas in {'2h': 2, '6h': 6, '12h': 12, '24h': 24}.items():
            for evento in _EVENTOS_ROLLING:
                corto   = (evento.replace('_duration', '').replace('_count', '')
                                 .replace('_time', '').replace('_and', '').replace('__', '_'))
                colRoll = f'{corto}_rolling_{nombreV}'
                df[colRoll]              = df.groupby('id_maquina_dfos')[evento].transform(
                    lambda x: x.rolling(window=horas, min_periods=1).sum()
                )
                df[f'{colRoll}_avg']     = df[colRoll] / horas
            gc.collect()
        return df

    def _tendencias(self, df):
        log.info("[9] Tendencias y aceleraciones...")
        for evento in ('minor_stoppages', 'speed_loss', 'process_failure', 'breakdown_equipment_failure'):
            col12h = f'{evento}_rolling_12h'
            if col12h not in df.columns:
                log.warning(f"  Columna {col12h} no encontrada")
                continue
            prev12h = df.groupby('id_maquina_dfos')[col12h].shift(12)
            df[f'{evento}_trend_24h'] = df[col12h] - prev12h.fillna(0)
            df[f'{evento}_accel_24h'] = np.where(
                prev12h > 0,
                ((df[col12h] - prev12h) / prev12h * 100), 0
            ).clip(-500, 500)
        return df

    def _ratios(self, df):
        log.info("[10] Ratios...")
        tiempoOp = (df['run_time'] + df['mpl_time'] + df['ucl_time'] + df['pdl_time']).clip(lower=0.1)
        df['ratio_run_total']       = (df['run_time'] / tiempoOp * 100).clip(0, 100)
        df['ratio_mpl_total']       = (df['mpl_time'] / tiempoOp * 100).clip(0, 100)
        df['ratio_ucl_total']       = (df['ucl_time'] / tiempoOp * 100).clip(0, 100)
        df['ratio_unplanned_total'] = ((df['mpl_time'] + df['ucl_time']) / tiempoOp * 100).clip(0, 100)

        df['ratio_minor_vs_critical']    = np.where(
            df['breakdown_and_equipment_failure_time_count'] > 0,
            df['minor_stoppages_count'] / df['breakdown_and_equipment_failure_time_count'], 0
        )
        df['ratio_quality_vs_performance'] = np.where(
            df['speed_loss_duration'] > 0,
            df['quality_defect_time_loss_duration'] / df['speed_loss_duration'], 0
        )
        df['events_per_hour']       = df['total_events'] / (df['total_duration'] / 60)
        df['avg_duration_per_event'] = np.where(df['total_events'] > 0,
                                                 df['total_duration'] / df['total_events'], 0)
        return df

    def _featuresTemporales(self, df):
        log.info("[11] Features temporales...")
        df['hora_del_dia']    = df['timestamp_hora'].dt.hour
        df['dia_semana']      = df['timestamp_hora'].dt.dayofweek
        df['es_fin_semana']   = (df['dia_semana'] >= 5).astype(int)
        df['es_turno_noche']  = ((df['hora_del_dia'] >= 22) | (df['hora_del_dia'] < 6)).astype(int)
        df['es_turno_tarde']  = ((df['hora_del_dia'] >= 14) & (df['hora_del_dia'] < 22)).astype(int)
        df['es_turno_mañana'] = ((df['hora_del_dia'] >= 6)  & (df['hora_del_dia'] < 14)).astype(int)
        df['es_inicio_turno'] = df['hora_del_dia'].isin([6, 14, 22]).astype(int)
        df['es_fin_turno']    = df['hora_del_dia'].isin([5, 13, 21]).astype(int)
        df['hora_sin']        = np.sin(2 * np.pi * df['hora_del_dia'] / 24)
        df['hora_cos']        = np.cos(2 * np.pi * df['hora_del_dia'] / 24)
        df['dia_sin']         = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_cos']         = np.cos(2 * np.pi * df['dia_semana'] / 7)
        return df

    def _variabilidad(self, df):
        # VARIABILIDAD 7D - std y CV sobre ventana de 168h; también días desde último breakdown
        log.info("[12] Variabilidad (7 días)...")
        for evento in _EVENTOS_VARIABILIDAD:
            corto    = (evento.replace('_duration', '').replace('_count', '')
                              .replace('_time', '').replace('_and', '').replace('__', '_'))
            rollMean = df.groupby('id_maquina_dfos')[evento].transform(
                lambda x: x.rolling(window=168, min_periods=24).mean()
            )
            df[f'{corto}_std_7d'] = df.groupby('id_maquina_dfos')[evento].transform(
                lambda x: x.rolling(window=168, min_periods=24).std()
            ).fillna(0)
            df[f'{corto}_cv_7d'] = np.where(rollMean > 0,
                                             df[f'{corto}_std_7d'] / rollMean, 0).clip(0, 10)

        log.info("  Calculando días desde último breakdown...")
        df['horas_desde_ultimo_breakdown'] = 9999.0
        maskBD = df['breakdown_and_equipment_failure_time_count'] > 0

        for maquina in tqdm(df['id_maquina_dfos'].unique(), desc="Máquinas (breakdown)"):
            maskMaq = df['id_maquina_dfos'] == maquina
            tsBDs   = df.loc[maskMaq & maskBD, 'timestamp_hora'].values
            if not len(tsBDs):
                continue
            for idx in df[maskMaq].index:
                tsActual = df.loc[idx, 'timestamp_hora']
                previos  = tsBDs[tsBDs < tsActual]
                if len(previos):
                    df.loc[idx, 'horas_desde_ultimo_breakdown'] = (
                        tsActual - previos.max()
                    ).total_seconds() / 3600

        df['dias_desde_ultimo_breakdown'] = df['horas_desde_ultimo_breakdown'] / 24
        return df

    # PASOS 13-17 (con artifacts) ─────────────────────────────────────────────

    def _encodingCategorias(self, df):
        # ONE-HOT FIJO - usa exactamente las subcategorías vistas en entrenamiento
        log.info("[13] Encoding de categorías (artifacts)...")

        subcatsEsperadas  = self.artifacts['onehot_columns']['subcategorias']
        subcatsEnDatos    = set(df['id_subcategoria'].unique())
        subcatsNuevas     = subcatsEnDatos - set(subcatsEsperadas)
        if subcatsNuevas:
            log.warning(f"  Subcategorías nuevas (se ignorarán): {sorted(subcatsNuevas)[:10]}")

        for subcat in subcatsEsperadas:
            df[f'subcategoria_{int(subcat)}'] = 0
        for subcat in subcatsEnDatos:
            if subcat in subcatsEsperadas:
                df.loc[df['id_subcategoria'] == subcat, f'subcategoria_{int(subcat)}'] = 1

        subcatCols  = [f'subcategoria_{int(s)}' for s in subcatsEsperadas]
        sumaOnehot  = df[subcatCols].sum(axis=1)
        if (sumaOnehot > 1).any():
            raise ValueError("One-hot encoding incoherente: hay filas con múltiples subcategorías activas")
        log.info(f"  {len(subcatsEsperadas)} columnas one-hot | "
                 f"activas: {(sumaOnehot == 1).sum():,} | vacías: {(sumaOnehot == 0).sum():,}")

        # LABEL ENCODING ÁREA y FÁBRICA - unseen values → código por defecto
        for campo, leKey, defaultKey, colDest in (
            ('id_fabrica_area', 'area',    'area_default_code',    'id_fabrica_area_encoded'),
            ('id_fabrica',      'fabrica', 'fabrica_default_code', 'id_fabrica_encoded'),
        ):
            le      = self.labelEncoders[leKey]
            default = self.artifacts['defaults'][defaultKey]
            known   = set(le.classes_)
            valores = df[campo].astype(str).values

            noVistos = set(valores) - known
            if noVistos:
                log.warning(f"  {campo}: {len(noVistos)} valores no vistos → default={default}")

            df[colDest] = [le.transform([v])[0] if v in known else default for v in valores]
            nConocidos  = sum(1 for v in valores if str(v) in known)
            log.info(f"  {colDest}: {nConocidos:,}/{len(valores):,} conocidos | "
                     f"{df[colDest].nunique()} únicos")

        return df

    def _featuresAnomalias(self, df):
        # Z-SCORES con estadísticas del entrenamiento - máquinas nuevas usan media/std global
        log.info("[14] Features de anomalías (artifacts)...")
        statsZscore = self.artifacts['stats_por_grupo']['zscore']

        for colEvento, eventoCorto in _EVENTOS_ZSCORE.items():
            if colEvento not in df.columns or eventoCorto not in statsZscore:
                continue

            stats         = statsZscore[eventoCorto]
            statsPorMaq   = stats['por_maquina']
            mediaGlobal   = stats['global_mean']
            stdGlobal     = stats['global_std']

            zscores = []
            for _, fila in df.iterrows():
                statsMaq = statsPorMaq.get(fila['id_maquina_dfos'],
                                           {'mean': mediaGlobal, 'std': stdGlobal})
                std = statsMaq['std']
                z   = (fila[colEvento] - statsMaq['mean']) / std if std > 0 else 0
                zscores.append(np.clip(z, -10, 10))

            df[f'{eventoCorto}_zscore']    = zscores
            df[f'{eventoCorto}_is_outlier'] = (np.abs(df[f'{eventoCorto}_zscore']) > 3).astype(int)

        outliersCol = [c for c in df.columns if c.endswith('_is_outlier')]
        df['anomaly_score_composite'] = df[outliersCol].sum(axis=1)
        return df

    def _comparacionPares(self, df):
        # COMPARACIÓN CON PARES - desviación vs media de subcategoría en entrenamiento
        log.info("[15] Comparación con pares (artifacts)...")
        statsPares = self.artifacts['stats_por_grupo']['pares']

        for colEvento, eventoCorto in _EVENTOS_PARES.items():
            if colEvento not in df.columns or eventoCorto not in statsPares:
                continue
            mediaPorSubcat = statsPares[eventoCorto]['media_por_subcategoria']
            mediaGlobal    = statsPares[eventoCorto]['media_global']

            df[f'avg_{eventoCorto}_subcategoria']  = df['id_subcategoria'].map(
                lambda x: mediaPorSubcat.get(x, mediaGlobal)
            )
            df[f'diff_{eventoCorto}_vs_subcategoria'] = (
                df[colEvento] - df[f'avg_{eventoCorto}_subcategoria']
            )

        # ESTRÉS POR ÁREA (24h rolling)
        colBD     = 'breakdown_and_equipment_failure_time_count'
        evtCorto  = 'breakdown_equipment_failure'
        df[f'concurrent_{evtCorto}_area_24h']  = df.groupby('id_fabrica_area')[colBD].transform(
            lambda x: x.rolling(window=24, min_periods=1).sum()
        )
        df['area_stress_level_24h'] = df.groupby('id_fabrica_area')['total_events'].transform(
            lambda x: x.rolling(window=24, min_periods=1).sum()
        )

        # FLAG ALTO RIESGO - threshold calculado en entrenamiento
        threshold      = self.artifacts['thresholds'].get('high_risk_threshold', 0.1)
        tasasPorSubcat = self.artifacts['thresholds'].get('breakdown_rate_by_subcategoria', {})
        log.info(f"  Threshold alto riesgo: {threshold:.6f}")
        df['is_high_risk_subcategoria'] = (
            df['id_subcategoria'].map(lambda x: tasasPorSubcat.get(x, 0)) > threshold
        ).astype(int)

        return df

    def _limpiezaFinal(self, df):
        log.info("[16] Limpieza final...")
        df = df.fillna(0)

        for col in (c for c in df.columns if c.startswith(('es_', 'is_', 'tiene_', 'subcategoria_'))):
            df[col] = df[col].astype('int8')
        for col in (c for c in df.columns if '_count' in c and df[c].dtype == 'float64'):
            df[col] = df[col].astype('int32')
        for col in df.select_dtypes(include='float64').columns:
            df[col] = df[col].astype('float32')

        df = df.drop(columns=[c for c in ('id_subcategoria', 'id_fabrica_area', 'id_fabrica')
                               if c in df.columns])
        return df

    def _filtrarUltimas24h(self, df):
        # FILTRO TEMPORAL - solo las ventanas del último día son relevantes para predicción
        log.info("[17] Filtrando últimas 24h...")
        fechaCorte  = df['timestamp_hora'].max() - pd.Timedelta(hours=24)
        dfPred      = df[df['timestamp_hora'] > fechaCorte].copy()
        log.info(f"  {len(df):,} → {len(dfPred):,} ventanas "
                 f"| {dfPred['timestamp_hora'].min()} → {dfPred['timestamp_hora'].max()}")
        return dfPred