"""
predictor.py - Inferencia con modelo LightGBM.
Carga modelo, valida features y clasifica el riesgo de falla por ventana horaria.
"""

import json
import pickle
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

log = logging.getLogger(__name__)


class Predictor:

    def __init__(self, config):
        self.config          = config
        self.modelo          = None
        self.featuresReq     = None
        self.configInferencia = None

        self._cargarModelo()
        self._cargarFeatures()
        self._cargarConfigInferencia()

        log.info("Predictor inicializado")

    # CARGA DE ARTEFACTOS ─────────────────────────────────────────────────────

    def _cargarModelo(self):
        # CARGA MODELO - soporta .pkl (pickle) y .txt/.json (booster nativo)
        rutaModelo = Path(self.config['model']['path'])
        if not rutaModelo.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {rutaModelo}")

        log.info(f"Cargando modelo: {rutaModelo}  (LightGBM {lgb.__version__})")

        try:
            if rutaModelo.suffix == '.pkl':
                with open(rutaModelo, 'rb') as f:
                    self.modelo = pickle.load(f)
            elif rutaModelo.suffix in ('.txt', '.json'):
                self.modelo = lgb.Booster(model_file=str(rutaModelo))
            else:
                raise ValueError(f"Formato no soportado: {rutaModelo.suffix}")
        except (pickle.UnpicklingError, Exception) as e:
            raise RuntimeError(
                f"Error cargando modelo ({rutaModelo.suffix}): {e}\n"
                f"Si es incompatibilidad de numpy, ejecuta: pip install --upgrade numpy\n"
                f"O re-exporta el modelo en el mismo entorno de inferencia."
            ) from e

        log.info(f"  v{self.config['model']['version']} | {self.modelo.num_trees()} árboles")

    def _cargarFeatures(self):
        # FEATURES REQUERIDAS - el orden debe coincidir EXACTAMENTE con el entrenamiento
        rutaFeatures = Path(self.config['model']['path']).parent / 'features_utiles.json'
        if not rutaFeatures.exists():
            raise FileNotFoundError(f"features_utiles.json no encontrado: {rutaFeatures}")

        with open(rutaFeatures, 'r') as f:
            self.featuresReq = json.load(f)['features']

        log.info(f"{len(self.featuresReq)} features requeridas cargadas")
        self._validarOrdenFeatures()

    def _validarOrdenFeatures(self):
        # VALIDACIÓN DE ORDEN - LightGBM es sensible al orden exacto de columnas
        if self.modelo is None:
            return
        try:
            nombresModelo = self.modelo.feature_name()
        except AttributeError:
            log.warning("El modelo no expone feature_name(); validación omitida")
            return

        if len(nombresModelo) != len(self.featuresReq):
            raise ValueError(
                f"Número de features no coincide: modelo={len(nombresModelo)}, "
                f"features_utiles.json={len(self.featuresReq)}"
            )

        for i, (mf, lf) in enumerate(zip(nombresModelo, self.featuresReq)):
            if mf != lf:
                raise ValueError(
                    f"Orden incorrecto en posición {i}: modelo='{mf}', json='{lf}'\n"
                    f"Usa modelo.feature_name() para obtener el orden exacto del entrenamiento."
                )

        log.info(f"  Orden validado | primeras: {nombresModelo[:3]} | últimas: {nombresModelo[-3:]}")

    def _cargarConfigInferencia(self):
        # CONFIG INFERENCIA - normaliza dos formatos históricos a estructura 'thresholds'
        rutaConfig = Path(self.config['model']['path']).parent / 'inference_config.json'
        if not rutaConfig.exists():
            raise FileNotFoundError(f"inference_config.json no encontrado: {rutaConfig}")

        with open(rutaConfig, 'r') as f:
            rawConfig = json.load(f)

        if 'thresholds' in rawConfig:
            self.configInferencia = rawConfig
        elif 'umbrales' in rawConfig and 'alertas' in rawConfig['umbrales']:
            # FORMATO NUEVO → normalización interna
            self.configInferencia = {
                'thresholds':          rawConfig['umbrales']['alertas'],
                'umbral_produccion':   rawConfig['umbrales'].get('produccion', 0.31),
                'metricas_referencia': rawConfig.get('metricas_referencia', {}),
                'version':             rawConfig.get('version', 'unknown'),
                'modelo':              rawConfig.get('modelo', {}),
            }
        else:
            raise ValueError(
                f"Formato de inference_config.json no reconocido. "
                f"Se esperaba 'thresholds' o 'umbrales.alertas'. "
                f"Claves encontradas: {list(rawConfig.keys())}"
            )

        th = self.configInferencia['thresholds']
        log.info(f"Umbrales — crítico: {th['critical']} | moderado: {th['moderate']} | bajo: {th['low']}")

    # INFERENCIA ──────────────────────────────────────────────────────────────

    def _prepararMatriz(self, dfFeatures):
        # SELECCIÓN Y VALIDACIÓN DE FEATURES - falla si falta alguna requerida
        faltantes = set(self.featuresReq) - set(dfFeatures.columns)
        if faltantes:
            raise ValueError(f"Faltan {len(faltantes)} features requeridas: {faltantes}")

        X = dfFeatures[self.featuresReq].copy()

        if X.isna().any().any():
            log.warning("NaN detectados en features — rellenando con 0")
            X = X.fillna(0)
        if np.isinf(X.values).any():
            log.warning("Inf detectados en features — reemplazando con 0")
            X = X.replace([np.inf, -np.inf], 0)

        return X

    def _clasificarNivelAlerta(self, scores):
        th     = self.configInferencia['thresholds']
        niveles = np.full(len(scores), 'normal', dtype=object)
        niveles[scores >= th['critical']] = 'critico'
        niveles[(scores >= th['moderate']) & (scores < th['critical'])] = 'moderado'
        niveles[(scores >= th['low'])      & (scores < th['moderate'])] = 'bajo'
        return niveles

    def predict(self, dfFeatures):
        """
        Genera scores de riesgo y nivel de alerta para cada ventana horaria.
        Retorna DataFrame con columnas: timestamp_ventana, timestamp_hora,
        id_maquina_dfos, id_linea, score, nivel_alerta.
        """
        if dfFeatures.empty:
            log.warning("DataFrame vacío, sin ventanas que predecir")
            return pd.DataFrame()

        identificadores = dfFeatures[['id_maquina_dfos', 'id_linea', 'timestamp_hora']].copy()
        X = self._prepararMatriz(dfFeatures)

        log.info(f"Ejecutando modelo | {len(X):,} ventanas | {len(X.columns)} features")
        scores = np.clip(np.array(self.modelo.predict(X)), 0, 1)

        log.info(f"  min={scores.min():.4f}  max={scores.max():.4f}  "
                 f"media={scores.mean():.4f}  mediana={np.median(scores):.4f}")

        niveles = self._clasificarNivelAlerta(scores)

        # DISTRIBUCIÓN DE ALERTAS
        for nivel, n in zip(*np.unique(niveles, return_counts=True)):
            log.info(f"  {nivel.upper()}: {n:,} ({n / len(scores) * 100:.1f}%)")

        resultados = identificadores.copy()
        resultados['score']            = scores
        resultados['nivel_alerta']     = niveles
        resultados['timestamp_ventana'] = resultados['timestamp_hora'] + pd.Timedelta(hours=1)

        resultados = resultados[[
            'timestamp_ventana', 'timestamp_hora',
            'id_maquina_dfos', 'id_linea',
            'score', 'nivel_alerta'
        ]]

        # ALERTAS CRÍTICAS - log detallado si las hay
        criticas = resultados[resultados['nivel_alerta'] == 'critico']
        if not criticas.empty:
            log.warning(f"ALERTAS CRÍTICAS: {len(criticas)}")
            for _, fila in criticas.head(10).iterrows():
                log.warning(f"  máquina={fila['id_maquina_dfos']} | score={fila['score']:.4f} "
                             f"| ventana={fila['timestamp_ventana']}")
            if len(criticas) > 10:
                log.warning(f"  ... y {len(criticas) - 10} más")

        return resultados

    def infoModelo(self):
        return {
            'version':      self.config['model']['version'],
            'n_features':   len(self.featuresReq),
            'features':     self.featuresReq,
            'thresholds':   self.configInferencia['thresholds'],
            'n_arboles':    self.modelo.num_trees() if self.modelo else None,
        }