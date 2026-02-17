"""
Predictor - Sistema de Mantenimiento Predictivo
Carga el modelo LightGBM y genera predicciones con niveles de alerta y causas raíz (SHAP)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import logging
import shap
from pathlib import Path

log = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.features_necesarias = None
        self.inference_config = None
        
        self._load_model()
        self._load_features()
        self._load_inference_config()
        
        log.info(" Predictor inicializado correctamente")

    def _load_model(self):
        """Carga el modelo LightGBM (soporta .pkl, .txt, .json)"""
        import pickle
        model_path = Path(self.config['model']['path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

        if model_path.suffix == '.pkl':
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = lgb.Booster(model_file=str(model_path))
        log.info(f" Modelo cargado exitosamente desde {model_path.suffix}")

    def _load_features(self):
        """Carga la lista de features y valida el orden"""
        features_path = Path(self.config['model']['path']).parent / 'features_utiles.json'
        with open(features_path, 'r') as f:
            features_data = json.load(f)
        self.features_necesarias = features_data['features']
        self._validate_feature_order()

    def _validate_feature_order(self):
        """Valida que las features coincidan con el modelo"""
        if self.model and hasattr(self.model, 'feature_name'):
            model_features = self.model.feature_name()
            if model_features != self.features_necesarias:
                log.error("¡Orden de features incorrecto!")
                # Aquí podrías lanzar una excepción si prefieres bloquear la ejecución

    def _load_inference_config(self):
        """Carga umbrales de alerta"""
        inference_path = Path(self.config['model']['path']).parent / 'inference_config.json'
        with open(inference_path, 'r') as f:
            config_raw = json.load(f)
        
        if 'thresholds' in config_raw:
            self.inference_config = config_raw
        else:
            self.inference_config = {'thresholds': config_raw['umbrales']['alertas']}

    def predict(self, df_features):
        """
        Genera predicciones y explica las causas raíz con SHAP
        """
        log.info("\n" + "=" * 80)
        log.info(" GENERANDO PREDICCIONES CON EXPLICABILIDAD SHAP")
        log.info("=" * 80)
        
        if len(df_features) == 0:
            return pd.DataFrame()
        
        # 1. Preparar datos y validar
        identificadores = df_features[['id_maquina_dfos', 'id_linea', 'timestamp_hora']].copy()
        X_input = df_features[self.features_necesarias].fillna(0).replace([np.inf, -np.inf], 0)
        
        # 2. Ejecutar predicción
        log.info(" Ejecutando modelo LightGBM...")
        scores = self.model.predict(X_input)
        scores = np.clip(scores, 0, 1) # Asegurar rango 0-1
        
        # 3. Calcular Explicaciones SHAP (Causas Raíz)
        log.info(" Calculando causas raíz con SHAP...")
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_input.values)
            
            # Para LightGBM binario, shap_values puede ser una lista [clase_0, clase_1]
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

            explicaciones = []
            for i in range(len(X_input)):
                importancias = dict(zip(self.features_necesarias, shap_vals[i]))
                # Top 3 causas que SUMAN al riesgo
                top_causas = sorted([item for item in importancias.items() if item[1] > 0], 
                                   key=lambda x: x[1], reverse=True)[:3]
                
                texto = " | ".join([c[0] for c in top_causas])
                explicaciones.append(texto if texto else "Variables estables")
        except Exception as e:
            log.error(f" Error en SHAP: {e}")
            explicaciones = ["Error en análisis"] * len(X_input)

        # 4. Clasificar alertas
        niveles_alerta = self._classify_alerts(scores)
        
        # 5. Consolidar resultados
        resultados = identificadores.copy()
        resultados['score'] = scores
        resultados['nivel_alerta'] = niveles_alerta
        resultados['de_causas_raiz'] = explicaciones
        
        # Hora que estamos prediciendo
        resultados['timestamp_ventana'] = resultados['timestamp_hora'] + pd.Timedelta(hours=1)
        
        return resultados[['timestamp_ventana', 'id_maquina_dfos', 'id_linea', 
                           'score', 'nivel_alerta', 'de_causas_raiz']]

    def _classify_alerts(self, scores):
        """Clasifica los scores según los umbrales de config"""
        t = self.inference_config['thresholds']
        niveles = np.full(len(scores), 'normal', dtype=object)
        niveles[scores >= t['low']] = 'bajo'
        niveles[scores >= t['moderate']] = 'moderado'
        niveles[scores >= t['critical']] = 'critico'
        return niveles

    def get_model_info(self):
        """Retorna info técnica del modelo"""
        return {
            'version': self.config['model']['version'],
            'num_features': len(self.features_necesarias),
            'thresholds': self.inference_config['thresholds'],
            'num_trees': self.model.num_trees() if self.model else None
        }