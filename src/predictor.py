"""
Predictor - Sistema de Mantenimiento Predictivo
Carga el modelo LightGBM y genera predicciones con niveles de alerta
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ============================================================================
# CLASE PREDICTOR
# ============================================================================

class Predictor:
    """
    Clase para cargar el modelo y generar predicciones de fallas
    """
    
    def __init__(self, config):
        """
        Inicializa el Predictor cargando modelo y configuraciones
        
        Args:
            config: dict con configuración del sistema
        """
        self.config = config
        self.model = None
        self.features_necesarias = None
        self.inference_config = None
        
        # Cargar modelo y configuraciones
        self._load_model()
        self._load_features()
        self._load_inference_config()
        
        log.info(" Predictor inicializado correctamente")
    
    
    def _load_model(self):
        """
        Carga el modelo LightGBM entrenado (versión corregida)
        """
        import pickle
        model_path = Path(self.config['model']['path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        log.info(f" Cargando modelo desde: {model_path}")
        log.info(f"   Versión de LightGBM: {lgb.__version__}")
        
        # DETECTAR FORMATO POR EXTENSIÓN
        if model_path.suffix == '.pkl':
            # FORMATO PICKLE
            log.info("   Formato detectado: PICKLE (.pkl)")
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                log.info(f" Modelo cargado exitosamente desde pickle")
                log.info(f"   Versión: {self.config['model']['version']}")
                log.info(f"   Número de árboles: {self.model.num_trees()}")
                return
            except Exception as e:
                log.error(f" Error cargando pickle: {e}")
                raise RuntimeError(
                    f"Error cargando modelo pickle: {e}\n\n"
                    f"Posible incompatibilidad de numpy. Ejecuta:\n"
                    f"  pip install --upgrade numpy\n"
                    f"O re-exporta el modelo en el mismo entorno donde lo usarás."
                )
        
        elif model_path.suffix in ['.txt', '.json']:
            # FORMATO TEXTO O JSON
            log.info(f"   Formato detectado: {model_path.suffix.upper()}")
            try:
                self.model = lgb.Booster(model_file=str(model_path))
                log.info(f" Modelo cargado exitosamente")
                log.info(f"   Versión: {self.config['model']['version']}")
                log.info(f"   Número de árboles: {self.model.num_trees()}")
                return
            except Exception as e:
                log.error(f" Error cargando modelo: {e}")
                raise RuntimeError(
                    f"Error cargando modelo {model_path.suffix}: {e}\n\n"
                    f"Intenta usar formato .pkl en su lugar."
                )
        
        else:
            raise ValueError(f"Formato de modelo no soportado: {model_path.suffix}")
    
    
    def _load_features(self):
        """
        Carga la lista de features necesarias para el modelo EN EL ORDEN CORRECTO

        CRÍTICO: En LightGBM, el orden de las features debe coincidir EXACTAMENTE
        con el orden en que fueron entrenadas. Si el orden es incorrecto, las
        predicciones serán completamente erróneas.
        """
        features_path = Path(self.config['model']['path']).parent / 'features_utiles.json'

        if not features_path.exists():
            raise FileNotFoundError(f"Archivo de features no encontrado en: {features_path}")

        log.info(f" Cargando features desde: {features_path}")

        with open(features_path, 'r') as f:
            features_data = json.load(f)

        self.features_necesarias = features_data['features']
        log.info(f" {len(self.features_necesarias)} features requeridas cargadas")

        # VALIDAR ORDEN DE FEATURES CON EL MODELO
        self._validate_feature_order()


    def _validate_feature_order(self):
        """
        Valida que el orden de features coincida EXACTAMENTE con el modelo LightGBM

        CRÍTICO: LightGBM espera las features en el orden exacto del entrenamiento.
        Esta validación previene errores silenciosos de predicción incorrecta.
        """
        if self.model is None:
            log.warning(" Modelo no cargado aún, saltando validación de orden de features")
            return

        try:
            # Obtener el orden de features del modelo LightGBM
            model_feature_names = self.model.feature_name()

            # Comparar con las features cargadas
            if len(model_feature_names) != len(self.features_necesarias):
                raise ValueError(
                    f" ERROR CRÍTICO: Número de features no coincide!\n"
                    f"   Modelo espera: {len(model_feature_names)} features\n"
                    f"   features_utiles.json tiene: {len(self.features_necesarias)} features"
                )

            # Verificar orden EXACTO
            for i, (model_feat, loaded_feat) in enumerate(zip(model_feature_names, self.features_necesarias)):
                if model_feat != loaded_feat:
                    raise ValueError(
                        f" ERROR CRÍTICO: Orden de features INCORRECTO en posición {i}!\n"
                        f"   Modelo espera: '{model_feat}'\n"
                        f"   features_utiles.json tiene: '{loaded_feat}'\n\n"
                        f"   Las features deben estar en el ORDEN EXACTO del entrenamiento.\n"
                        f"   Usa modelo.feature_name() para obtener el orden correcto."
                    )

            log.info(" Orden de features validado correctamente con el modelo")
            log.info(f"   Primeras 3 features: {model_feature_names[:3]}")
            log.info(f"   Últimas 3 features: {model_feature_names[-3:]}")

        except AttributeError:
            log.warning(" El modelo no tiene feature_name(), saltando validación de orden")


    def _load_inference_config(self):
        """
        Carga configuración de inferencia (umbrales, etc.)
        Soporta dos formatos:
        1. Formato antiguo: {"thresholds": {...}}
        2. Formato nuevo: {"umbrales": {"alertas": {...}}}
        """
        inference_path = Path(self.config['model']['path']).parent / 'inference_config.json'
        
        if not inference_path.exists():
            raise FileNotFoundError(f"Config de inferencia no encontrado en: {inference_path}")
        
        log.info(f" Cargando config de inferencia desde: {inference_path}")
        
        with open(inference_path, 'r') as f:
            config_raw = json.load(f)
        
        # Normalizar estructura: siempre usar 'thresholds' internamente
        if 'thresholds' in config_raw:
            # Formato antiguo - usar directamente
            self.inference_config = config_raw
        elif 'umbrales' in config_raw and 'alertas' in config_raw['umbrales']:
            # Formato nuevo - normalizar a formato antiguo
            self.inference_config = {
                'thresholds': config_raw['umbrales']['alertas'],
                'umbral_produccion': config_raw['umbrales'].get('produccion', 0.31),
                'metricas_referencia': config_raw.get('metricas_referencia', {}),
                'version': config_raw.get('version', 'unknown'),
                'modelo': config_raw.get('modelo', {})
            }
            log.info(f"   Formato nuevo detectado - normalizado a 'thresholds'")
        else:
            raise ValueError(
                f"Formato de inference_config.json no reconocido. "
                f"Se esperaba 'thresholds' o 'umbrales.alertas'. "
                f"Claves encontradas: {list(config_raw.keys())}"
            )
        
        log.info(" Configuración de inferencia cargada:")
        log.info(f"   Umbral crítico: {self.inference_config['thresholds']['critical']}")
        log.info(f"   Umbral moderado: {self.inference_config['thresholds']['moderate']}")
        log.info(f"   Umbral bajo: {self.inference_config['thresholds']['low']}")
    
    
    def _validate_features(self, df):
        """
        Valida que el DataFrame tenga todas las features necesarias
        
        Args:
            df: DataFrame con features
            
        Returns:
            DataFrame con solo las features necesarias en el orden correcto
        """
        log.info("\n Validando features...")
        
        # Verificar features faltantes
        features_faltantes = set(self.features_necesarias) - set(df.columns)
        
        if features_faltantes:
            log.error(f" Features faltantes: {features_faltantes}")
            raise ValueError(f"Faltan {len(features_faltantes)} features requeridas")
        
        # Verificar features extra (informativo)
        features_extra = set(df.columns) - set(self.features_necesarias) - {
            'id_maquina_dfos', 'id_linea', 'timestamp_hora'
        }
        
        if features_extra:
            log.debug(f" Features extra (serán ignoradas): {len(features_extra)}")
        
        # Seleccionar solo features necesarias en el orden correcto
        X = df[self.features_necesarias].copy()
        
        log.info(f" Validación exitosa: {X.shape}")
        log.info(f"   Ventanas: {len(X):,}")
        log.info(f"   Features: {len(X.columns)}")
        
        return X
    
    
    def _classify_alerts(self, scores):
        """
        Clasifica scores en niveles de alerta
        
        Args:
            scores: array con scores de predicción (0-1)
            
        Returns:
            array con niveles de alerta
        """
        thresholds = self.inference_config['thresholds']
        
        # Inicializar con 'normal'
        niveles = np.full(len(scores), 'normal', dtype=object)
        
        # Clasificar según umbrales (de mayor a menor)
        niveles[scores >= thresholds['critical']] = 'critico'
        niveles[(scores >= thresholds['moderate']) & (scores < thresholds['critical'])] = 'moderado'
        niveles[(scores >= thresholds['low']) & (scores < thresholds['moderate'])] = 'bajo'
        
        return niveles
    
    
    def predict(self, df_features):
        """
        Genera predicciones para el DataFrame de features
        
        Args:
            df_features: DataFrame con features generadas por FeatureEngineer
            
        Returns:
            DataFrame con predicciones y niveles de alerta
        """
        log.info("\n" + "=" * 80)
        log.info(" GENERANDO PREDICCIONES")
        log.info("=" * 80)
        
        if len(df_features) == 0:
            log.warning(" DataFrame vacío, no hay ventanas para predecir")
            return pd.DataFrame()
        
        # Guardar columnas identificadoras
        identificadores = df_features[['id_maquina_dfos', 'id_linea', 'timestamp_hora']].copy()
        
        # Validar y preparar features
        X = self._validate_features(df_features)
        
        # Verificar que no haya NaN o Inf
        if X.isna().any().any():
            log.warning(" Se encontraron NaN en features, rellenando con 0")
            X = X.fillna(0)
        
        if np.isinf(X.values).any():
            log.warning(" Se encontraron Inf en features, reemplazando con 0")
            X = X.replace([np.inf, -np.inf], 0)
        
        # Hacer predicción
        log.info(" Ejecutando modelo LightGBM...")
        scores = self.model.predict(X)
        
        # Validar scores
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        
        # Asegurar que scores estén en rango [0, 1]
        scores = np.clip(scores, 0, 1)
        
        log.info(f" Predicciones generadas: {len(scores):,}")
        log.info(f"   Score mínimo: {scores.min():.4f}")
        log.info(f"   Score máximo: {scores.max():.4f}")
        log.info(f"   Score promedio: {scores.mean():.4f}")
        log.info(f"   Score mediana: {np.median(scores):.4f}")
        
        # Clasificar en niveles de alerta
        log.info("\n Clasificando en niveles de alerta...")
        niveles_alerta = self._classify_alerts(scores)
        
        # Estadísticas de alertas
        unique, counts = np.unique(niveles_alerta, return_counts=True)
        log.info("\n Distribución de alertas:")
        for nivel, count in zip(unique, counts):
            porcentaje = count / len(scores) * 100
            emoji = {
                'critico': 'CRTICO',
                'moderado': 'MODERADO',
                'bajo': 'BAJO',
                'normal': 'BLANCO'
            }.get(nivel, '?')
            log.info(f"   {emoji} {nivel.upper()}: {count:,} ({porcentaje:.2f}%)")
        
        # Crear DataFrame de resultados
        resultados = identificadores.copy()
        resultados['score'] = scores
        resultados['nivel_alerta'] = niveles_alerta
        
        # Calcular timestamp_ventana (la hora que estamos prediciendo = siguiente hora)
        resultados['timestamp_ventana'] = resultados['timestamp_hora'] + pd.Timedelta(hours=1)
        
        # Reordenar columnas
        resultados = resultados[[
            'timestamp_ventana',
            'timestamp_hora',
            'id_maquina_dfos',
            'id_linea',
            'score',
            'nivel_alerta'
        ]]
        
        # Resumen final
        log.info("\n Predicciones completadas:")
        log.info(f"   Total ventanas: {len(resultados):,}")
        log.info(f"   Máquinas únicas: {resultados['id_maquina_dfos'].nunique()}")
        log.info(f"   Líneas únicas: {resultados['id_linea'].nunique()}")
        
        # Alertas críticas (si las hay)
        alertas_criticas = resultados[resultados['nivel_alerta'] == 'critico']
        if len(alertas_criticas) > 0:
            log.warning(f"\n ¡ATENCIÓN! {len(alertas_criticas)} ALERTAS CRÍTICAS detectadas:")
            for _, row in alertas_criticas.head(10).iterrows():
                log.warning(f"   Máquina: {row['id_maquina_dfos']} | Score: {row['score']:.4f} | Ventana: {row['timestamp_ventana']}")
            if len(alertas_criticas) > 10:
                log.warning(f"   ... y {len(alertas_criticas) - 10} más")
        
        return resultados
    
    
    def get_model_info(self):
        """
        Retorna información del modelo cargado
        
        Returns:
            dict con información del modelo
        """
        return {
            'version': self.config['model']['version'],
            'num_features': len(self.features_necesarias),
            'features': self.features_necesarias,
            'thresholds': self.inference_config['thresholds'],
            'num_trees': self.model.num_trees() if self.model else None
        }