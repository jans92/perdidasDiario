#  Sistema de Mantenimiento Predictivo - MVP

Sistema de predicción de fallas de equipos industriales con 24 horas de anticipación utilizando Machine Learning (LightGBM).

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Métricas de Rendimiento](#métricas-de-rendimiento)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Pipeline de Datos](#pipeline-de-datos)
- [Tabla de Predicciones](#tabla-de-predicciones)
- [Niveles de Alerta](#niveles-de-alerta)
- [Archivos del Modelo](#archivos-del-modelo)
- [Monitoreo y Logs](#monitoreo-y-logs)
- [Troubleshooting](#troubleshooting)
- [Mantenimiento](#mantenimiento)

---

## 📖 Descripción

Este sistema analiza datos de producción y mantenimiento de equipos industriales para predecir fallas con **24 horas de anticipación**, permitiendo:

- **Reducir paradas no planificadas** mediante mantenimiento preventivo
- **Priorizar recursos** de mantenimiento según nivel de riesgo
- **Optimizar costos** evitando fallas catastróficas
- **Medir rendimiento** comparando predicciones vs. fallas reales

### Características principales

| Característica | Descripción |
|----------------|-------------|
| **Modelo** | LightGBM optimizado con 34 árboles |
| **Features** | 63 variables predictivas |
| **Horizonte** | Predicción a 24 horas |
| **Granularidad** | Por máquina y ventana horaria |
| **Actualización** | Ejecución horaria automatizada |
| **Validación** | Actualización diaria de targets reales |

---

##  Métricas de Rendimiento

Validación realizada sobre 5 fechas diferentes con alta incidencia de fallas:

### Rendimiento por Nivel de Alerta

| Nivel | Precision | Recall | Descripción |
|-------|-----------|--------|-------------|
| 🔴 **Crítico** | 67.6% | 42.9% | Alta confianza, requiere acción inmediata |
| 🔴+🟡 **Crítico + Moderado** | 57.2% | **68.6%** | Cobertura amplia de fallas |

### Interpretación Operativa

| Indicador | Valor | Significado |
|-----------|-------|-------------|
| **Recall 68.6%** | ~120/180 | **Previene 7 de cada 10 fallas** |
| **Precision 57.2%** | ~120/215 | 1 de cada 2 alertas es falla real |
| **Precision Crítico** | 67.6% | 2 de cada 3 alertas críticas son fallas |

### Consistencia Temporal

| Fecha Validación | Recall | Precision |
|------------------|--------|-----------|
| 2025-07-01 | 63.3% | 59.2% |
| 2025-07-22 | 64.4% | 57.0% |
| 2025-07-08 | 69.6% | 55.9% |
| 2025-07-03 | 74.6% | 56.4% |
| 2025-06-11 | 71.0% | 57.4% |

---

## 🏗️ Arquitectura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Base de Datos │────▶│   Extracción    │────▶│    Feature      │
│   (MySQL/Azure) │     │ (7d eventos,    │     │   Engineering   │
│                 │     │  60d mant.)     │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Alertas      │◀────│   Clasificación │◀────│    Modelo       │
│  (Email/Log)    │     │   (Umbrales)    │     │   LightGBM      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  Update Targets │◀────│    Guardar      │
                        │   (diario)      │     │   Predicciones  │
                        └─────────────────┘     └─────────────────┘
                                │                        │
                                └────────────┬───────────┘
                                             ▼
                                ┌─────────────────────────┐
                                │  bui_predicciones_hora  │
                                │  (tabla de resultados)  │
                                └─────────────────────────┘
```

### Componentes

| Archivo | Descripción |
|---------|-------------|
| `main.py` | Orquestador principal del pipeline |
| `save_predictions.py` | Guarda predicciones en BD |
| `update_targets.py` | Actualiza target_real después de 24h |
| `src/db_connector.py` | Conexión y extracción de datos desde MySQL/Azure |
| `src/feature_engineering.py` | Transformación de datos raw a 63 features |
| `src/predictor.py` | Carga del modelo y generación de predicciones |
| `src/utils.py` | Funciones auxiliares (logging, alertas) |
| `src/config_loader.py` | Carga configuración y credenciales |

---

##  Requisitos

### Sistema

- Python 3.10+
- Acceso a base de datos MySQL/Azure
- 4GB RAM mínimo (8GB recomendado)

### Dependencias principales

```
numpy==2.0.2
pandas==2.2.2
scikit-learn>=1.6.1
lightgbm>=4.6.0
sqlalchemy>=2.0.0
pymysql>=1.1.0
pyyaml>=6.0
tqdm>=4.66.0

```

---

##  Instalación

### 1. Clonar repositorio

```bash
git clone <repository-url>
cd Run_predictor_daily
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Crear tabla de predicciones

Ejecutar el DDL en la base de datos:

```bash
mysql -h your-host -u your-user -p your_database < scripts/ddl_bui_predicciones_hora.sql
```

### 5. Verificar archivos del modelo

Asegurarse de que existen los siguientes archivos en `model/`:

```
model/
├── model_optimized.pkl           # Modelo LightGBM
├── features_utiles.json          # Lista de 63 features
├── inference_config.json         # Umbrales de alertas
└── preprocessing_artifacts.pkl   # Artefactos de preprocesamiento
```

---

## ⚙️ Configuración

### config.yaml

```yaml
# Conexión a base de datos
database:
  host: "your-server.mysql.database.azure.com"
  port: 3306
  bbdd: "your_database"
  user: "your_user"
  password: "your_password"  # O usar variable de entorno

# Rutas del modelo
model:
  path: "model/model_optimized.pkl"
  version: "v1.1"

# Configuración de ejecución
execution:
  lookback_hours: 168              # 7 días para bui_perdida
  lookback_days_maintenance: 60    # 60 días para bui_pm_ewo
  timeout_minutes: 30
  # fecha_referencia: '2025-07-25' # Descomentar para modo test

# Alertas por email (opcional)
alerts:
  email_enabled: false
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender: "alertas@empresa.com"
  recipients:
    - "mantenimiento@empresa.com"

# Logging
logging:
  level: "INFO"
  console: true
  file: true
  log_dir: "logs/"
```

### Variables de entorno (recomendado para producción)

```bash
export DB_HOST="your-server.mysql.database.azure.com"
export DB_USER="your_user"
export DB_PASSWORD="your_password"
export DB_NAME="your_database"
```

---

## 🎯 Uso

### Ejecución manual

```bash
# Ejecutar predicciones (extrae, procesa, predice y guarda en BD)
python main.py

# Actualizar targets reales (después de 24h)
python update_targets.py
```

### Ejecución con fecha específica (modo test)

Editar `config.yaml`:

```yaml
execution:
  fecha_referencia: '2025-07-25'  # Usar esta fecha en lugar de NOW()
```

### Test de inferencia

```bash
python test_inference.py
```

### Configurar ejecución automática (cron)

```bash
# Editar crontab
crontab -e

# Agregar líneas:

# Predicciones cada hora
0 * * * * cd /path/to/Run_predictor_daily && /path/to/.venv/bin/python main.py >> logs/cron.log 2>&1

# Actualizar targets 1 vez al día (23:00)
0 23 * * * cd /path/to/Run_predictor_daily && /path/to/.venv/bin/python update_targets.py >> logs/cron.log 2>&1
```

---

## 📁 Estructura del Proyecto

```
/Run_predictor_daily/
│
├── .env                         # Credenciales secretas
├── main.py                      # Orquestador principal
├── save_predictions.py          # Guarda predicciones en BD
├── update_targets.py            # Actualiza target_real y acierto
├── config.yaml                  # Configuraciones
├── requirements.txt             # Dependencias Python
├── README.md                    # Esta documentación
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py         # Carga credenciales del .env
│   ├── db_connector.py          # Conexión y extracción de BD
│   ├── feature_engineering.py   # Generación de 63 features
│   ├── predictor.py             # Carga modelo y predicción
│   └── utils.py                 # Funciones auxiliares
│
├── model/
│   ├── model_optimized.pkl      # Modelo LightGBM (pickle)
│   ├── model_optimized.txt      # Modelo LightGBM (texto)
│   ├── features_utiles.json     # Lista de 63 features
│   ├── inference_config.json    # Umbrales y parámetros
│   └── preprocessing_artifacts.pkl  # Artefactos de preprocesamiento
│
├── logs/
│   ├── predictions_YYYY-MM-DD.log
│   └── errors.log
│
├── scripts/
│   ├── ddl_bui_predicciones_hora.sql  # DDL para crear tabla
│   ├── setup_cron.sh                   # Configurar cron job
│   └── monitor_logs.sh                 # Monitorear logs
│
└── tests/
    ├── test_features.py         # Tests unitarios de features
    └── test_predictor.py        # Tests del predictor
```

---

##  Pipeline de Datos

### 1. Extracción (db_connector.py)

Extrae datos de 3 tablas principales:

| Tabla | Descripción | Ventana |
|-------|-------------|---------|
| `bui_perdida` | Eventos de pérdida/producción | 7 días (168 horas) |
| `bui_pm_ewo` | Órdenes de mantenimiento | 60 días |
| `bui_line` | Metadata de líneas | Todas |

### 2. Feature Engineering (feature_engineering.py)

Genera 63 features en 17 pasos:

| Paso | Descripción | Features |
|------|-------------|----------|
| 1-4 | Pivoteo y agregación base | ~74 |
| 5 | Features detalladas por evento | +41 |
| 6 | Métricas agregadas | +7 |
| 7 | Features de mantenimiento | +5 |
| 8 | Rolling windows (2h, 6h, 12h, 24h) | +48 |
| 9 | Tendencias y aceleraciones | +8 |
| 10 | Ratios y proporciones | +8 |
| 11 | Features temporales | +13 |
| 12 | Variabilidad (std, cv) | +8 |
| 13 | Encoding categorías | +22 |
| 14 | Z-scores y anomalías | +11 |
| 15 | Comparación con pares | +7 |

### 3. Predicción (predictor.py)

- Carga modelo LightGBM
- Valida presencia de 63 features
- Genera scores de probabilidad
- Clasifica en niveles de alerta

### 4. Guardado (save_predictions.py)

- Inserta predicciones en `bui_predicciones_hora`
- Calcula `fl_pred_modelo` (1 si score >= 0.31)
- Maneja duplicados con INSERT IGNORE

### 5. Actualización de Targets (update_targets.py)

- Se ejecuta 1 vez al día
- Busca predicciones donde ya pasaron 24h
- Verifica en `bui_perdida` si hubo falla grave
- Actualiza `fl_target_real` y `fl_acierto`

---

## 🗃️ Tabla de Predicciones

### Estructura: `bui_predicciones_hora`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id_prediccion` | INT AUTO_INCREMENT | PK |
| `id_maquina_dfos` | VARCHAR(50) | ID de máquina (ej: "23_L01-M001") |
| `id_linea` | INT | ID de línea de producción |
| `fe_ventana` | DATETIME | Ventana horaria predicha |
| `nm_score` | DECIMAL(5,4) | Score del modelo (0.0000-1.0000) |
| `de_nivel_riesgo` | VARCHAR(20) | critico/moderado/bajo/normal |
| `fl_pred_modelo` | TINYINT | 1 si score >= 0.31, 0 si no |
| `fl_target_real` | TINYINT NULL | NULL=pendiente, 0=no falla, 1=falla |
| `fl_acierto` | TINYINT NULL | NULL=pendiente, 0=error, 1=acierto |
| `de_modelo_version` | VARCHAR(20) | Versión del modelo (ej: "v1.1") |
| `fe_creado` | TIMESTAMP | Fecha de inserción (auto) |
| `fe_actualizado` | TIMESTAMP NULL | Fecha de actualización |

### Criterio de Falla Grave (fl_target_real = 1)

Una falla se considera **grave** si cumple:

**Base:** `de_perdida_2 = 'Breakdown & Equipment Failure Time'`

**Y al menos uno de:**

| # | Criterio | Condición |
|---|----------|-----------|
| 1 | Con EWO correctivo válido | duración >= 10 min |
| 2 | Duración prolongada | duración >= 60 min |
| 3 | Tipo crítico (Mechanical/Electrical/I&C) | duración >= 20 min |

### Consultas útiles

```sql
-- Predicciones pendientes de actualizar
SELECT * FROM bui_predicciones_hora 
WHERE fl_target_real IS NULL 
  AND fe_ventana < NOW() - INTERVAL 24 HOUR;

-- Matriz de confusión
SELECT 
    fl_pred_modelo,
    fl_target_real,
    COUNT(*) as cantidad
FROM bui_predicciones_hora
WHERE fl_target_real IS NOT NULL
GROUP BY fl_pred_modelo, fl_target_real;

-- Métricas de precisión
SELECT 
    SUM(CASE WHEN fl_pred_modelo = 1 AND fl_target_real = 1 THEN 1 ELSE 0 END) as TP,
    SUM(CASE WHEN fl_pred_modelo = 1 AND fl_target_real = 0 THEN 1 ELSE 0 END) as FP,
    SUM(CASE WHEN fl_pred_modelo = 0 AND fl_target_real = 1 THEN 1 ELSE 0 END) as FN,
    SUM(CASE WHEN fl_pred_modelo = 0 AND fl_target_real = 0 THEN 1 ELSE 0 END) as TN
FROM bui_predicciones_hora
WHERE fl_target_real IS NOT NULL;
```

---

## 🚦 Niveles de Alerta

| Nivel | Umbral | Acción recomendada |
|-------|--------|-------------------|
| 🔴 **CRÍTICO** | Score >= 0.40 | Inspección inmediata, programar mantenimiento |
| 🟡 **MODERADO** | 0.31 - 0.40 | Monitoreo aumentado, planificar revisión |
| 🟢 **BAJO** | 0.10 - 0.31 | Operación normal con vigilancia |
| ⚪ **NORMAL** | < 0.10 | Sin acción requerida |

### Distribución esperada de alertas

```
NORMAL:   ~20-25%
BAJO:     ~55-60%
MODERADO: ~15-20%
CRÍTICO:  ~5-8%
```

---

## 📦 Archivos del Modelo

### model_optimized.pkl

Modelo LightGBM serializado con pickle.

```python
import pickle
with open('model/model_optimized.pkl', 'rb') as f:
    model = pickle.load(f)
```

### features_utiles.json

Lista de las 63 features requeridas por el modelo.

```json
{
  "features": [
    "breakdown_equipment_failure_std_7d",
    "breakdown_equipment_failure_zscore",
    "avg_breakdown_equipment_failure_subcategoria",
    ...
  ],
  "n_features": 63
}
```

### inference_config.json

Configuración de umbrales y parámetros.

```json
{
  "umbral_optimo": 0.31,
  "thresholds": {
    "critical": 0.40,
    "moderate": 0.31,
    "low": 0.10
  },
  "scale_pos_weight": 33.35,
  "n_features": 63,
  "de_modelo_version": "v1.1"
}
```

### preprocessing_artifacts.pkl

Artefactos para garantizar consistencia entre entrenamiento e inferencia:

| Artefacto | Propósito |
|-----------|-----------|
| LabelEncoders | Codificación consistente de categorías |
| Subcategorías one-hot | Columnas fijas esperadas |
| Threshold alto riesgo | Umbral fijo (0.096857) |
| Estadísticas Z-score | Media/Std por máquina del entrenamiento |
| Medias por subcategoría | Para comparación con pares |

---

## 📊 Monitoreo y Logs

### Estructura de logs

```
logs/
├── predictions_2025-12-05.log   # Log diario de predicciones
├── errors.log                    # Errores del sistema
└── cron.log                      # Salida del cron job
```

### Formato de log

```
2025-12-05 06:00:15 - INFO -  INICIANDO PREDICCIÓN DIARIA
2025-12-05 06:00:16 - INFO -  Conexión a BD establecida
2025-12-05 06:05:23 - INFO -  Feature engineering completado: (9366, 268)
2025-12-05 06:05:24 - INFO - 🚦 Distribución de alertas:
2025-12-05 06:05:24 - INFO -    🔴 CRITICO: 498 (5.32%)
2025-12-05 06:05:24 - INFO -    🟡 MODERADO: 1,588 (16.95%)
2025-12-05 06:05:24 - WARNING - 🚨 498 ALERTAS CRÍTICAS detectadas
2025-12-05 06:05:25 - INFO - 💾 Predicciones guardadas: 9,366 registros
```

### Monitorear en tiempo real

```bash
tail -f logs/predictions_$(date +%Y-%m-%d).log
```

---

##  Troubleshooting

### Error: "Features faltantes"

**Causa**: El feature engineering no generó todas las 63 features.

**Solución**:
1. Verificar que `preprocessing_artifacts.pkl` existe
2. Verificar datos de entrada (mínimo 7 días)
3. Revisar logs para identificar paso fallido

### Error: "Conexión a BD fallida"

**Causa**: Credenciales incorrectas o servidor no accesible.

**Solución**:
1. Verificar `config.yaml` o variables de entorno
2. Probar conexión manualmente: `mysql -h host -u user -p`
3. Verificar firewall/VPN si aplica

### Error: "Duplicate entry"

**Causa**: Predicción duplicada para misma máquina y ventana.

**Solución**: Normal si se re-ejecuta para la misma hora. El sistema usa INSERT IGNORE.

### Error: "Score máximo muy bajo"

**Causa**: Posible training-serving skew.

**Solución**:
1. Verificar que se usa `preprocessing_artifacts.pkl`
2. Comparar distribución de features vs entrenamiento
3. Regenerar artifacts si es necesario

### Warning: "sklearn version mismatch"

**Causa**: Versión de sklearn diferente entre entrenamiento e inferencia.

**Solución**:
1. Instalar misma versión: `pip install scikit-learn==1.6.1`
2. O regenerar artifacts con versión actual

---

## 🔄 Mantenimiento

### Actualización del modelo

1. Entrenar nuevo modelo en notebook
2. Generar nuevos artifacts:
   ```python
   %run generar_preprocessing_artifacts.py
   ```
3. Copiar archivos a `model/`:
   - `model_optimized.pkl`
   - `features_utiles.json`
   - `inference_config.json`
   - `preprocessing_artifacts.pkl`
4. Ejecutar test de validación
5. Desplegar

### Re-entrenamiento recomendado

- **Frecuencia**: Cada 3-6 meses
- **Trigger**: Si recall cae por debajo del 60%
- **Datos**: Mínimo 6 meses de historia

### Monitoreo de drift

Verificar periódicamente:
- Distribución de scores (debe mantenerse estable)
- Ratio de alertas críticas (~5-8%)
- Precision/Recall en fallas conocidas (consultar `bui_predicciones_hora`)

### Consulta de rendimiento histórico

```sql
SELECT 
    DATE(fe_creado) as fecha,
    COUNT(*) as total_predicciones,
    SUM(fl_target_real) as fallas_reales,
    SUM(fl_acierto) as aciertos,
    ROUND(AVG(fl_acierto) * 100, 2) as accuracy_pct
FROM bui_predicciones_hora
WHERE fl_target_real IS NOT NULL
GROUP BY DATE(fe_creado)
ORDER BY fecha DESC
LIMIT 30;
```

---

## 📝 Changelog

### v1.1 (2025-12-06)
- ✨ Nuevo: `save_predictions.py` para guardar predicciones en BD
- ✨ Nuevo: `update_targets.py` para actualizar target_real
- ✨ Nueva tabla `bui_predicciones_hora` para tracking de predicciones
-  Fix: Soporte para fecha_referencia configurable (modo test)
- 🔧 Fix: Extracción de 60 días para bui_pm_ewo (antes 7 días)
- 📝 Documentación actualizada

### v1.0 (2025-12-04)
- 🚀 Release inicial
- Modelo LightGBM con 63 features
- Pipeline completo de extracción a predicción