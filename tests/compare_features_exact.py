"""
compare_features_exact.py - Comparación exacta con entrenamiento
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
import json

BASE_DIR = Path(__file__).parent.parent  #  Subir un nivel (salir de /tests/)
sys.path.insert(0, str(BASE_DIR))

from src.db_connector import DatabaseConnector
from src.feature_engineering_backup import FeatureEngineer

print("=" * 80)
print(" COMPARACIÓN EXACTA: INFERENCIA vs ENTRENAMIENTO")
print("=" * 80)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Cargar config
with open(BASE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Fecha específica del TEST (debe estar en df_con_target)
FECHA_COMPARACION = '2025-07-24 12:00:00'  # Usa una hora con muchas ventanas

print(f"\n Fecha de comparación: {FECHA_COMPARACION}")

# ============================================================================
# PASO 1: CARGAR DATASET DE ENTRENAMIENTO
# ============================================================================

print("\n" + "=" * 80)
print(" PASO 1: CARGANDO DATASET DE ENTRENAMIENTO")
print("=" * 80)

TRAINING_PATH = Path('/Users/juanmanuel/Desktop/df_con_target.parquet')

if not TRAINING_PATH.exists():
    print(f" Archivo no encontrado: {TRAINING_PATH}")
    print("Por favor ajusta la ruta en el script")
    sys.exit(1)

print(f"Cargando: {TRAINING_PATH}")
df_training = pd.read_parquet(TRAINING_PATH)

print(f" Dataset cargado: {df_training.shape}")
print(f"   Columnas: {len(df_training.columns)}")
print(f"   Rango temporal: {df_training['timestamp_hora'].min()} a {df_training['timestamp_hora'].max()}")

# Filtrar a la fecha específica
fecha_comp = pd.to_datetime(FECHA_COMPARACION)
df_training_fecha = df_training[df_training['timestamp_hora'] == fecha_comp].copy()

print(f"\n Ventanas en fecha {FECHA_COMPARACION}: {len(df_training_fecha)}")

if len(df_training_fecha) == 0:
    print(" No hay datos para esa fecha en el dataset de entrenamiento")
    print("\nFechas disponibles:")
    print(df_training['timestamp_hora'].value_counts().head(20))
    sys.exit(1)

# ============================================================================
# PASO 2: EXTRAER DATOS DE BD Y GENERAR FEATURES (INFERENCIA)
# ============================================================================

print("\n" + "=" * 80)
print(" PASO 2: GENERANDO FEATURES DE INFERENCIA")
print("=" * 80)

# Extraer desde 45 días antes hasta la fecha específica
fecha_fin = fecha_comp
fecha_inicio = fecha_comp - pd.Timedelta(days=45)

print(f"Extrayendo desde {fecha_inicio} hasta {fecha_fin}")

db = DatabaseConnector(config)

# Extraer bui_perdida
query_perdida = f"""
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

bui_perdida = db.execute_query(query_perdida)
print(f" bui_perdida: {len(bui_perdida):,} registros")

# Extraer bui_pm_ewo
query_ewo = f"""
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

bui_pm_ewo = db.execute_query(query_ewo)
print(f" bui_pm_ewo: {len(bui_pm_ewo):,} registros")

# Calcular estado_ewo
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

# Extraer bui_line
query_line = "SELECT id_linea, id_subcategoria, id_fabrica_area FROM bui_line"
bui_line = db.execute_query(query_line)
print(f" bui_line: {len(bui_line):,} registros")

# Convertir fechas
bui_perdida['fe_inicio'] = pd.to_datetime(bui_perdida['fe_inicio'])
bui_perdida['fe_fin'] = pd.to_datetime(bui_perdida['fe_fin'])
bui_pm_ewo['fe_inicio_averia'] = pd.to_datetime(bui_pm_ewo['fe_inicio_averia'])
bui_pm_ewo['fe_inicio_prod'] = pd.to_datetime(bui_pm_ewo['fe_inicio_prod'])

db.close()

# Feature Engineering
print("\nGenerando features...")
fe = FeatureEngineer(config)
df_inference = fe.transform(bui_perdida, bui_pm_ewo, bui_line)

print(f" Features generadas: {df_inference.shape}")

# Filtrar solo la fecha específica
df_inference_fecha = df_inference[df_inference['timestamp_hora'] == fecha_comp].copy()
print(f" Ventanas en inferencia para fecha {FECHA_COMPARACION}: {len(df_inference_fecha)}")

# ============================================================================
# PASO 3: COMPARACIÓN DETALLADA
# ============================================================================

print("\n" + "=" * 80)
print(" PASO 3: COMPARACIÓN DETALLADA")
print("=" * 80)

# Cargar features necesarias del modelo
with open(BASE_DIR / 'model/features_utiles.json', 'r') as f:
    features_info = json.load(f)

features_necesarias = features_info['features']

print(f"\nFeatures a comparar: {len(features_necesarias)}")

# Encontrar máquinas comunes
maquinas_training = set(df_training_fecha['id_maquina_dfos'].unique())
maquinas_inference = set(df_inference_fecha['id_maquina_dfos'].unique())
maquinas_comunes = maquinas_training & maquinas_inference

print(f"\nMáquinas en training: {len(maquinas_training)}")
print(f"Máquinas en inference: {len(maquinas_inference)}")
print(f"Máquinas comunes: {len(maquinas_comunes)}")

if len(maquinas_comunes) == 0:
    print("\n No hay máquinas comunes para comparar")
    print("\nMáquinas en training:", sorted(list(maquinas_training))[:10])
    print("Máquinas en inference:", sorted(list(maquinas_inference))[:10])
    sys.exit(1)

# Seleccionar 3 máquinas para comparar en detalle
maquinas_test = list(maquinas_comunes)[:3]

print(f"\n Comparando {len(maquinas_test)} máquinas en detalle:")
for maq in maquinas_test:
    print(f"   - {maq}")

# ============================================================================
# COMPARACIÓN POR MÁQUINA
# ============================================================================

resultados_comparacion = []

for maquina in maquinas_test:
    print("\n" + "=" * 80)
    print(f" MÁQUINA: {maquina}")
    print("=" * 80)
    
    # Obtener fila de training e inference
    row_train = df_training_fecha[df_training_fecha['id_maquina_dfos'] == maquina].iloc[0]
    row_infer = df_inference_fecha[df_inference_fecha['id_maquina_dfos'] == maquina].iloc[0]
    
    print(f"\n TOP 30 FEATURES CON MAYOR DIFERENCIA:")
    print("-" * 100)
    print(f"{'Feature':<50} {'Training':>15} {'Inference':>15} {'Diff':>12} {'Diff %':>10}")
    print("-" * 100)
    
    diferencias_maquina = []
    
    for feat in features_necesarias:
        if feat in row_train.index and feat in row_infer.index:
            val_train = row_train[feat]
            val_infer = row_infer[feat]
            
            diff_abs = val_infer - val_train
            
            if abs(val_train) > 1e-6:
                diff_pct = abs(diff_abs) / abs(val_train) * 100
            else:
                diff_pct = 0 if abs(val_infer) < 1e-6 else 999
            
            diferencias_maquina.append({
                'maquina': maquina,
                'feature': feat,
                'training': val_train,
                'inference': val_infer,
                'diff_abs': diff_abs,
                'diff_pct': diff_pct
            })
    
    # Ordenar por diferencia porcentual
    df_diff_maq = pd.DataFrame(diferencias_maquina).sort_values('diff_pct', ascending=False)
    
    # Mostrar top 30
    for idx, row in df_diff_maq.head(30).iterrows():
        flag = "" if row['diff_pct'] > 50 else "" if row['diff_pct'] > 10 else ""
        print(f"{flag} {row['feature']:<48} {row['training']:>15.4f} {row['inference']:>15.4f} {row['diff_abs']:>12.4f} {row['diff_pct']:>9.1f}%")
    
    resultados_comparacion.append(df_diff_maq)
    
    # Resumen
    print(f"\n RESUMEN PARA {maquina}:")
    print(f"   Features con diff >100%: {(df_diff_maq['diff_pct'] > 100).sum()}")
    print(f"   Features con diff >50%:  {(df_diff_maq['diff_pct'] > 50).sum()}")
    print(f"   Features con diff >10%:  {(df_diff_maq['diff_pct'] > 10).sum()}")

# ============================================================================
# ANÁLISIS AGREGADO
# ============================================================================

print("\n" + "=" * 80)
print(" ANÁLISIS AGREGADO - TODAS LAS MÁQUINAS COMUNES")
print("=" * 80)

# Comparar TODAS las máquinas comunes
diferencias_totales = []

for maquina in maquinas_comunes:
    row_train = df_training_fecha[df_training_fecha['id_maquina_dfos'] == maquina].iloc[0]
    row_infer = df_inference_fecha[df_inference_fecha['id_maquina_dfos'] == maquina].iloc[0]
    
    for feat in features_necesarias:
        if feat in row_train.index and feat in row_infer.index:
            val_train = row_train[feat]
            val_infer = row_infer[feat]
            diff_abs = val_infer - val_train
            
            if abs(val_train) > 1e-6:
                diff_pct = abs(diff_abs) / abs(val_train) * 100
            else:
                diff_pct = 0 if abs(val_infer) < 1e-6 else 999
            
            diferencias_totales.append({
                'feature': feat,
                'diff_pct': diff_pct
            })

df_diff_total = pd.DataFrame(diferencias_totales)

# Agrupar por feature y calcular estadísticas
diff_por_feature = df_diff_total.groupby('feature')['diff_pct'].agg(['mean', 'median', 'max', 'std']).sort_values('mean', ascending=False)

print(f"\n TOP 20 FEATURES MÁS PROBLEMÁTICAS (promedio todas las máquinas):")
print("-" * 100)
print(f"{'Feature':<50} {'Mean %':>10} {'Median %':>10} {'Max %':>10} {'Std':>10}")
print("-" * 100)

for feat, row in diff_por_feature.head(20).iterrows():
    flag = "" if row['mean'] > 50 else "" if row['mean'] > 10 else ""
    print(f"{flag} {feat:<48} {row['mean']:>10.1f} {row['median']:>10.1f} {row['max']:>10.1f} {row['std']:>10.1f}")

# Guardar resultados
output_file = BASE_DIR / f'feature_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
diff_por_feature.to_csv(output_file)
print(f"\n Comparación guardada en: {output_file}")

# ============================================================================
# CONCLUSIONES
# ============================================================================

print("\n" + "=" * 80)
print(" DIAGNÓSTICO")
print("=" * 80)

features_problematicas = diff_por_feature[diff_por_feature['mean'] > 50]

if len(features_problematicas) > 0:
    print(f"\n ENCONTRADAS {len(features_problematicas)} FEATURES CON PROBLEMAS SERIOS:")
    for feat in features_problematicas.head(10).index:
        print(f"   - {feat}")
    print("\n ACCIÓN: Revisar el cálculo de estas features en feature_engineering.py")
else:
    print("\n No se encontraron features con diferencias críticas (>50%)")
    print("\n Las predicciones bajas pueden ser normales si esta fecha")
    print("   no tuvo fallas graves en el periodo de entrenamiento")

print("\n" + "=" * 80)