"""
verify_extraction.py - Verificar si los eventos se están extrayendo
"""

import sys
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timedelta

# Ajustar ruta al directorio raíz del proyecto
BASE_DIR = Path(__file__).parent.parent  #  CORREGIDO: subir un nivel
sys.path.insert(0, str(BASE_DIR))

with open(BASE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

from src.db_connector import DatabaseConnector

db = DatabaseConnector(config)

fecha_test = '2025-07-24 12:00:00'
fecha_test_dt = pd.to_datetime(fecha_test)

print("=" * 80)
print(f" VERIFICANDO EXTRACCIÓN PARA: {fecha_test}")
print("=" * 80)

# TEST 1: Query original (hasta fecha_fin)
print("\n TEST 1: Query original (hasta fecha_fin exacta)")
query1 = f"""
SELECT COUNT(*) as count
FROM bui_perdida
WHERE fe_inicio >= '{(fecha_test_dt - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')}'
  AND fe_inicio <= '{fecha_test_dt.strftime('%Y-%m-%d %H:%M:%S')}'
"""
result1 = db.execute_query(query1)
print(f"   Eventos extraídos: {result1['count'].iloc[0]:,}")

# TEST 2: Query extendida (hasta fecha_fin + 1h)
print("\n TEST 2: Query extendida (hasta fecha_fin + 1 hora)")
query2 = f"""
SELECT COUNT(*) as count
FROM bui_perdida
WHERE fe_inicio >= '{(fecha_test_dt - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')}'
  AND fe_inicio <= '{(fecha_test_dt + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')}'
"""
result2 = db.execute_query(query2)
print(f"   Eventos extraídos: {result2['count'].iloc[0]:,}")

# Diferencia
diff = result2['count'].iloc[0] - result1['count'].iloc[0]
print(f"\n    Diferencia: {diff:,} eventos")

# TEST 3: Eventos específicos en la ventana 12:00-13:00
print("\n TEST 3: Eventos específicamente en la hora 12:00-13:00")
query3 = f"""
SELECT 
    id_maquina_dfos,
    COUNT(*) as num_eventos,
    SUM(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as duracion_total
FROM bui_perdida
WHERE fe_inicio >= '{fecha_test_dt.strftime('%Y-%m-%d %H:%M:%S')}'
  AND fe_inicio < '{(fecha_test_dt + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')}'
GROUP BY id_maquina_dfos
ORDER BY num_eventos DESC
LIMIT 10
"""
result3 = db.execute_query(query3)
print(f"\nTop 10 máquinas con eventos en esa hora:")
print(result3.to_string(index=False))

# TEST 4: Verificar máquinas específicas que fallaron en comparación
print("\n TEST 4: Verificar máquinas específicas (A1_AKASH2, 801500109651)")

maquinas_test = ['A1_AKASH2', '801500109651']

for maquina in maquinas_test:
    query4 = f"""
    SELECT 
        COUNT(*) as num_eventos,
        SUM(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as total_duration,
        SUM(CASE WHEN de_perdida_1 = 'MPL' THEN TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin) ELSE 0 END) as mpl_time,
        MAX(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as max_event_duration,
        AVG(TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin)) as avg_event_duration
    FROM bui_perdida
    WHERE id_maquina_dfos = '{maquina}'
      AND fe_inicio >= '{fecha_test_dt.strftime('%Y-%m-%d %H:%M:%S')}'
      AND fe_inicio < '{(fecha_test_dt + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')}'
    """
    result4 = db.execute_query(query4)
    
    print(f"\n   Máquina: {maquina}")
    if result4['num_eventos'].iloc[0] > 0:
        print(f"      Eventos:           {result4['num_eventos'].iloc[0]}")
        print(f"      total_duration:    {result4['total_duration'].iloc[0]:.2f} min")
        print(f"      mpl_time:          {result4['mpl_time'].iloc[0]:.2f} min")
        print(f"      max_event_duration:{result4['max_event_duration'].iloc[0]:.2f} min")
        print(f"      avg_event_duration:{result4['avg_event_duration'].iloc[0]:.2f} min")
    else:
        print(f"       NO hay eventos en esa hora")

db.close()

print("\n" + "=" * 80)
print(" CONCLUSIONES:")
print("=" * 80)

if diff > 0:
    print(f" CONFIRMADO: La query original NO incluye eventos de la hora completa")
    print(f"   Eventos perdidos: {diff:,}")
    print(f"   Porcentaje perdido: {diff/result2['count'].iloc[0]*100:.1f}%")
    print("\n SOLUCIÓN: Agregar +1 hora a fecha_fin en todas las queries")
    print("   Archivos a modificar:")
    print("   - test_inference.py (línea ~106)")
    print("   - main.py (línea ~51)")
    print("   - compare_features_exact.py (línea ~82)")
else:
    print(" La query captura todos los eventos correctamente")
    print(" El problema debe estar en otra parte del pipeline")
    print("   Posibles causas:")
    print("   - Filtros adicionales en feature_engineering.py")
    print("   - Problemas en la función de pivoteo")
    print("   - Timestamps con zona horaria diferente")

print("\n" + "=" * 80)