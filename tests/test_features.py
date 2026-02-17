import lightgbm as lgb
import json
from pathlib import Path

print("=" * 80)
print(" DIAGNÓSTICO DE MODELO Y VERSIÓN")
print("=" * 80)

# 1. Versión actual de LightGBM
print(f"\n Versión de LightGBM instalada: {lgb.__version__}")

# 2. Verificar que el archivo existe y su tamaño
model_path = Path("/Users/juanmanuel/Desktop/Run_predictor_daily/model/model_optimized.txt")
print(f"\n Archivo del modelo:")
print(f"   Existe: {model_path.exists()}")
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024**2)
    print(f"   Tamaño: {size_mb:.2f} MB")
    
    # Leer primeras líneas
    with open(model_path, 'r') as f:
        first_lines = [f.readline() for _ in range(5)]
    print(f"\n Primeras líneas del archivo:")
    for i, line in enumerate(first_lines, 1):
        print(f"   {i}: {line.strip()[:100]}")

# 3. Verificar inference_config.json
inference_path = model_path.parent / 'inference_config.json'
print(f"\n Archivo inference_config.json:")
print(f"   Existe: {inference_path.exists()}")
if inference_path.exists():
    with open(inference_path, 'r') as f:
        config = json.load(f)
    print(f"   Contenido: {json.dumps(config, indent=2)}")

# 4. Verificar features_utiles.json
features_path = model_path.parent / 'features_utiles.json'
print(f"\n Archivo features_utiles.json:")
print(f"   Existe: {features_path.exists()}")
if features_path.exists():
    with open(features_path, 'r') as f:
        features_data = json.load(f)
    print(f"   Número de features: {len(features_data.get('features', []))}")

print("\n" + "=" * 80)