import pandas as pd
import logging
import sys
import os
from pathlib import Path

# Configurar logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Aseguramos que Python encuentre los módulos
sys.path.append(os.getcwd())

try:
    from config_loader import load_config
    from save_predictions import save_predictions_batch
except ImportError as e:
    logger.error(f"Error de importación: {e}")
    logger.error("Asegúrate de que config_loader.py y save_predictions.py están en la misma carpeta.")
    sys.exit(1)

def probar_fix_guardado():
    print("\n" + "="*60)
    print(" INICIANDO TEST RÁPIDO DE GUARDADO")
    print("="*60)

    # 1. CARGAR CONFIGURACIÓN
    try:
        config = load_config()
        print(" Configuración cargada correctamente.")
        
        # DEBUG: Imprimir estructura de la base de datos para verificar
        db_conf = config.get('database', {})
        print(f"ℹ  Estructura detectada en config['database']: {list(db_conf.keys())}")
        if 'target' in db_conf:
             print(f"ℹ  Usuario detectado en target: {db_conf['target'].get('user')}")
        
    except Exception as e:
        print(f" Error cargando config: {e}")
        return

    # 2. BUSCAR EL CSV MÁS RECIENTE
    output_dir = Path('outputs')
    if not output_dir.exists():
        print(" No existe la carpeta 'outputs/'.")
        return

    files = list(output_dir.glob('predictions_*.csv'))
    if not files:
        print(" No hay archivos .csv en 'outputs/'. Ejecuta main.py al menos una vez.")
        return

    # Ordenar por fecha de modificación (el más nuevo primero)
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f" Usando archivo de predicciones: {latest_file.name}")
    
    try:
        predictions_df = pd.read_csv(latest_file)
        print(f" DataFrame cargado: {len(predictions_df)} registros.")
    except Exception as e:
        print(f" Error leyendo el CSV: {e}")
        return

    # 3. CORREGIR EL ERROR DEL ESPACIO EN 'SCORE ' SI EXISTE
    # Esto simula la corrección que ya hiciste en main.py
    if 'score ' in predictions_df.columns:
        print("  Corrigiendo columna 'score ' (con espacio) -> 'score'")
        predictions_df.rename(columns={'score ': 'score'}, inplace=True)

    # 4. INTENTAR GUARDAR
    print("\n Intentando guardar en Base de Datos...")
    try:
        # Obtener versión del modelo
        version = config.get('model', {}).get('version', 'v1.1')
        
        # LLAMADA A LA FUNCIÓN QUE FALLABA
        # Si aplicaste el fix en save_predictions.py (función get_db_engine), esto debería funcionar
        registros = save_predictions_batch(predictions_df, config, de_modelo_version=version)
        
        print("\n" + "="*60)
        print(f" ¡ÉXITO! Se guardaron {registros} registros.")
        print("   La conexión a la BD y la estructura del config.yaml son correctas.")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f" FALLÓ EL GUARDADO: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    probar_fix_guardado()
