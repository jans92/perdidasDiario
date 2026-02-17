import pandas as pd
from config_loader import load_config
from save_predictions import save_predictions_batch
from pathlib import Path

def test():
    print(" Probando SOLO el guardado en BD...")
    
    # 1. Cargar config
    config = load_config()
    
    # 2. Buscar último CSV
    output_dir = Path('outputs')
    files = list(output_dir.glob('predictions_*.csv'))
    if not files:
        print(" No hay archivos CSV en outputs/ para probar.")
        return
        
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f" Usando archivo: {latest}")
    df = pd.read_csv(latest)
    
    # Pequeño fix si la columna score tiene espacio (por si acaso)
    if 'score ' in df.columns:
        df.rename(columns={'score ': 'score'}, inplace=True)

    # 3. Intentar guardar
    try:
        # Usamos save_predictions_batch que es la que usa main.py
        save_predictions_batch(df, config, de_modelo_version="TEST_v1.1")
        print("\n ¡ÉXITO! La conexión y el guardado funcionan correctamente.")
    except Exception as e:
        print(f"\n ERROR: {e}")
        # Imprimir detalles para ayudar
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
