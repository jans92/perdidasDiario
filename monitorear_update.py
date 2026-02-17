"""
monitorear_targets.py
=====================
Verifica estado de targets diariamente.
"""

from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

load_dotenv()

def get_connection():
    host = os.getenv('DB_DEV_HOST')
    user = os.getenv('DB_DEV_USER')
    password = os.getenv('DB_DEV_PASSWORD')
    database = os.getenv('DB_DEV_NAME')
    port = os.getenv('DB_DEV_PORT', '3306')
    
    connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(connection_string, connect_args={'ssl': {'ssl_mode': 'PREFERRED'}})

def verificar_estado():
    engine = get_connection()
    
    # Targets pendientes (> 24h atrás sin calcular)
    q_pendientes = text("""
        SELECT COUNT(*) as pendientes
        FROM bui_predicciones_hora_dia
        WHERE fe_ventana < DATE_SUB(NOW(), INTERVAL 24 HOUR)
          AND fl_target_real IS NULL
    """)
    
    # Actualizados hoy
    q_hoy = text("""
        SELECT COUNT(*) as actualizados_hoy
        FROM bui_predicciones_hora_dia
        WHERE DATE(fe_actualizado) = CURDATE()
    """)
    
    # Accuracy últimos 7 días
    q_accuracy = text("""
        SELECT 
            ROUND(SUM(fl_acierto) / COUNT(*) * 100, 2) as accuracy_7d
        FROM bui_predicciones_hora_dia
        WHERE fl_target_real IS NOT NULL
          AND fe_ventana >= DATE_SUB(NOW(), INTERVAL 7 DAY)
    """)
    
    pendientes = pd.read_sql(q_pendientes, engine)['pendientes'].values[0]
    actualizados = pd.read_sql(q_hoy, engine)['actualizados_hoy'].values[0]
    accuracy = pd.read_sql(q_accuracy, engine)['accuracy_7d'].values[0]
    
    print("=" * 60)
    print(f"📊 MONITOREO DE TARGETS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"\n   Targets pendientes (>24h):  {pendientes:,}")
    print(f"   Actualizados hoy:            {actualizados:,}")
    print(f"   Accuracy últimos 7 días:     {accuracy:.2f}%")
    
    if pendientes > 1000:
        print("\n   ⚠️  ALERTA: Muchos targets pendientes")
        print("   Verificar ejecución de update_targets.py")
    else:
        print("\n   ✅ Sistema operando correctamente")
    
    print("\n" + "=" * 60)
    
    engine.dispose()

if __name__ == "__main__":
    verificar_estado()
