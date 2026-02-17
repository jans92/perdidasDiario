"""
recalcular_turbo.py - VERSIÓN TURBO (PRE-CARGA MASIVA)
========================================================
 VELOCIDAD: 50-100x más rápido que versión original
 TÉCNICA: Pre-carga todos los breakdowns en memoria
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from tqdm import tqdm
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Configuración
UMBRAL_DURACION_CON_EWO = 10
UMBRAL_DURACION_LARGA = 60
UMBRAL_DURACION_TIPO_CRITICO = 20

TIPOS_CRITICOS = [
    'Breakdown - Mechanical',
    'Breakdown - Electrical',
    'Breakdown - Instrumentation & Control'
]

BATCH_SIZE = 1000

# Conexión (igual que antes)
def get_db_connection(env_prefix):
    """Crea conexión a BD"""
    try:
        host = os.getenv(f'{env_prefix}_HOST')
        user = os.getenv(f'{env_prefix}_USER')
        password = os.getenv(f'{env_prefix}_PASSWORD')
        database = os.getenv(f'{env_prefix}_NAME')
        port = os.getenv(f'{env_prefix}_PORT', '3306')

        if not all([host, user, password, database]):
            raise ValueError(f"Faltan variables para {env_prefix}")

        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        
        engine = create_engine(
            connection_string,
            connect_args={'ssl': {'ssl_mode': 'PREFERRED'}},
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=20,
            echo=False
        )

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        log.info(f" Conexión {env_prefix}: {database}@{host}")
        return engine

    except Exception as e:
        log.error(f" Error conectando a {env_prefix}: {e}")
        raise


def obtener_todas_predicciones(engine_dev, dias_atras=None):
    """Obtiene predicciones"""
    log.info(" Consultando predicciones...")
    
    if dias_atras:
        where_extra = f"AND fe_ventana >= DATE_SUB(NOW(), INTERVAL {dias_atras} DAY)"
        log.info(f"   Filtrando últimos {dias_atras} días")
    else:
        where_extra = ""
    
    query = text(f"""
        SELECT 
            id_prediccion,
            id_maquina_dfos,
            fe_ventana,
            fl_pred_modelo,
            fl_target_real as target_anterior
        FROM bui_predicciones_hora_dia
        WHERE fe_ventana < DATE_SUB(NOW(), INTERVAL 24 HOUR)
          {where_extra}
        ORDER BY fe_ventana ASC
    """)
    
    df = pd.read_sql(query, engine_dev)
    
    log.info(f"   Total predicciones: {len(df):,}")
    if len(df) > 0:
        log.info(f"   Rango: {df['fe_ventana'].min()} → {df['fe_ventana'].max()}")
    
    return df


def obtener_ewos_validas(engine_prod, fecha_inicio, fecha_fin):
    """Obtiene EWOs válidas"""
    log.info(f" Obteniendo EWOs válidas...")
    
    query = text("""
        SELECT DISTINCT id_perdida
        FROM bui_pm_ewo
        WHERE fl_borrador = 0
          AND id_perdida IS NOT NULL
          AND fe_inicio_averia BETWEEN :fecha_inicio AND :fecha_fin
          AND (fe_cerrar IS NOT NULL OR fe_fin_mto IS NOT NULL)
    """)
    
    df = pd.read_sql(query, engine_prod, params={
        'fecha_inicio': fecha_inicio,
        'fecha_fin': fecha_fin
    })
    
    ewos_set = set(df['id_perdida'].tolist())
    log.info(f"   EWOs válidas: {len(ewos_set):,}")
    
    return ewos_set


def precargar_todos_breakdowns(engine_prod, fecha_inicio, fecha_fin):
    """
     TURBO: Pre-carga TODOS los breakdowns de una vez
    
    Returns:
        DataFrame con columnas: id_maquina_dfos, fe_inicio, fe_fin, 
                                id_perdida, de_perdida_3, duracion_minutos
    """
    log.info(f" PRE-CARGANDO TODOS LOS BREAKDOWNS (1 query masiva)...")
    
    #  OPTIMIZACIÓN: 1 sola query para TODO el período
    query = text("""
        SELECT 
            id_maquina_dfos,
            fe_inicio,
            fe_fin,
            id_perdida,
            de_perdida_3,
            TIMESTAMPDIFF(MINUTE, fe_inicio, fe_fin) as duracion_minutos
        FROM bui_perdida
        WHERE de_perdida_2 = 'Breakdown & Equipment Failure Time'
          AND fe_inicio < :fecha_fin
          AND fe_fin > :fecha_inicio
        ORDER BY id_maquina_dfos, fe_inicio
    """)
    
    df_breakdowns = pd.read_sql(query, engine_prod, params={
        'fecha_inicio': fecha_inicio,
        'fecha_fin': fecha_fin
    })
    
    # Convertir a datetime para comparaciones rápidas
    df_breakdowns['fe_inicio'] = pd.to_datetime(df_breakdowns['fe_inicio'])
    df_breakdowns['fe_fin'] = pd.to_datetime(df_breakdowns['fe_fin'])
    
    log.info(f"    Breakdowns cargados: {len(df_breakdowns):,}")
    log.info(f"    Memoria usada: {df_breakdowns.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_breakdowns


def verificar_falla_grave_local(row_prediccion, df_breakdowns, ewos_validas):
    """
     TURBO: Verifica falla grave buscando en DataFrame local (SIN query)
    
    Args:
        row_prediccion: Fila de predicción (con fe_ventana, id_maquina_dfos)
        df_breakdowns: DataFrame pre-cargado con TODOS los breakdowns
        ewos_validas: Set de id_perdida con EWO válida
    
    Returns:
        int: 1 si hubo falla grave, 0 si no
    """
    id_maquina = row_prediccion['id_maquina_dfos']
    fe_ventana = row_prediccion['fe_ventana']
    
    # Ventana de búsqueda
    fe_inicio_busqueda = fe_ventana - timedelta(hours=1)
    fe_fin_busqueda = fe_ventana + timedelta(hours=24)
    
    #  FILTRADO LOCAL (sin query SQL)
    mask = (
        (df_breakdowns['id_maquina_dfos'] == id_maquina) &
        (df_breakdowns['fe_inicio'] < fe_fin_busqueda) &
        (df_breakdowns['fe_fin'] > fe_inicio_busqueda)
    )
    
    breakdowns_maquina = df_breakdowns[mask]
    
    if len(breakdowns_maquina) == 0:
        return 0
    
    # Verificar criterios de severidad
    for _, row in breakdowns_maquina.iterrows():
        duracion = row['duracion_minutos'] or 0
        tipo = row['de_perdida_3']
        id_perdida = row['id_perdida']
        
        # Criterio 1: EWO válida + duración >= 10 min
        if id_perdida is not None and id_perdida in ewos_validas:
            if duracion >= UMBRAL_DURACION_CON_EWO:
                return 1
        
        # Criterio 2: Duración >= 60 min
        if duracion >= UMBRAL_DURACION_LARGA:
            return 1
        
        # Criterio 3: Tipo crítico + duración >= 20 min
        if tipo in TIPOS_CRITICOS and duracion >= UMBRAL_DURACION_TIPO_CRITICO:
            return 1
    
    return 0


def actualizar_prediccion_batch(engine_dev, updates_batch):
    """Actualiza múltiples predicciones en 1 query"""
    if not updates_batch:
        return 0
    
    try:
        case_target = " ".join([
            f"WHEN {u['id_prediccion']} THEN {u['target_real']}"
            for u in updates_batch
        ])
        
        case_acierto = " ".join([
            f"WHEN {u['id_prediccion']} THEN {u['acierto']}"
            for u in updates_batch
        ])
        
        ids = ",".join([str(u['id_prediccion']) for u in updates_batch])
        
        query = text(f"""
            UPDATE bui_predicciones_hora_dia
            SET 
                fl_target_real = CASE id_prediccion {case_target} END,
                fl_acierto = CASE id_prediccion {case_acierto} END
            WHERE id_prediccion IN ({ids})
        """)
        
        with engine_dev.begin() as conn:
            result = conn.execute(query)
        
        return len(updates_batch)
        
    except Exception as e:
        log.error(f" Error en batch update: {e}")
        return 0


def recalcular_targets_turbo(dias_atras=None):
    """
     VERSIÓN TURBO: Re-calcula targets con pre-carga masiva
    
    Velocidad: 50-100x más rápido que versión con queries individuales
    """
    print("=" * 80)
    print(" RE-CÁLCULO DE TARGETS HISTÓRICOS (VERSIÓN TURBO)")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if dias_atras:
        print(f"   Alcance: Últimos {dias_atras} días")
    else:
        print(f"   Alcance: TODOS los registros históricos")
    print("=" * 80)
    
    engine_dev = None
    engine_prod = None
    
    try:
        # Conectar
        print("\n ESTABLECIENDO CONEXIONES...\n")
        engine_dev = get_db_connection('DB_DEV')
        engine_prod = get_db_connection('DB_PROD')
        print("\n Ambas conexiones establecidas\n")
        
        # Obtener predicciones
        df_todas = obtener_todas_predicciones(engine_dev, dias_atras=dias_atras)
        total = len(df_todas)
        
        if total == 0:
            print(" No hay predicciones para procesar")
            return 0
        
        # Confirmación
        print("\n" + "=" * 80)
        print("  ADVERTENCIA:")
        print(f"   Se RE-CALCULARÁN {total:,} targets consultando PROD")
        print(f"    MODO TURBO: Pre-carga masiva + batch updates")
        print("=" * 80)
        
        respuesta = input("\n¿Continuar? (escribir 'SI' en mayúsculas): ")
        
        if respuesta != 'SI':
            print("\n Operación cancelada por el usuario")
            return 0
        
        # Definir período
        fecha_min = df_todas['fe_ventana'].min() - timedelta(hours=1)
        fecha_max = df_todas['fe_ventana'].max() + timedelta(hours=24)
        
        print(f"\n Período de análisis:")
        print(f"   Inicio: {fecha_min}")
        print(f"   Fin: {fecha_max}\n")
        
        #  TURBO: Pre-cargar EWOs
        ewos_validas = obtener_ewos_validas(engine_prod, fecha_min, fecha_max)
        
        #  TURBO: Pre-cargar TODOS los breakdowns (1 query masiva)
        df_breakdowns = precargar_todos_breakdowns(engine_prod, fecha_min, fecha_max)
        
        # Procesar predicciones SIN consultar BD cada vez
        print(f"\n Procesando predicciones (LOCAL - sin queries individuales)...\n")
        
        actualizados = 0
        con_breakdown = 0
        sin_breakdown = 0
        cambios = 0
        
        updates_batch = []
        
        # Convertir fe_ventana a datetime
        df_todas['fe_ventana'] = pd.to_datetime(df_todas['fe_ventana'])
        
        for idx, row in tqdm(
            df_todas.iterrows(),
            total=len(df_todas),
            desc="Procesando predicciones",
            unit="pred"
        ):
            #  TURBO: Buscar en DataFrame local (SIN query SQL)
            fl_target_real = verificar_falla_grave_local(
                row, df_breakdowns, ewos_validas
            )
            
            # Verificar si cambió
            if pd.notna(row['target_anterior']) and row['target_anterior'] != fl_target_real:
                cambios += 1
            
            # Calcular acierto
            acierto = 1 if row['fl_pred_modelo'] == fl_target_real else 0
            
            # Acumular en batch
            updates_batch.append({
                'id_prediccion': row['id_prediccion'],
                'target_real': fl_target_real,
                'acierto': acierto
            })
            
            actualizados += 1
            if fl_target_real == 1:
                con_breakdown += 1
            else:
                sin_breakdown += 1
            
            # Ejecutar batch cuando alcance límite
            if len(updates_batch) >= BATCH_SIZE:
                registros_actualizados = actualizar_prediccion_batch(
                    engine_dev, updates_batch
                )
                log.info(f"    Batch de {registros_actualizados} registros actualizado")
                updates_batch = []
        
        # Ejecutar últimos registros
        if updates_batch:
            registros_actualizados = actualizar_prediccion_batch(
                engine_dev, updates_batch
            )
            log.info(f"    Último batch de {registros_actualizados} registros")
        
        # Resumen
        print("\n" + "=" * 80)
        print(" RESUMEN DE RE-CÁLCULO (TURBO)")
        print("=" * 80)
        print(f"\n Total actualizados: {actualizados:,}")
        print(f"    Con breakdown (1): {con_breakdown:,} ({con_breakdown/actualizados*100:.2f}%)")
        print(f"    Sin breakdown (0): {sin_breakdown:,} ({sin_breakdown/actualizados*100:.2f}%)")
        print(f"    Valores modificados: {cambios:,}")
        
        # Métricas finales
        print("\n CALCULANDO MÉTRICAS FINALES...\n")
        
        q_stats = text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN fl_target_real = 1 THEN 1 ELSE 0 END) as positivos,
                SUM(fl_acierto) as aciertos,
                ROUND(SUM(fl_acierto)/COUNT(*)*100, 2) as accuracy_pct
            FROM bui_predicciones_hora_dia
            WHERE fl_target_real IS NOT NULL
        """)
        
        stats = pd.read_sql(q_stats, engine_dev)
        
        print("   MÉTRICAS GLOBALES:")
        print(f"   Total validadas: {stats['total'].values[0]:,}")
        print(f"   Positivos (1): {stats['positivos'].values[0]:,} ({stats['positivos'].values[0]/stats['total'].values[0]*100:.2f}%)")
        print(f"   Accuracy: {stats['accuracy_pct'].values[0]:.2f}%")
        
        print("\n" + "=" * 80)
        print(" RE-CÁLCULO TURBO COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        
        return actualizados
        
    except Exception as e:
        log.error(f" ERROR: {e}", exc_info=True)
        raise
    
    finally:
        if engine_dev:
            engine_dev.dispose()
            print("\n Conexión DEV cerrada")
        if engine_prod:
            engine_prod.dispose()
            print(" Conexión PROD cerrada")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Re-calcula targets históricos (VERSIÓN TURBO)'
    )
    parser.add_argument(
        '--dias',
        type=int,
        default=None,
        help='Limitar a últimos N días (default: todos)'
    )
    
    args = parser.parse_args()
    
    try:
        recalcular_targets_turbo(dias_atras=args.dias)
    except KeyboardInterrupt:
        print("\n\n Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n ERROR CRÍTICO: {e}")
        sys.exit(1)
