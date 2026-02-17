-- ============================================================================
-- DDL: Tabla de predicciones horarias
-- Sistema de Mantenimiento Predictivo
-- ============================================================================

-- Eliminar tabla si existe (solo para desarrollo)
-- DROP TABLE IF EXISTS bui_predicciones_hora_dia;

CREATE TABLE bui_predicciones_hora_dia (
    -- Identificadores
    id_prediccion INT NOT NULL AUTO_INCREMENT,
    id_maquina_dfos VARCHAR(50) NOT NULL COMMENT 'ID de máquina en DFOS (formato: fabrica_linea-maquina)',
    id_linea INT NOT NULL COMMENT 'ID de línea de producción',
    
    -- Predicción
    fe_ventana DATETIME NOT NULL COMMENT 'Ventana horaria predicha (próximas 24h desde esta hora)',
    nm_score DECIMAL(5,4) NOT NULL COMMENT 'Score del modelo (0.0000-1.0000)',
    de_nivel_riesgo VARCHAR(20) NOT NULL COMMENT 'critico (>=0.40), moderado (0.31-0.40), bajo (0.10-0.31), normal (<0.10)',
    fl_pred_modelo TINYINT NOT NULL COMMENT 'Predicción binaria: 1=falla predicha (score>=0.31), 0=sin falla predicha',
    
    -- Resultado real (se actualiza después de 24h por update_targets.py)
    fl_target_real TINYINT NULL COMMENT 'NULL=pendiente, 0=no hubo falla, 1=hubo falla (Breakdown & Equipment Failure Time)',
    fl_acierto TINYINT NULL COMMENT 'NULL=pendiente, 0=predicción incorrecta, 1=predicción correcta',
    
    -- Metadata
    de_modelo_version VARCHAR(20) NOT NULL COMMENT 'Versión del modelo (ej: v1.1)',
    fe_creado TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Fecha/hora de inserción del registro',
    fe_actualizado TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP COMMENT 'Fecha/hora de actualización (cuando se completa target_real)',
    
    -- Constraints
    PRIMARY KEY (id_prediccion),
    
    -- Índices para consultas frecuentes
    INDEX idx_maquina (id_maquina_dfos),
    INDEX idx_linea (id_linea),
    INDEX idx_ventana (fe_ventana),
    INDEX idx_nivel_riesgo (de_nivel_riesgo),
    INDEX idx_pred_modelo (fl_pred_modelo),
    INDEX idx_pendientes (fl_target_real, fe_ventana) COMMENT 'Para update_targets.py: buscar pendientes de actualizar',
    
    -- Índice único para evitar duplicados (misma máquina + misma ventana horaria)
    UNIQUE INDEX idx_unique_prediccion (id_maquina_dfos, fe_ventana)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Predicciones horarias de averías por máquina - Sistema de Mantenimiento Predictivo';


-- ============================================================================
-- Consultas de ejemplo
-- ============================================================================

-- 1. Predicciones pendientes de actualizar (ya pasaron 24h)
/*
SELECT * FROM bui_predicciones_hora_dia 
WHERE fl_target_real IS NULL 
  AND fe_ventana < NOW() - INTERVAL 24 HOUR
LIMIT 10000;
*/

-- 2. Matriz de confusión del último día
/*
SELECT 
    fl_pred_modelo,
    fl_target_real,
    COUNT(*) as cantidad
FROM bui_predicciones_hora_dia
WHERE fl_target_real IS NOT NULL
  AND DATE(fe_creado) = CURDATE() - INTERVAL 1 DAY
GROUP BY fl_pred_modelo, fl_target_real;
*/

-- 3. Métricas de precisión
/*
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN fl_pred_modelo = 1 AND fl_target_real = 1 THEN 1 ELSE 0 END) as TP,
    SUM(CASE WHEN fl_pred_modelo = 1 AND fl_target_real = 0 THEN 1 ELSE 0 END) as FP,
    SUM(CASE WHEN fl_pred_modelo = 0 AND fl_target_real = 1 THEN 1 ELSE 0 END) as FN,
    SUM(CASE WHEN fl_pred_modelo = 0 AND fl_target_real = 0 THEN 1 ELSE 0 END) as TN,
    SUM(fl_acierto) / COUNT(*) * 100 as accuracy_pct
FROM bui_predicciones_hora_dia
WHERE fl_target_real IS NOT NULL;
*/

-- 4. Alertas críticas actuales (pendientes)
/*
SELECT id_maquina_dfos, id_linea, fe_ventana, nm_score
FROM bui_predicciones_hora_dia
WHERE de_nivel_riesgo = 'critico'
  AND fe_ventana >= NOW()
ORDER BY nm_score DESC;
*/
