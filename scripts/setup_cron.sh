#!/bin/bash
# ============================================================
# Setup Cron Job para Predictor de Mantenimiento
# ============================================================

PROJECT_DIR="./Run_predictor_daily"
PYTHON_PATH="$PROJECT_DIR/.conda/bin/python"
HORA_EJECUCION="5"  # 5:00 AM
MINUTO_EJECUCION="0"

echo " Configurando cron job..."

# Verificar que existe el directorio
if [ ! -d "$PROJECT_DIR" ]; then
    echo " Error: Directorio $PROJECT_DIR no existe"
    exit 1
fi

# Verificar Python
if [ ! -f "$PYTHON_PATH" ]; then
    echo " Error: Python no encontrado en $PYTHON_PATH"
    exit 1
fi

# Crear directorio de logs si no existe
mkdir -p "$PROJECT_DIR/logs"

# Crear entrada de cron
CRON_ENTRY="$MINUTO_EJECUCION $HORA_EJECUCION * * * cd $PROJECT_DIR && $PYTHON_PATH main.py >> logs/cron.log 2>&1"

# Verificar si ya existe
if crontab -l 2>/dev/null | grep -q "Run_predictor_daily"; then
    echo "  Ya existe un cron job para este proyecto"
    echo "¿Desea reemplazarlo? (s/n)"
    read respuesta
    if [ "$respuesta" != "s" ]; then
        echo "Cancelado"
        exit 0
    fi
    # Eliminar entrada anterior
    crontab -l | grep -v "Run_predictor_daily" | crontab -
fi

# Agregar nueva entrada
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo " Cron job configurado:"
echo "   Hora: $HORA_EJECUCION:$MINUTO_EJECUCION AM (diariamente)"
echo "   Comando: python main.py"
echo "   Log: $PROJECT_DIR/logs/cron.log"
echo ""
echo " Para verificar: crontab -l"
echo " Para editar: crontab -e"