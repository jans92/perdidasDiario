@echo off
REM Script para ejecutar predicciones cada hora

REM Ir a la carpeta del proyecto
cd /d C:\Users\Janire.Perez\Apache24\htdocs\dataia\prueba2\predictive-maintenance-mvp-main

REM Activar entorno virtual (CMD)
call venv\Scripts\activate.bat

REM Crear carpeta logs si no existe
if not exist logs mkdir logs

REM Ejecutar main.py con logging a archivo
echo ========================================== >> logs\cron.log
echo Ejecucion: %date% %time% >> logs\cron.log
echo ========================================== >> logs\cron.log
python main.py >> logs\cron.log 2>&1

echo Ejecucion completada
