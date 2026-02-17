@echo off
cd /d C:\Users\Janire.Perez\Apache24\htdocs\dataia\prueba2\predictive-maintenance-mvp-main

call venv\Scripts\activate.bat

if not exist logs mkdir logs

echo ========================================== >> logs\update.log
echo Update Targets: %date% %time% >> logs\update.log
echo ========================================== >> logs\update.log
python update_targets.py >> logs\update.log 2>&1

echo Update completado
