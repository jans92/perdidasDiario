# 📖 Guía de Arquitectura y Mantenimiento: Gemelo Predictivo BALATON

## 1. Visión General del Sistema
Este proyecto es un sistema de **Mantenimiento Predictivo en Tiempo Real** diseñado para anticipar fallos mecánicos/eléctricos en la línea de producción BALATON.
Utiliza un modelo de Machine Learning (LightGBM) que evalúa constantemente el estado de la máquina utilizando una "Ventana Móvil" de 168 horas de datos históricos, emitiendo un score de riesgo para la próxima hora.

---

## 2. Arquitectura de Ejecución (Los Scripts)
El ecosistema se compone de 4 piezas fundamentales:

### A. `main.py` (El Motor - Se ejecuta cada hora)
* **Función:** Es el orquestador principal. Descarga los datos de las bases de datos `bui_perdida` y `bui_pm_ewo`, aplica los filtros para la línea BALATON (ID 52777) y genera las variables matemáticas (Feature Engineering).
* **Predicción:** Genera una única predicción exacta para la hora siguiente a su ejecución (Ej: Si corre a las 14:00, predice el riesgo de las 15:00).

### B. `save_predictions.py` (El Gestor de BBDD - Llamado por main.py)
* **Función:** Guarda las predicciones en la tabla `bui_predicciones_hora`.
* **Lógica de Inserción (Upsert):** Antes de insertar el dato de la hora calculada, ejecuta un `DELETE` de esa misma hora. Esto garantiza que NUNCA haya predicciones duplicadas para la misma máquina y hora, asegurando que el Dashboard lea un historial perfecto.
* **Umbrales:** * Riesgo Moderado: `0.37` (Activa la alerta en el Dashboard)
  * Riesgo Crítico: `0.60`

### C. `dashboardv3.py` (La Interfaz - Streamlit)
* **Función:** Lee la tabla de predicciones y las muestra en un *Timeline* hora a hora.
* **Código de Colores:**
  * 🟢 **Verde (Predicción Acertada):** El modelo avisó y la máquina efectivamente falló.
  * 🔴 **Rojo (Falsa Alarma):** El modelo avisó, pero pasaron 24h y la máquina no falló.
  * 🟠 **Naranja (Predicción - Alerta):** El modelo avisa de un riesgo actual/futuro que aún está pendiente de evaluarse.

### D. `update_targets.py` (El Evaluador - Ejecución Diaria/Manual)
* **Función:** Mira hacia atrás en el tiempo. Busca predicciones que se hicieron hace más de 24 horas y comprueba en la base de datos de producción real si hubo una parada de más de 20 min (Crítica) o 60 min (Larga).
* **Resultado:** Actualiza las columnas `fl_target_real` (hubo fallo) y `fl_acierto` (el modelo acertó) para alimentar los KPIs y los colores Verde/Rojo del Dashboard.

---

## 3. Automatización (Programador de Tareas)
El sistema está automatizado en Windows a través del **Task Scheduler (schtasks)**. 
* El script `main.py` se ejecuta **cada hora en punto**.
* El comando configurado en el servidor para crear la tarea fue:
  `schtasks /Create /SC HOURLY /TN "Predictivo_BALATON_Hora" /TR "cmd.exe /c cd /d C:\Ruta\Al\Proyecto && python main.py" /F`

---

## 4. Mantenimiento a Futuro (Reentrenamiento)
Actualmente, el modelo (`model_optimized.txt`) tiene un **AUC-ROC del 80%** y un **Recall del 80%**, por lo que **NO es necesario reentrenarlo a corto plazo**.

**¿Cuándo reentrenar?**
Se recomienda reentrenar el modelo cada 3 a 6 meses para combatir el *Model Drift* (desgaste natural por cambio de piezas, envejecimiento de la máquina o cambios de estacionalidad).

**¿Cómo reentrenar?**
1. Abrir el archivo `pipeline_reentrenamiento.ipynb`.
2. Ejecutar las celdas secuencialmente. El script extraerá un nuevo set masivo de datos históricos.
3. El motor de optimización (Optuna) buscará los mejores parámetros automáticamente.
4. Se generará un nuevo archivo `model_optimized.txt` y te indicará el nuevo umbral óptimo.
5. Si el umbral óptimo cambia, actualizar ese número en `inference_config.json` y en las variables globales de `save_predictions.py`.