# De la telemetría a la estrategia: tiempos de vuelta y espacios latentes en Fórmula 1

Este repositorio contiene el código, notebooks y material del proyecto final de la materia **Machine Learning (I302)** de la Universidad de San Andrés.

El objetivo del proyecto es usar datos reales de **Fórmula 1** (Mónaco 2025, vía [FastF1](https://docs.fastf1.dev/)) para:

1. **Predecir tiempos de vuelta** de un piloto a partir de variables de contexto (compuesto, stint, vida del neumático, número de vuelta, tipo de sesión, etc.).
2. **Aprender representaciones de baja dimensión** (PCA y autoencoders) para analizar cómo se organizan las vueltas en un espacio latente (por compuesto, piloto ritmo…).
3. **Simular estrategias de neumáticos** usando el modelo de tiempo de vuelta para comparar estrategias alternativas contra la estrategia real de carrera.

---

## Estructura del repositorio

```text
data/
  processed/          # CSV limpios usados para entrenar y evaluar modelos
src/
  preprocessing.py    # Limpieza de sesiones, filtrado FAST/SLOW, TrackStatus
  models.py           # Definición y entrenamiento de modelos
  metrics.py          # MAE, RMSE, R2, regression_report
  plots.py            # Funciones para gráficos (PCA, AE, y vs y_hat, histogramas)
notebooks/
  01_eda_y_splitting.ipynb            
  02_modelado_lap_time.ipynb
  03_representaciones_latentes.ipynb
  04_simulador_estrategias.ipynb
docs/
  Informe_Pini_Carruthers_PF.pdf      # Informe estilo IEEE
  Poster_ML_F1_Pini_Carruthers.pdf    # Póster presentado en el AI Fest de la Universidad de San Andrés
  graficos/                           # Imágenes usadas en informe/póster

---