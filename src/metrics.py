# src/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _as_numpy(y_true, y_pred):
    """Helper interno para asegurar que son arrays 1D."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return y_true, y_pred


def MAE(y_true, y_pred):
    """
    Mean Absolute Error (MAE) entre y_true e y_pred.
    Devuelve un escalar en mismas unidades que y (segundos en tu caso).
    """
    y_true, y_pred = _as_numpy(y_true, y_pred)
    return mean_absolute_error(y_true, y_pred)


def RMSE(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) entre y_true e y_pred.
    Escala igual que y, pero penaliza más los errores grandes.
    """
    y_true, y_pred = _as_numpy(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def R2(y_true, y_pred):
    """
    Coeficiente de determinación R^2.
    1 = perfecto, 0 = modelo igual a promedio, <0 peor que el promedio.
    """
    y_true, y_pred = _as_numpy(y_true, y_pred)
    return r2_score(y_true, y_pred)


def regression_report(y_true, y_pred, rounded=4):
    """
    Devuelve un dict con MAE, RMSE y R2 para loguear/imprimir fácil.
    """
    mae = MAE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    r2 = R2(y_true, y_pred)

    return {
        "MAE": round(mae, rounded),
        "RMSE": round(rmse, rounded),
        "R2": round(r2, rounded),
    }

# =============================================================================
# FUNCIONES PARA COMPARACIÓN DE ESTRATEGIAS Y TIEMPOS
# =============================================================================

def calculate_comparison_statistics(df_optimal, df_real_sim, df_real_actual):
    """
    Calcula estadísticas de comparación entre tres estrategias:
    - Óptima (simulada con mejor estrategia)
    - Real simulada (simulada con estrategia que usó el piloto)
    - Real actual (datos reales de la carrera)
    
    Parameters:
    -----------
    df_optimal : pd.DataFrame
        DataFrame con tiempos de vuelta de estrategia óptima simulada
    df_real_sim : pd.DataFrame
        DataFrame con tiempos de vuelta de estrategia real simulada
    df_real_actual : pd.DataFrame
        DataFrame con tiempos de vuelta reales de carrera
        
    Returns:
    --------
    dict : Diccionario con estadísticas de comparación
    """
    # Calcular tiempos totales
    total_optimal = df_optimal["LapTime_s"].sum()
    total_real_sim = df_real_sim["LapTime_s"].sum()
    total_real_actual = df_real_actual["LapTime_s"].sum()
    
    # Calcular diferencias
    diff_optimal_vs_real_sim = total_real_sim - total_optimal
    diff_optimal_vs_real_actual = total_real_actual - total_optimal
    diff_real_sim_vs_real_actual = total_real_actual - total_real_sim
    
    # Calcular tiempos promedio por vuelta
    avg_optimal = df_optimal["LapTime_s"].mean()
    avg_real_sim = df_real_sim["LapTime_s"].mean()
    avg_real_actual = df_real_actual["LapTime_s"].mean()
    
    # Calcular MAE entre simulación y realidad
    mae_sim = mean_absolute_error(df_real_actual["LapTime_s"], df_real_sim["LapTime_s"])
    
    return {
        "total_optimal": total_optimal,
        "total_real_sim": total_real_sim,
        "total_real_actual": total_real_actual,
        "diff_optimal_vs_real_sim": diff_optimal_vs_real_sim,
        "diff_optimal_vs_real_actual": diff_optimal_vs_real_actual,
        "diff_real_sim_vs_real_actual": diff_real_sim_vs_real_actual,
        "avg_optimal": avg_optimal,
        "avg_real_sim": avg_real_sim,
        "avg_real_actual": avg_real_actual,
        "mae_sim": mae_sim,
    }


def print_comparison_report(stats, strategy_names):
    """
    Imprime un reporte formateado con las estadísticas de comparación.
    
    Parameters:
    -----------
    stats : dict
        Diccionario de estadísticas retornado por calculate_comparison_statistics
    strategy_names : dict
        Diccionario con claves 'optimal', 'real_sim', 'real_actual' y sus nombres
    """
    def format_time(seconds):
        """Formatea segundos a MM:SS.mmm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    print("\n" + "="*60)
    print("COMPARACIÓN DE ESTRATEGIAS - RESUMEN")
    print("="*60)
    
    print(f"\n📊 TIEMPOS TOTALES:")
    print(f"  • {strategy_names.get('optimal', 'Óptima')}: {format_time(stats['total_optimal'])} ({stats['total_optimal']:.3f}s)")
    print(f"  • {strategy_names.get('real_sim', 'Real simulada')}: {format_time(stats['total_real_sim'])} ({stats['total_real_sim']:.3f}s)")
    print(f"  • {strategy_names.get('real_actual', 'Real actual')}: {format_time(stats['total_real_actual'])} ({stats['total_real_actual']:.3f}s)")
    
    print(f"\n⏱️  DIFERENCIAS DE TIEMPO:")
    print(f"  • Óptima vs Real simulada: {stats['diff_optimal_vs_real_sim']:+.3f}s")
    print(f"  • Óptima vs Real actual: {stats['diff_optimal_vs_real_actual']:+.3f}s")
    print(f"  • Real simulada vs Real actual: {stats['diff_real_sim_vs_real_actual']:+.3f}s")
    
    print(f"\n📈 TIEMPOS PROMEDIO POR VUELTA:")
    print(f"  • {strategy_names.get('optimal', 'Óptima')}: {stats['avg_optimal']:.3f}s")
    print(f"  • {strategy_names.get('real_sim', 'Real simulada')}: {stats['avg_real_sim']:.3f}s")
    print(f"  • {strategy_names.get('real_actual', 'Real actual')}: {stats['avg_real_actual']:.3f}s")
    
    print(f"\n🎯 PRECISIÓN DEL MODELO:")
    print(f"  • MAE (simulación vs realidad): {stats['mae_sim']:.3f}s por vuelta")
    
    print("="*60 + "\n")


#METRICAS PARA REPRESENTACIONES

def cluster_purity(labels_true, labels_pred):
    """
    Calcula pureza de clustering:
    sum_k (max_j n_{k,j}) / N
    donde k = cluster, j = clase real.
    """
    contingency = pd.crosstab(labels_pred, labels_true)
    return np.sum(contingency.max(axis=1)) / np.sum(contingency.values)
