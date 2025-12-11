import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

# CONSTANTES DE COLORES
COMPOUND_COLORS = {
    "SOFT": "red",
    "MEDIUM": "yellow", 
    "HARD": "grey",
}

# FUNCIONES PARA EDA

def eda_practice(col_laps, title_prefix="FP"):
    # Aseguramos columna LapTime_s
    if "LapTime_s" not in col_laps.columns:
        col_laps = col_laps.copy()
        col_laps["LapTime_s"] = col_laps["LapTime"].dt.total_seconds()
    
    # 1) LapTime vs LapNumber
    plt.figure(figsize=(10, 4))
    plt.plot(col_laps["LapNumber"], col_laps["LapTime_s"], marker="o")
    plt.xlabel("Número de vuelta")
    plt.ylabel("Tiempo de vuelta [s]")
    plt.title(f"{title_prefix} - Evolución del tiempo de vuelta (Colapinto)")
    plt.grid(True)
    plt.show()
    
    # 2) Histograma de tiempos de vuelta
    plt.figure(figsize=(6, 4))
    plt.hist(col_laps["LapTime_s"], bins=15, edgecolor="black")
    plt.xlabel("Tiempo de vuelta [s]")
    plt.ylabel("Frecuencia")
    plt.title(f"{title_prefix} - Distribución de tiempos de vuelta (Colapinto)")
    plt.grid(axis="y")
    plt.show()
    
    # 3) LapTime vs Stint (si existe)
    if "Stint" in col_laps.columns:
        plt.figure(figsize=(8, 4))
        for stint, df_stint in col_laps.groupby("Stint"):
            plt.plot(df_stint["LapNumber"], df_stint["LapTime_s"],
                     marker="o", linestyle="-", label=f"Stint {stint}")
        plt.xlabel("Número de vuelta")
        plt.ylabel("Tiempo de vuelta [s]")
        plt.title(f"{title_prefix} - LapTime por stint (Colapinto)")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # 4) LapTime vs TyreLife coloreado por Compound (si existen)
    if "TyreLife" in col_laps.columns and "Compound" in col_laps.columns:
        plt.figure(figsize=(8, 5))
        compounds = col_laps["Compound"].dropna().unique()
        for comp in compounds:
            df_c = col_laps[col_laps["Compound"] == comp]
            plt.scatter(df_c["TyreLife"], df_c["LapTime_s"],
                        alpha=0.7, label=comp)
        plt.xlabel("TyreLife [vueltas]")
        plt.ylabel("LapTime [s]")
        plt.title(f"{title_prefix} - LapTime vs TyreLife por compuesto (Colapinto)")
        plt.legend(title="Compound")
        plt.grid(True)
        plt.show()
    
    # 5) LapTime coloreado por TrackStatus (si existe)
    if "TrackStatus" in col_laps.columns:
        status_codes = {st: i for i, st in enumerate(col_laps["TrackStatus"].astype(str).unique())}
        col_laps_plot = col_laps.copy()
        col_laps_plot["TrackStatus_code"] = col_laps_plot["TrackStatus"].astype(str).map(status_codes)
        
        plt.figure(figsize=(10, 4))
        plt.scatter(col_laps_plot["LapNumber"], col_laps_plot["LapTime_s"],
                    c=col_laps_plot["TrackStatus_code"], cmap="tab10")
        cbar = plt.colorbar()
        cbar.set_ticks(list(status_codes.values()))
        cbar.set_ticklabels(list(status_codes.keys()))
        plt.xlabel("Número de vuelta")
        plt.ylabel("LapTime [s]")
        plt.title(f"{title_prefix} - LapTime coloreado por TrackStatus (Colapinto)")
        plt.grid(True)
        plt.show()


# FUNCIONES PARA METRICAS
# src/plots.py

import numpy as np
import matplotlib.pyplot as plt


def y_true_vs_y_pred(y_true, y_pred, title="y_true vs y_pred", ax=None):
    """
    Scatter de y_true vs y_pred con línea diagonal y=x.
    Útil para ver qué tan bien se alinean las predicciones.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("y_true (LapTime_s real)")
    ax.set_ylabel("y_pred (LapTime_s predicho)")
    ax.set_title(title)
    ax.grid(True)

    if ax is None:
        plt.show()


def residuals_hist(y_true, y_pred, bins=20, title="Histograma de residuos", ax=None):
    """
    Histograma de residuos (y_true - y_pred).
    Ayuda a ver si los errores están centrados o sesgados.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(residuals, bins=bins, edgecolor="black")
    ax.set_xlabel("Residuo (y_true - y_pred) [s]")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    ax.grid(axis="y")

    if ax is None:
        plt.show()


# FUNCIONES PARA REPRESENTACIONES



def latent_scatter(Z, labels, title="", ax=None):
    """
    Scatter 2D del espacio latente Z (shape: [n_samples, 2]),
    coloreado por 'labels' (array-like, categoría por punto).
    """
    Z = np.asarray(Z)
    labels = np.asarray(labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=pd.factorize(labels)[0], alpha=0.7)
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(title)
    ax.grid(True)

    # Leyenda discreta
    unique_labels = pd.unique(labels)
    handles = []
    for i, lab in enumerate(unique_labels):
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="", label=str(lab))
        )
    ax.legend(handles=handles, title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")

    if ax is None:
        plt.show()


# =============================================================================
# FUNCIONES PARA SIMULADOR DE ESTRATEGIAS
# =============================================================================

def plot_strategy_lap_times(df_sim, strategy_name, compound_colors=None, title_suffix=""):
    """
    Grafica los tiempos de vuelta por stint para una estrategia simulada.
    
    Parameters:
    -----------
    df_sim : pd.DataFrame
        DataFrame con la simulación (debe tener: LapNumber, Stint, Compound, 
        LapTime_pred_s, TyreLife, PitLoss_s)
    strategy_name : str
        Nombre de la estrategia
    compound_colors : dict, optional
        Diccionario con colores por compuesto. Default: COMPOUND_COLORS
    title_suffix : str, optional
        Sufijo para agregar al título
    """
    if compound_colors is None:
        compound_colors = COMPOUND_COLORS
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot de tiempos predichos por vuelta
    for stint in df_sim["Stint"].unique():
        stint_data = df_sim[df_sim["Stint"] == stint]
        compound = stint_data.iloc[0]["Compound"]
        
        ax.plot(
            stint_data["LapNumber"],
            stint_data["LapTime_pred_s"],
            marker='o',
            color=compound_colors.get(compound, "gray"),
            label=f"Stint {stint} - {compound}",
            linewidth=2,
            markersize=4,
        )
        
        # Marcar pit stops (primera vuelta de cada stint > 1)
        if stint > 1:
            first_lap = stint_data.iloc[0]
            ax.axvline(x=first_lap["LapNumber"], color='red', linestyle='--', alpha=0.5)
            ax.text(first_lap["LapNumber"], ax.get_ylim()[1]*0.98, 
                    f'PIT (+{first_lap["PitLoss_s"]:.0f}s)', 
                    rotation=90, va='top', ha='right', fontsize=9)
    
    ax.set_xlabel("Número de Vuelta", fontsize=12)
    ax.set_ylabel("Tiempo de Vuelta (s)", fontsize=12)
    title = f"Simulación de Estrategia: {strategy_name}"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Resumen de tiempos por stint
    print("\n=== Resumen por Stint ===")
    for stint in df_sim["Stint"].unique():
        stint_data = df_sim[df_sim["Stint"] == stint]
        compound = stint_data.iloc[0]["Compound"]
        avg_time = stint_data["LapTime_pred_s"].mean()
        n_laps = len(stint_data)
        
        print(f"Stint {stint} ({compound}): {n_laps} vueltas - Promedio: {avg_time:.3f}s")


def plot_strategy_comparison(standings, title="Comparación de Estrategias - Monaco GP 2025"):
    """
    Grafica comparación de estrategias con barras horizontales.
    
    Parameters:
    -----------
    standings : pd.DataFrame
        DataFrame con columnas: strategy_name, total_time_s, total_time_str, position
    title : str
        Título del gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gráfico de barras con los tiempos totales
    strategies_names = standings["strategy_name"]
    times_seconds = standings["total_time_s"]
    
    # Diferencias con respecto al mejor
    time_diff = times_seconds - times_seconds.min()
    
    bars = ax.barh(strategies_names, times_seconds, 
                   color=['purple' if i == 0 else 'steelblue' for i in range(len(standings))])
    
    # Añadir etiquetas con tiempo y diferencia
    for i, (time, diff) in enumerate(zip(standings["total_time_str"], time_diff)):
        if diff == 0:
            label = f"{time}"
        else:
            label = f"{time} (+{diff:.1f}s)"
        ax.text(times_seconds.iloc[i], i, label, va='center', ha='left', 
                fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Tiempo Total de Carrera (s)", fontsize=12)
    ax.set_ylabel("Estrategia", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_three_way_comparison(df_optimal, df_real_sim, df_real_actual, 
                               best_strategy_name, title_suffix=""):
    """
    Grafica comparación de tres versiones: óptima, real simulada, y real actual.
    Retorna diccionario con estadísticas calculadas.
    
    Parameters:
    -----------
    df_optimal : pd.DataFrame
        Simulación de estrategia óptima (con LapTime_total_s)
    df_real_sim : pd.DataFrame
        Simulación de estrategia real (con LapTime_total_s)
    df_real_actual : pd.DataFrame
        Datos reales de carrera (con LapTime_s)
    best_strategy_name : str
        Nombre de la mejor estrategia
    title_suffix : str, optional
        Sufijo para el título
        
    Returns:
    --------
    dict : Diccionario con estadísticas comparativas
    """
    # Asegurar que estén ordenados
    df_optimal = df_optimal.sort_values("LapNumber").copy()
    df_real_sim = df_real_sim.sort_values("LapNumber").copy()
    df_real_actual = df_real_actual.sort_values("LapNumber").copy()
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Tiempos de vuelta comparativos (incluyendo penalización de pit stops)
    ax.plot(
        df_optimal["LapNumber"],
        df_optimal["LapTime_total_s"],
        marker='o',
        color='green',
        label=f'Estrategia Óptima ({best_strategy_name}) - Simulada',
        linewidth=2,
        markersize=4,
        alpha=0.8
    )
    
    ax.plot(
        df_real_sim["LapNumber"],
        df_real_sim["LapTime_total_s"],
        marker='s',
        color='blue',
        label='Estrategia Real (H-M-M) - Simulada',
        linewidth=2,
        markersize=4,
        alpha=0.8
    )
    
    ax.plot(
        df_real_actual["LapNumber"],
        df_real_actual["LapTime_s"],
        marker='^',
        color='red',
        label='Tiempos Reales de Carrera',
        linewidth=2,
        markersize=4,
        alpha=0.8
    )
    
    # Marcar pit stops de estrategia óptima
    for stint in df_optimal["Stint"].unique():
        if stint > 1:
            first_lap_stint = df_optimal[df_optimal["Stint"] == stint].iloc[0]
            ax.axvline(x=first_lap_stint["LapNumber"], color='green', 
                       linestyle='--', alpha=0.3, linewidth=1.5)
            ax.text(first_lap_stint["LapNumber"], ax.get_ylim()[1]*0.98,
                    f'PIT\n(Opt)', rotation=0, va='top', ha='center', 
                    fontsize=8, color='green', fontweight='bold')
    
    # Marcar pit stops de estrategia real
    for stint in df_real_actual["Stint"].unique():
        if stint > 1:
            first_lap_stint = df_real_actual[df_real_actual["Stint"] == stint].iloc[0]
            ax.axvline(x=first_lap_stint["LapNumber"], color='red', 
                       linestyle=':', alpha=0.3, linewidth=1.5)
            ax.text(first_lap_stint["LapNumber"], ax.get_ylim()[1]*0.93,
                    f'PIT\n(Real)', rotation=0, va='top', ha='center', 
                    fontsize=8, color='red', fontweight='bold')
    
    ax.set_xlabel("Número de Vuelta", fontsize=12, fontweight='bold')
    ax.set_ylabel("Tiempo de Vuelta (s)", fontsize=12, fontweight='bold')
    title = "Comparación de Tiempos de Vuelta: Óptima vs Real Simulada vs Real"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calcular tiempos acumulados para estadísticas
    cumulative_optimal = df_optimal["LapTime_total_s"].cumsum()
    cumulative_real_sim = df_real_sim["LapTime_total_s"].cumsum()
    cumulative_real_actual = df_real_actual["LapTime_s"].cumsum()
    
    # Diferencias acumuladas (respecto a la óptima)
    diff_real_sim = cumulative_real_sim.values - cumulative_optimal.values
    diff_real_actual = cumulative_real_actual.values - cumulative_optimal.values
    
    # Retornar estadísticas con los nombres correctos que espera print_comparison_report
    from sklearn.metrics import mean_absolute_error
    
    n_laps = len(df_optimal)
    total_optimal = cumulative_optimal.iloc[-1]
    total_real_sim = cumulative_real_sim.iloc[-1]
    total_real_actual = cumulative_real_actual.iloc[-1]
    
    # Calcular MAE entre simulación y realidad
    mae_sim = mean_absolute_error(df_real_actual["LapTime_s"], df_real_sim["LapTime_total_s"])
    
    return {
        "total_optimal": total_optimal,
        "total_real_sim": total_real_sim,
        "total_real_actual": total_real_actual,
        "diff_optimal_vs_real_sim": total_real_sim - total_optimal,
        "diff_optimal_vs_real_actual": total_real_actual - total_optimal,
        "diff_real_sim_vs_real_actual": total_real_actual - total_real_sim,
        "avg_optimal": df_optimal['LapTime_total_s'].mean(),
        "avg_real_sim": df_real_sim['LapTime_total_s'].mean(),
        "avg_real_actual": df_real_actual['LapTime_s'].mean(),
        "mae_sim": mae_sim,
    }


def plot_leaderboard_comparison(leaderboard_comparison, title="Clasificación Monaco GP 2025",
                                 subtitle="Colapinto: Real vs Mejor Estrategia Simulada"):
    """
    Grafica el leaderboard con comparación de pilotos.
    
    Parameters:
    -----------
    leaderboard_comparison : pd.DataFrame
        DataFrame con columnas: Driver, Time_s, New_Position
    title : str
        Título principal
    subtitle : str
        Subtítulo
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Filtrar solo los que tienen tiempo válido
    leaderboard_plot = leaderboard_comparison[leaderboard_comparison["Time_s"].notna()].copy()
    
    # Colores: resaltar a Colapinto
    colors = []
    for driver in leaderboard_plot["Driver"]:
        if driver == "COL":
            colors.append("blue")  # Azul para real
        elif driver == "COL*":
            colors.append("green")  # Verde para simulado
        else:
            colors.append("gray")  # Gris para resto
    
    # Gráfico horizontal
    bars = ax.barh(
        leaderboard_plot["Driver"],
        leaderboard_plot["Time_s"],
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Añadir etiquetas de posición
    for idx, row in leaderboard_plot.iterrows():
        pos = row["New_Position"]
        time = row["Time_s"]
        driver = row["Driver"]
        
        # Etiqueta de posición
        label = f"P{int(pos)}"
        if driver in ["COL", "COL*"]:
            fontweight = 'bold'
            fontsize = 10
        else:
            fontweight = 'normal'
            fontsize = 9
        
        ax.text(time, driver, f"  {label}", va='center', ha='left', 
                fontsize=fontsize, fontweight=fontweight)
    
    ax.set_xlabel("Tiempo Total (s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Piloto", fontsize=12, fontweight='bold')
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=14, fontweight='bold', pad=20)
    
    # Leyenda
    legend_elements = [
        Patch(facecolor='blue', label='Colapinto (Estrategia Real)'),
        Patch(facecolor='green', label='Colapinto (Mejor Estrategia)'),
        Patch(facecolor='gray', label='Otros Pilotos')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Primer lugar arriba
    
    plt.tight_layout()
    plt.show()
