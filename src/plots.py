import matplotlib.pyplot as plt


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
