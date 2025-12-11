# src/simulator.py

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .preprocessing import add_basic_features


@dataclass
class Strategy:
    """
    Representa una estrategia de neumáticos para toda la carrera.

    Ejemplos:
        Strategy("M-H", [("MEDIUM", 25), ("HARD", 30)])
        Strategy("H-S", [("HARD", 35), ("SOFT", 20)])
    """
    name: str
    stints: List[Tuple[str, int]]  # [(compound, n_vueltas), ...]


def build_lap_plan(strategy: Strategy, total_laps: int) -> pd.DataFrame:
    """
    Expande una Strategy a un plan por vuelta: compuesto, stint, TyreLife, FreshTyre.

    Devuelve un DataFrame con columnas:
        ["LapNumber", "Compound", "Stint", "TyreLife", "FreshTyre"]
    """
    laps = []
    current_lap = 1
    stint_idx = 1

    for compound, n_laps in strategy.stints:
        for i in range(n_laps):
            if current_lap > total_laps:
                break
            tyre_life = i + 1
            fresh = (i == 0)
            laps.append(
                {
                    "LapNumber": current_lap,
                    "Compound": compound,
                    "Stint": stint_idx,
                    "TyreLife": tyre_life,
                    "FreshTyre": int(fresh),
                }
            )
            current_lap += 1
        stint_idx += 1

    if current_lap <= total_laps:
        raise ValueError(
            f"La estrategia '{strategy.name}' no cubre todas las vueltas: "
            f"definió {current_lap - 1} vueltas para una carrera de {total_laps}"
        )

    return pd.DataFrame(laps)


def simulate_strategy_for_driver(
    df_race_base: pd.DataFrame,
    lap_time_model,
    legal_features_num: List[str],
    legal_features_cat: List[str],
    strategy: Strategy,
    pit_loss_s: float = 20.0,
) -> Dict[str, Any]:
    """
    Simula una carrera completa para Colapinto bajo una estrategia de neumáticos.

    - Toma las vueltas reales de carrera (df_race_base).
    - Reemplaza Compound / Stint / TyreLife / FreshTyre según la Strategy.
    - Recalcula features con add_basic_features.
    - Usa lap_time_model.predict(...) para LapTime_s.
    - Suma tiempos y agrega un pit_loss_s en cada cambio de compuesto
      (primera vuelta de cada stint, excepto el primero).

    Devuelve un dict con:
        - strategy_name
        - total_time_s
        - lap_times_s
        - df_sim (DataFrame con columnas LapTime_pred_s, LapTime_total_s, PitLoss_s)
    """
    # Ordenamos por vuelta
    df_race = df_race_base.sort_values("LapNumber").reset_index(drop=True).copy()
    total_laps = int(df_race["LapNumber"].max())

    # Plan de vueltas según la estrategia
    lap_plan = build_lap_plan(strategy, total_laps)

    # Merge por LapNumber
    df_sim = df_race.merge(lap_plan, on="LapNumber", how="inner", suffixes=("", "_plan"))

    # Sobrescribir columnas con las de la estrategia
    df_sim["Compound"] = df_sim["Compound_plan"]
    df_sim["Stint"] = df_sim["Stint_plan"]
    df_sim["TyreLife"] = df_sim["TyreLife_plan"]
    df_sim["FreshTyre"] = df_sim["FreshTyre_plan"]

    df_sim = df_sim.drop(
        columns=[c for c in df_sim.columns if c.endswith("_plan")],
        errors="ignore",
    )

    # Recalcular FE (stints, tyrelife_norm, etc.) con la nueva estrategia
    df_fe = add_basic_features(df_sim)
    
    # Si el modelo usa features avanzadas (como degradation_rate, S1_delta, etc.),
    # agregarlas. Detectamos esto viendo si alguna feature avanzada está en la lista.
    # Features avanzadas típicas: degradation_rate, S1_delta, track_evo, etc.
    advanced_features = {
        "degradation_rate", "S1_delta", "S2_delta", "S3_delta", 
        "S1_rel", "S2_rel", "S3_rel", "Lap_global", "track_evo",
        "speed_drop_fl", "speed_ratio_fl_st", "SpeedI1_norm_st", "SpeedI2_norm_st",
        "laps_since_pit", "compound_offset"
    }
    needs_advanced = bool(set(legal_features_num) & advanced_features)
    
    if needs_advanced:
        from src.preprocessing import add_advanced_features
        df_fe = add_advanced_features(df_fe)

    # Armar X con las features que usa el modelo
    feature_cols = [c for c in (legal_features_num + legal_features_cat) if c in df_fe.columns]
    X_sim = df_fe[feature_cols]

    # Predicción de laptimes "limpios"
    lap_times_pred = lap_time_model.predict(X_sim)

    # Penalización por pit y largada
    df_ordered = df_fe.sort_values("LapNumber").reset_index(drop=True)
    pit_penalty = np.zeros_like(lap_times_pred, dtype=float)

    for i, row in df_ordered.iterrows():
        # Penalización de largada (vuelta 1)
        if row["LapNumber"] == 1:
            # Calcular penalización basada en la diferencia real observada
            # En los datos reales, la vuelta 1 es ~15-20s más lenta
            pit_penalty[i] += 15.0
        # Penalización por pit stop (primera vuelta de cada stint > 1)
        elif row["Stint"] > 1 and row["TyreLife"] == 1:
            pit_penalty[i] += pit_loss_s

    lap_times_total = lap_times_pred + pit_penalty

    df_ordered["LapTime_pred_s"] = lap_times_pred
    df_ordered["PitLoss_s"] = pit_penalty
    df_ordered["LapTime_total_s"] = lap_times_total

    total_time_s = float(lap_times_total.sum())

    return {
        "strategy_name": strategy.name,
        "total_time_s": total_time_s,
        "lap_times_s": lap_times_total,
        "df_sim": df_ordered,
    }


def simulate_strategies(
    df_race_base: pd.DataFrame,
    lap_time_model,
    legal_features_num: List[str],
    legal_features_cat: List[str],
    strategies: List[Strategy],
    pit_loss_s: float = 20.0,
) -> pd.DataFrame:
    """
    Simula varias estrategias y devuelve una tabla tipo 'clasificación'.

    Devuelve un DataFrame con columnas:
        - strategy_name
        - total_time_s
        - total_time_str (mm:ss.xxx)
        - position (1 = estrategia más rápida)
    """
    results = []
    for strat in strategies:
        res = simulate_strategy_for_driver(
            df_race_base=df_race_base,
            lap_time_model=lap_time_model,
            legal_features_num=legal_features_num,
            legal_features_cat=legal_features_cat,
            strategy=strat,
            pit_loss_s=pit_loss_s,
        )
        results.append(res)

    standings = pd.DataFrame(
        {
            "strategy_name": [r["strategy_name"] for r in results],
            "total_time_s": [r["total_time_s"] for r in results],
        }
    ).sort_values("total_time_s", ascending=True).reset_index(drop=True)

    standings["position"] = standings.index + 1

    def format_time(t):
        minutes = int(t // 60)
        seconds = t - 60 * minutes
        return f"{minutes:02d}:{seconds:06.3f}"

    standings["total_time_str"] = standings["total_time_s"].apply(format_time)

    return standings


# =============================================================================
# FUNCIONES PARA EXTRACCIÓN DE ESTRATEGIA REAL
# =============================================================================

def extract_real_strategy_from_data(df_race):
    """
    Extrae una Strategy a partir de un DataFrame de carrera.
    
    Analiza los stints para determinar el patrón de neumáticos usado.
    Usa la columna "Stint" si está disponible, o detecta cambios de compuesto.
    
    Parameters:
    -----------
    df_race : pd.DataFrame
        DataFrame con datos de carrera que incluye columnas "Compound", "LapNumber" y opcionalmente "Stint"
        
    Returns:
    --------
    Strategy : Objeto Strategy con el nombre y los stints extraídos
    """
    if df_race.empty:
        return Strategy("EMPTY", [])
    
    df_sorted = df_race.sort_values("LapNumber").reset_index(drop=True)
    
    # Si existe la columna Stint, usarla directamente
    if "Stint" in df_sorted.columns:
        stints = []
        strategy_letters = []
        
        for stint_num in sorted(df_sorted["Stint"].unique()):
            stint_data = df_sorted[df_sorted["Stint"] == stint_num]
            compound = stint_data.iloc[0]["Compound"]
            n_laps = len(stint_data)
            
            stints.append((compound, n_laps))
            strategy_letters.append(compound[0])  # Primera letra del compuesto
    else:
        # Fallback: Detectar cambios de compuesto
        df_sorted["compound_change"] = (df_sorted["Compound"] != df_sorted["Compound"].shift(1))
        stint_starts = df_sorted[df_sorted["compound_change"]].copy()
        
        stints = []
        strategy_letters = []
        
        for i, start_row in stint_starts.iterrows():
            compound = start_row["Compound"]
            start_lap = start_row["LapNumber"]
            
            # Encontrar el final del stint
            if i < len(stint_starts) - 1:
                end_lap = stint_starts.iloc[stint_starts.index.get_loc(i) + 1]["LapNumber"] - 1
            else:
                end_lap = df_sorted["LapNumber"].max()
            
            n_laps = int(end_lap - start_lap + 1)
            stints.append((compound, n_laps))
            strategy_letters.append(compound[0])  # Primera letra del compuesto
    
    # Crear nombre de estrategia (ej: "M-H" o "S-M-H")
    strategy_name = "-".join(strategy_letters)
    
    return Strategy(strategy_name, stints)
