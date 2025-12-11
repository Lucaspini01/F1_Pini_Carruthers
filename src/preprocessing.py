import numpy as np
import pandas as pd

def clean_session(col_laps, laptime_max_s=None, delta_from_best=None, verbose=True):
    """
    Limpia una sesión de prácticas filtrando vueltas demasiado lentas.
    
    Parámetros
    ----------
    col_laps : DataFrame de una práctica (solo Colapinto)
    laptime_max_s : float o None
        Umbral absoluto en segundos. Se mantienen solo vueltas con LapTime_s <= laptime_max_s.
    delta_from_best : float o None
        Si se pasa, se ignora laptime_max_s y se define:
        cut = best_lap + delta_from_best
    """
    df = col_laps.copy()
    
    if "LapTime_s" not in df.columns:
        df["LapTime_s"] = df["LapTime"].dt.total_seconds()
    
    best = df["LapTime_s"].min()
    
    if delta_from_best is not None:
        cut = best + delta_from_best
    elif laptime_max_s is not None:
        cut = laptime_max_s
    else:
        raise ValueError("Tenés que pasar laptime_max_s o delta_from_best")
    
    before = len(df)
    df = df[df["LapTime_s"] <= cut].copy()
    after = len(df)
    
    if verbose:
        print(f"Best lap: {best:.3f} s, corte: {cut:.3f} s")
        print(f"Vueltas antes: {before}, después del filtrado: {after} (se eliminaron {before - after})")
    
    return df, cut


# FEATURE ENGINEERING

# src/preprocessing.py

import pandas as pd
import numpy as np

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering v1 para el problema de LapTime.
    No usa LapTime ni info futura. Devuelve un nuevo DataFrame.
    """
    df = df.copy()

    # 1) LapNumber normalizado por sesión (fase de la sesión)
    if {"Session", "LapNumber"}.issubset(df.columns):
        max_laps_session = df.groupby("Session")["LapNumber"].transform("max")
        df["lap_norm_session"] = df["LapNumber"] / max_laps_session

    # 2) Progresión dentro del stint
    if {"Session", "Stint", "LapNumber"}.issubset(df.columns):
        # largo del stint
        stint_sizes = df.groupby(["Session", "Stint"])["LapNumber"].transform("count")
        df["stint_len"] = stint_sizes

        # índice de vuelta dentro del stint (1..stint_len)
        df["stint_lap_index"] = (
            df.groupby(["Session", "Stint"]).cumcount() + 1
        )

        # normalizado 0..1
        df["stint_lap_norm"] = df["stint_lap_index"] / df["stint_len"]

    # 3) TyreLife normalizado dentro del stint
    if {"Session", "Stint", "TyreLife"}.issubset(df.columns):
        max_tyre_stint = df.groupby(["Session", "Stint"])["TyreLife"].transform("max")
        # evitar división por cero o NaN
        df["tyrelife_norm_stint"] = df["TyreLife"] / max_tyre_stint.replace(0, np.nan)

    # 4) Indicador de carrera vs práctica
    if "Session" in df.columns:
        df["is_race"] = (df["Session"] == "RACE").astype(int)

    # 5) Orden de compuestos (más blando -> más duro)
    if "Compound" in df.columns:
        compound_map = {
            "SOFT": 0,
            "MEDIUM": 1,
            "HARD": 2,
            "INTERMEDIATE": 3,
            "WET": 4,
        }
        df["compound_order"] = df["Compound"].map(compound_map).astype("float")

    return df

# FE 2.0
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering v2 para el problema de LapTime.
    NO modifica nada de lo anterior: solo agrega columnas nuevas.
    No usa información futura (solo pasado o info "online" de la vuelta).
    """
    df = df.copy()

    # -------------------------------------------------
    # 1) Lap global dentro de cada sesión
    # -------------------------------------------------
    # Idea: en carrera el fuel load hace que las primeras vueltas sean más lentas.
    # Esto le da al modelo una pista de "en qué fase" de la sesión estamos.
    if "Session" in df.columns:
        df = df.sort_values(["Session", "LapNumber"])
        df["Lap_global"] = df.groupby("Session").cumcount() + 1
        df = df.sort_index()

    # -------------------------------------------------
    # 2) Evolución de la pista / sesión (track evolution proxy)
    # -------------------------------------------------
    if {"Session", "LapNumber"}.issubset(df.columns):
        max_laps_session = df.groupby("Session")["LapNumber"].transform("max")
        # similar a lap_norm_session, pero lo dejamos separado por claridad
        df["track_evo"] = df["LapNumber"] / max_laps_session

    # -------------------------------------------------
    # 3) Deltas y ratios de sectores vs mejor de la sesión
    #    (con conversión robusta a segundos)
    # -------------------------------------------------
    from pandas.api.types import is_numeric_dtype

    for i in (1, 2, 3):
        col_time = f"Sector{i}Time"
        if col_time in df.columns:
            # Tomamos la columna original
            time = df[col_time]

            # Si no es numérica, la pasamos a timedelta -> segundos
            if not is_numeric_dtype(time):
                time = pd.to_timedelta(time, errors="coerce").dt.total_seconds()

            # Calcular el mejor tiempo de este sector en cada sesión
            if "Session" in df.columns:
                best = df.groupby("Session")[col_time].transform(
                    lambda x: pd.to_timedelta(x, errors="coerce").dt.total_seconds().min()
                )
            else:
                best = pd.to_timedelta(df[col_time], errors="coerce").dt.total_seconds().min()

            # Delta (cuánto más lento que el mejor sector de la sesión)
            df[f"S{i}_delta"] = time - best

            # Ratio (sector actual / mejor sector)
            denom = best.replace(0, np.nan)
            df[f"S{i}_rel"] = time / denom

    # -------------------------------------------------
    # 4) Features de velocidades (pace / aero / tracción)
    # -------------------------------------------------
    # Drop y ratio entre SpeedTrap y Speed en línea de meta
    if {"SpeedST", "SpeedFL"}.issubset(df.columns):
        df["speed_drop_fl"] = df["SpeedST"] - df["SpeedFL"]

        denom = df["SpeedST"].replace(0, np.nan)
        df["speed_ratio_fl_st"] = df["SpeedFL"] / denom

    # Velocidades intermedias normalizadas por SpeedTrap
    if "SpeedST" in df.columns:
        denom = df["SpeedST"].replace(0, np.nan)
        for col in ["SpeedI1", "SpeedI2"]:
            if col in df.columns:
                df[f"{col}_norm_st"] = df[col] / denom

    # -------------------------------------------------
    # 5) Dinámica de stint / neumáticos
    # -------------------------------------------------
    # a) Vueltas desde que saliste del pit (por stint)
    if {"Session", "Stint", "LapNumber"}.issubset(df.columns):
        first_lap_stint = df.groupby(["Session", "Stint"])["LapNumber"].transform("min")
        df["laps_since_pit"] = df["LapNumber"] - first_lap_stint

    # b) Tasa de degradación instantánea (diferencia vs vuelta anterior en el mismo stint)
    if {"Session", "Stint", "LapNumber", "LapTime_s"}.issubset(df.columns):
        # ordenamos por Session/Stint/LapNumber para que diff tenga sentido
        df_sorted = df.sort_values(["Session", "Stint", "LapNumber"])
        degr = df_sorted.groupby(
            ["Session", "Stint"]
        )["LapTime_s"].diff()
        # reindex al orden original
        df["degradation_rate"] = degr.reindex(df_sorted.index).reindex(df.index)
    else:
        # Si no hay LapTime_s (por ejemplo, durante simulación), crear columna con NaN
        # El imputer del pipeline se encargará de rellenarla
        df["degradation_rate"] = np.nan

    # c) Offset teórico de compuesto (muy simple, solo como prior)
    if "Compound" in df.columns:
        compound_base = {
            "SOFT": 0.0,
            "MEDIUM": 1.0,
            "HARD": 2.0,
            "INTERMEDIATE": 3.0,
            "WET": 4.0,
        }
        df["compound_offset"] = (
            df["Compound"].map(compound_base).astype("float")
        )

    # -------------------------------------------------
    # 6) TrackStatus simplificado (banderas / condiciones raras)
    # -------------------------------------------------
    if "TrackStatus" in df.columns:
        # Pasamos a numérico por si viene como string; NaN => 0
        ts_num = pd.to_numeric(df["TrackStatus"], errors="coerce").fillna(0)

        # Flag simple: 1 si NO está en condiciones "green" estándar
        df["TS_non_green"] = (ts_num != 1).astype(int)

    return df

