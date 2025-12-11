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


# =============================================================================
# FUNCIONES PARA PROCESAMIENTO DE DATOS DE CARRERA (FastF1)
# =============================================================================

def time_to_seconds(time_val):
    """
    Convierte un valor de tiempo de FastF1 (timedelta o string) a segundos.
    
    Parameters:
    -----------
    time_val : timedelta, str, or None
        Valor de tiempo a convertir
        
    Returns:
    --------
    float or None : Tiempo en segundos, o None si no es válido
    """
    if pd.isna(time_val) or time_val == "":
        return None
    try:
        time_td = pd.to_timedelta(time_val)
        return time_td.total_seconds()
    except:
        return None


def create_leaderboard_from_session(session):
    """
    Crea un DataFrame con el leaderboard de una sesión de FastF1.
    Calcula tiempos absolutos desde gaps.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        Sesión de FastF1 cargada
        
    Returns:
    --------
    pd.DataFrame : Leaderboard con columnas Position, Driver, Team, Time, Status, 
                   Gap_s, Time_s
    """
    race_results = session.results
    
    # Crear DataFrame con clasificación real
    leaderboard = pd.DataFrame({
        "Position": race_results["Position"],
        "Driver": race_results["Abbreviation"],
        "Team": race_results["TeamName"],
        "Time": race_results["Time"],
        "Status": race_results["Status"],
    })
    
    # Convertir tiempo a segundos
    leaderboard["Gap_s"] = leaderboard["Time"].apply(time_to_seconds)
    
    # El ganador (P1) tiene el tiempo absoluto, los demás tienen gap
    winner = leaderboard[leaderboard["Position"] == 1].iloc[0]
    winner_time_s = winner["Gap_s"]  # Este es el tiempo absoluto del ganador
    
    # Calcular tiempos absolutos para todos
    leaderboard["Time_s"] = leaderboard.apply(
        lambda row: winner_time_s if row["Position"] == 1 else (
            winner_time_s + row["Gap_s"] if row["Gap_s"] else None
        ),
        axis=1
    )
    
    return leaderboard


def add_simulated_driver_to_leaderboard(leaderboard, session, driver_abbr, 
                                        simulated_time_s, simulated_laps,
                                        suffix="*"):
    """
    Agrega un piloto simulado al leaderboard y recalcula posiciones.
    
    Parameters:
    -----------
    leaderboard : pd.DataFrame
        Leaderboard original
    session : fastf1.core.Session
        Sesión de FastF1
    driver_abbr : str
        Abreviación del piloto (ej: "COL")
    simulated_time_s : float
        Tiempo simulado en segundos
    simulated_laps : int
        Número de vueltas completadas
    suffix : str
        Sufijo para el piloto simulado (default: "*")
        
    Returns:
    --------
    dict : Diccionario con leaderboard_comparison, position_gain, time_saved, y display
    """
    leaderboard_extended = leaderboard.copy()
    
    # Obtener datos del piloto real
    driver_real = leaderboard[leaderboard["Driver"] == driver_abbr].iloc[0]
    
    # Agregar fila de piloto simulado
    driver_simulated = {
        "Position": None,
        "Driver": f"{driver_abbr}{suffix}",
        "Team": driver_real["Team"],
        "Time": None,
        "Time_s": simulated_time_s,
        "Status": driver_real["Status"],
        "Laps_completed": simulated_laps
    }
    
    # Convertir a lista
    leaderboard_list = leaderboard_extended.to_dict('records')
    
    # Calcular vueltas completadas para cada piloto
    total_race_laps = session.total_laps
    
    for record in leaderboard_list:
        if record["Status"] == "Finished":
            record["Laps_completed"] = total_race_laps
        elif record["Status"] == "Lapped":
            # Obtener las vueltas del piloto desde FastF1
            driver_laps = session.laps.pick_driver(record["Driver"])
            if len(driver_laps) > 0:
                record["Laps_completed"] = int(driver_laps["LapNumber"].max())
            else:
                record["Laps_completed"] = 0
        else:  # DNF, Retired, etc.
            driver_laps = session.laps.pick_driver(record["Driver"])
            if len(driver_laps) > 0:
                record["Laps_completed"] = int(driver_laps["LapNumber"].max())
            else:
                record["Laps_completed"] = 0
    
    leaderboard_list.append(driver_simulated)
    
    # Crear DataFrame
    leaderboard_comparison = pd.DataFrame(leaderboard_list)
    
    # Ordenar: primero por vueltas completadas (desc), luego por tiempo (asc)
    leaderboard_comparison = leaderboard_comparison.sort_values(
        by=["Laps_completed", "Time_s"],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    # Asignar nuevas posiciones
    leaderboard_comparison["New_Position"] = range(1, len(leaderboard_comparison) + 1)
    
    # Formatear tiempos para display
    winner_time_final = leaderboard_comparison.iloc[0]["Time_s"]
    
    for idx, row in leaderboard_comparison.iterrows():
        if pd.isna(row["Time_s"]) or row["Time_s"] is None:
            leaderboard_comparison.at[idx, "Time_Display"] = "DNF"
        elif row["Laps_completed"] < total_race_laps:
            # Mostrar cuántas vueltas perdió
            laps_down = total_race_laps - row["Laps_completed"]
            if laps_down == 1:
                leaderboard_comparison.at[idx, "Time_Display"] = "1 LAP"
            else:
                leaderboard_comparison.at[idx, "Time_Display"] = f"{int(laps_down)} LAPS"
        elif idx == 0:
            # El ganador muestra su tiempo total
            time_val = row["Time_s"]
            minutes = int(time_val // 60)
            seconds = time_val % 60
            leaderboard_comparison.at[idx, "Time_Display"] = f"{minutes:02d}:{seconds:06.3f}"
        else:
            # Los que completaron todas las vueltas muestran gap con el ganador
            gap = row["Time_s"] - winner_time_final
            minutes = int(gap // 60)
            seconds = gap % 60
            leaderboard_comparison.at[idx, "Time_Display"] = f"+{minutes:02d}:{seconds:06.3f}"
    
    # Preparar display
    display_cols = ["New_Position", "Driver", "Team", "Time_Display", "Status", "Laps_completed"]
    leaderboard_display = leaderboard_comparison[display_cols].copy()
    leaderboard_display.columns = ["Pos", "Driver", "Team", "Time/Gap", "Status", "Laps"]
    
    # Calcular métricas
    driver_real_pos = leaderboard_comparison[leaderboard_comparison["Driver"] == driver_abbr]["New_Position"].values[0]
    driver_sim_pos = leaderboard_comparison[leaderboard_comparison["Driver"] == f"{driver_abbr}{suffix}"]["New_Position"].values[0]
    position_gain = driver_real_pos - driver_sim_pos
    
    # Calcular tiempo ahorrado
    driver_real_time = leaderboard_comparison[leaderboard_comparison["Driver"] == driver_abbr]["Time_s"].values[0]
    time_saved = driver_real_time - simulated_time_s
    
    return {
        "leaderboard_comparison": leaderboard_comparison,
        "leaderboard_display": leaderboard_display,
        "position_gain": position_gain,
        "time_saved": time_saved,
        "driver_real_pos": driver_real_pos,
        "driver_sim_pos": driver_sim_pos,
    }


def calculate_real_race_times(session):
    """
    Calcula los tiempos totales reales de carrera sumando los tiempos de cada vuelta.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        Sesión de FastF1 cargada
        
    Returns:
    --------
    pd.DataFrame : DataFrame con columnas Driver, Total_Time_s, Laps_Completed, Status
    """
    all_drivers = session.results["Abbreviation"].values
    
    real_times = {}
    
    for driver_abbr in all_drivers:
        driver_laps = session.laps.pick_driver(driver_abbr)
        
        if len(driver_laps) == 0:
            real_times[driver_abbr] = {
                "total_time_s": None,
                "laps_completed": 0,
                "status": "DNF"
            }
            continue
        
        # Sumar todos los tiempos de vuelta (convertir a segundos)
        lap_times_s = driver_laps["LapTime"].dt.total_seconds()
        total_time_s = lap_times_s.sum()
        laps_completed = int(driver_laps["LapNumber"].max())
        
        # Obtener status
        driver_result = session.results[session.results["Abbreviation"] == driver_abbr]
        status = driver_result["Status"].values[0] if len(driver_result) > 0 else "Unknown"
        
        real_times[driver_abbr] = {
            "total_time_s": total_time_s,
            "laps_completed": laps_completed,
            "status": status
        }
    
    # Crear DataFrame con tiempos reales calculados
    real_times_df = pd.DataFrame([
        {
            "Driver": driver,
            "Total_Time_s": data["total_time_s"],
            "Laps_Completed": data["laps_completed"],
            "Status": data["status"]
        }
        for driver, data in real_times.items()
    ])
    
    return real_times_df

