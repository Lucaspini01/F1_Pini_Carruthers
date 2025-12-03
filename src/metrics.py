# src/metrics.py

import numpy as np
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
