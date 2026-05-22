import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def detect_feature_drift(df_train, df_test, threshold):
    """
    Detecta columnas numéricas cuya distribución ha cambiado significativamente
    entre df_train y df_test, usando la diferencia absoluta de medias como métrica.
    Retorna un diccionario {columna: magnitud_del_cambio} para las que superan el threshold.
    """
    num_cols = df_train.select_dtypes(include=[np.number]).columns
    result = {}
    for col in num_cols:
        if col in df_test.columns:
            diff = float(np.abs(df_test[col].mean() - df_train[col].mean()))
            if diff > threshold:
                result[col] = diff
    return result
