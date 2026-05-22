import pandas as pd
import numpy as np


def detectar_cambios_significativos(df, empresa_col, fecha_col, precio_col, umbral=2.5):
    df_resultado = df.copy()
    df_resultado[fecha_col] = pd.to_datetime(df_resultado[fecha_col])
    df_resultado = df_resultado.sort_values(by=[empresa_col, fecha_col])

    df_resultado["media_movil_30"] = (
        df_resultado.groupby(empresa_col)[precio_col]
        .transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    )

    df_resultado["desviacion_pct"] = (
        abs(df_resultado[precio_col] - df_resultado["media_movil_30"])
        / df_resultado["media_movil_30"]
    )

    df_resultado["cambio_significativo"] = df_resultado["desviacion_pct"] > umbral

    return df_resultado
