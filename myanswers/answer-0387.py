import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error


def evaluar_modelo_poisson(df, columnas, columna_objetivo, test_size):
    X_data = df[columnas].to_numpy()
    y_data = df[columna_objetivo].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = PoissonRegressor()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    return mean_absolute_error(y_test, y_pred)
