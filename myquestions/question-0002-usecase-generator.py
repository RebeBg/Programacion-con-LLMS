import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import random

# -------------------------------------------------------------
# GENERADOR DE CASOS DE USO 
# -------------------------------------------------------------
def generar_caso_de_uso_predecir_dificultad():

    # 1. Configuración aleatoria
    n_rows = random.randint(5, 15)
    n_features = random.randint(2, 5)

    # 2. Generar datos aleatorios
    data = np.random.randn(n_rows, n_features)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_cols)

    # Introducir NaNs aleatorios (~10%)
    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df[mask] = np.nan

    # Target
    target_col = 'target_variable'
    df[target_col] = np.random.randint(0, 2, size=n_rows)

    # Input para el generador
    input_data = {'df': df.copy(), 'target_col': target_col}

    # Output esperado (imputación básica)
    X_expected = df.drop(columns=[target_col])
    y_expected = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_expected)

    output_data = (X_imputed, y_expected)

    return input_data, output_data


# -------------------------------------------------------------
# FUNCIÓN SOLUCIÓN (DESPUÉS DEL GENERADOR)
# -------------------------------------------------------------
def predecir_dificultad(df, target_col):
    """
    Entrena un DecisionTreeClassifier para predecir la dificultad.
    Imputa NaNs con la media y reporta accuracy.
    """

    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Imputar NaNs
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    # Modelo
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Predicción
    y_pred = clf.predict(X_test)

    # Métricas
    correctas = np.sum(y_pred == y_test)
    incorrectas = np.sum(y_pred != y_test)
    accuracy = correctas / len(y_test)

    return {
        'accuracy': float(accuracy),
        'correctas': int(correctas),
        'incorrectas': int(incorrectas)
    }


# -------------------------------------------------------------
# EJECUCIÓN DE PRUEBA
# -------------------------------------------------------------
if __name__ == "__main__":

    entrada, salida_esperada = generar_caso_de_uso_predecir_dificultad()

    print("=== INPUT ===")
    print("Target:", entrada['target_col'])
    print(entrada['df'].head())

    print("\n=== OUTPUT ESPERADO ===")
    X_res, y_res = salida_esperada
    print("Shape X:", X_res.shape)
    print("Shape y:", y_res.shape)

    print("\n=== RESULTADO MODELO ===")
    resultado = predecir_dificultad(entrada['df'], entrada['target_col'])
    print(resultado)
