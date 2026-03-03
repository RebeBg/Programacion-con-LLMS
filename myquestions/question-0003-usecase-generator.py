import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import random

# -------------------------------------------------------------
# Función para evaluar fincas
# -------------------------------------------------------------
def evaluar_finca(df, target_col):
    """
    Entrena un modelo Ridge para predecir el precio de fincas.
    Calcula el error absoluto por muestra y métricas agregadas.
    
    Args:
        df (pd.DataFrame): DataFrame con características y target (precio).
        target_col (str): Nombre de la columna target.
        
    Returns:
        dict: {'r2': float, 'mae_promedio': float, 'mae_maximo': float, 'mae_minimo': float}
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    
    # 2. Dividir en train/test 75/25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    
    # 3. Entrenar Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # 4. Predecir en test
    y_pred = model.predict(X_test)
    
    # 5. Calcular errores absolutos por muestra
    errores = np.abs(y_test - y_pred)
    
    # 6. Métricas agregadas
    r2 = model.score(X_test, y_test)
    mae_promedio = errores.mean()
    mae_maximo = errores.max()
    mae_minimo = errores.min()
    
    return {
        'r2': r2,
        'mae_promedio': mae_promedio,
        'mae_maximo': mae_maximo,
        'mae_minimo': mae_minimo
    }

# -------------------------------------------------------------
# Generador de caso de uso aleatorio
# -------------------------------------------------------------
def generar_caso_de_uso_evaluar_finca():
    """
    Genera un caso de prueba aleatorio (input/output) para evaluar_finca.
    """
    # 1. Configuración aleatoria
    n_rows = random.randint(10, 30)  # Número de fincas
    n_features = 3  # hectareas, distancia_ciudad_km, num_construcciones
    
    # 2. Generar datos aleatorios
    df = pd.DataFrame({
        'hectareas': np.random.uniform(1, 50, n_rows),
        'distancia_ciudad_km': np.random.uniform(1, 100, n_rows),
        'num_construcciones': np.random.randint(0, 10, n_rows)
    })
    
    # Precio aleatorio con cierta relación con las variables
    df['precio'] = (
        df['hectareas'] * np.random.uniform(5000, 10000) -
        df['distancia_ciudad_km'] * np.random.uniform(50, 200) +
        df['num_construcciones'] * np.random.uniform(10000, 50000) +
        np.random.normal(0, 20000, n_rows)  # Ruido
    )
    
    target_col = 'precio'
    input_data = {'df': df.copy(), 'target_col': target_col}
    
    # Calcular output esperado (simulando la función)
    # Dividir train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_col]), df[target_col].to_numpy(),
        test_size=0.25, random_state=7
    )
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errores = np.abs(y_test - y_pred)
    
    output_data = {
        'r2': model.score(X_test, y_test),
        'mae_promedio': errores.mean(),
        'mae_maximo': errores.max(),
        'mae_minimo': errores.min()
    }
    
    return input_data, output_data

# -------------------------------------------------------------
# Ejemplo de ejecución
# -------------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_evaluar_finca()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Target Column: {entrada['target_col']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO ===")
    for k, v in salida_esperada.items():
        print(f"{k}: {v:.2f}")
    
    # Ejemplo de llamada a la función real
    resultado = evaluar_finca(**entrada)
    print("\n=== Resultado evaluar_finca ===")
    for k, v in resultado.items():
        print(f"{k}: {v:.2f}")
