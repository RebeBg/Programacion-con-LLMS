import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random

# ------------------------------
# Función principal de clasificación
# ------------------------------
def clasificar_urgencia(df, target_col):
    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelo
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Errores
    errores = np.where(y_pred != y_test.values)[0]
    
    # Accuracy
    accuracy = (y_pred == y_test.values).mean()
    
    # Retorno
    return {
        "accuracy": accuracy,
        "n_errores": len(errores),
        "indices_error": errores
    }

# ------------------------------
# Función generadora de casos de uso para preparar_datos
# ------------------------------
def generar_caso_de_uso_preparar_datos():
    # Input aleatorio
    input_data = {
        "peso": [random.uniform(1, 50) for _ in range(10)],
        "edad": [random.randint(1, 15) for _ in range(10)],
        "temperatura": [random.uniform(36, 40) for _ in range(10)],
        "frecuencia_cardiaca": [random.randint(60, 180) for _ in range(10)],
        "urgente": [random.randint(0, 1) for _ in range(10)]
    }

    # Output simulado de preparar_datos: convertir a DataFrame
    output_data = pd.DataFrame(input_data)

    return input_data, output_data

# ------------------------------
# Ejemplo de uso completo
# ------------------------------
if __name__ == "__main__":
    # Generar un caso de uso aleatorio
    input_dict, df = generar_caso_de_uso_preparar_datos()
    
    print("Input generado para preparar_datos:")
    print(input_dict)
    
    print("\nOutput simulado de preparar_datos (DataFrame):")
    print(df)
    
    # Probar clasificar_urgencia con este DataFrame
    resultado = clasificar_urgencia(df, target_col="urgente")
    
    print("\nResultado de clasificar_urgencia:")
    print(resultado)
