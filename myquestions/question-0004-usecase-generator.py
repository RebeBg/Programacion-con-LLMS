import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

# -------------------------------------------------------------
# Función para marcar anomalías
# -------------------------------------------------------------
def marcar_anomalias(df, umbral):
    """
    Escala las columnas numéricas y marca como anomalía las filas cuya
    distancia euclidiana al origen supere el umbral.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas numéricas.
        umbral (float): Distancia mínima para considerar una fila anómala.
        
    Returns:
        pd.DataFrame: DataFrame con columna 'anomalia' añadida.
    """
    # Seleccionar columnas numéricas
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols_num])
    
    # Calcular distancia euclidiana al origen
    distancias = np.linalg.norm(X_scaled, axis=1)
    
    # Añadir columna 'anomalia'
    df_result = df.copy()
    df_result['anomalia'] = distancias > umbral
    
    return df_result

# -------------------------------------------------------------
# Generador de caso de uso aleatorio
# -------------------------------------------------------------
def generar_caso_de_uso_marcar_anomalias():
    """
    Genera un caso de prueba aleatorio (input/output) para marcar_anomalias.
    """
    # 1. Configuración aleatoria
    n_rows = random.randint(10, 25)
    
    # 2. Generar datos aleatorios de transacciones
    df = pd.DataFrame({
        'monto': np.random.uniform(10, 5000, n_rows),
        'hora_del_dia': np.random.randint(0, 24, n_rows),
        'num_intentos': np.random.randint(1, 5, n_rows)
    })
    
    # Umbral aleatorio razonable
    umbral = random.uniform(1.5, 3.0)
    
    # Input
    input_data = {'df': df.copy(), 'umbral': umbral}
    
    # Calcular output esperado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    distancias = np.linalg.norm(X_scaled, axis=1)
    
    df_output = df.copy()
    df_output['anomalia'] = distancias > umbral
    
    return input_data, df_output

# -------------------------------------------------------------
# Ejemplo de ejecución
# -------------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_marcar_anomalias()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Umbral: {entrada['umbral']:.2f}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (primeras 5 filas) ===")
    print(salida_esperada.head())
    
    # Ejemplo de llamada a la función real
    resultado = marcar_anomalias(**entrada)
    print("\n=== Resultado marcar_anomalias (primeras 5 filas) ===")
    print(resultado.head())
