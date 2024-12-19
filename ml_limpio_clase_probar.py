from tensorflow.keras.models import load_model
import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def read_data_from_csv(file_path):
    """
    Leer y procesar datos CSV incluyendo acelerómetro, giroscopio y etiquetas.
    """
    df = pd.read_csv(file_path)
        
    if df.shape[1] == 8:
        df.columns = ['timestamp_acc', 'acc_x', 'acc_y', 'acc_z', 'timestamp_gyro', 'gyro_x', 'gyro_y', 'gyro_z']
    else:
        raise ValueError(f"El archivo {file_path} no tiene el número esperado de columnas.")

    # Suavizado con Savitzky-Golay
    window_size = 101
    df['acc_x_smooth'] = savgol_filter(df['acc_x'], window_length=window_size, polyorder=2)
    df['acc_y_smooth'] = savgol_filter(df['acc_y'], window_length=window_size, polyorder=2)
    df['acc_z_smooth'] = savgol_filter(df['acc_z'], window_length=window_size, polyorder=2)
    df['gyro_x_smooth'] = savgol_filter(df['gyro_x'], window_length=window_size, polyorder=2)
    df['gyro_y_smooth'] = savgol_filter(df['gyro_y'], window_length=window_size, polyorder=2)
    df['gyro_z_smooth'] = savgol_filter(df['gyro_z'], window_length=window_size, polyorder=2)

    return df


def create_windows(data, window_size, step_size):
    """
    Divide datos en ventanas deslizantes.
    """
    features = ['acc_x_smooth', 'acc_y_smooth', 'acc_z_smooth',
                'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth']
    data_array = data[features].values
    n_samples = len(data_array)
    
    # Crear ventanas
    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        window = data_array[start:start + window_size]
        windows.append(window)
    
    return np.array(windows)


# Cargar el modelo
modelo_cargado = load_model("C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/modelo_lstm_multisalida.h5")
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Mezcla"

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        nuevas_ventanas = create_windows(df, window_size=100, step_size=50)

        # Realizar predicciones con el modelo cargado
        predicciones = modelo_cargado.predict(nuevas_ventanas, verbose=0)
        
        # Separar las predicciones
        pred_clasificacion = predicciones[0].flatten()  # Predicciones de flexión/extensión
        pred_repeticiones = predicciones[1].flatten()   # Predicciones de número de repeticiones
        
        # Determinar el resultado global de clasificación
        promedio_clasificacion = np.mean(pred_clasificacion)
        resultado_clasificacion = "Extensión" if promedio_clasificacion >= 0.5 else "Flexión"
        
        # Calcular el promedio de repeticiones predichas
        promedio_repeticiones = np.mean(pred_repeticiones)

        # Imprimir resultados
        print(f"Archivo: {filename}")
        print(f"Promedio de clasificación: {promedio_clasificacion:.4f}")
        print(f"Predicción global (clasificación): {resultado_clasificacion}")
        print(f"Promedio de repeticiones predichas: {promedio_repeticiones:.2f}")
        print("-" * 30)
