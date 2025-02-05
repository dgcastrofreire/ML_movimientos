import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
from tensorflow.keras.layers import GRU


# Configuración de simulación
sampling_rate = 100  # Frecuencia de muestreo en Hz
time_step = 1 / sampling_rate  # Intervalo entre muestras
window_size = 50  # Ventana deslizante (número de timesteps por ventana)
features = 11  # Número de características (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)

def moving_average(data, window_size):
    """
    Calcula el promedio móvil para suavizar datos.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def read_data_from_csv(file_path):
    """
    Leer y procesar datos desde un archivo CSV.
    """
    print(f"Leyendo archivo {file_path}")
    df = pd.read_csv(file_path)
    if df.shape[1] == 8:
        df.columns = ['timestamp_acc', 'acc_x', 'acc_y', 'acc_z', 'timestamp_gyro', 'gyro_x', 'gyro_y', 'gyro_z']
    else:
        raise ValueError(f"El archivo {file_path} no tiene el número esperado de columnas.")
    
    # Suavizado con Savitzky-Golay
    window = 50
    df['acc_x_smooth'] = moving_average(df['acc_x'], window)
    df['acc_y_smooth'] = moving_average(df['acc_y'], window)
    df['acc_z_smooth'] = moving_average(df['acc_z'], window)
    df['gyro_x_smooth'] = moving_average(df['gyro_x'],window)
    df['gyro_y_smooth'] = moving_average(df['gyro_y'], window)
    df['gyro_z_smooth'] = moving_average(df['gyro_z'], window)
    
    # Magnitudes (rápidas de calcular)
    df['acc_magnitude'] = np.sqrt(df['acc_x_smooth']**2 + df['acc_y_smooth']**2 + df['acc_z_smooth']**2)
    df['gyro_magnitude'] = np.sqrt(df['gyro_x_smooth']**2 + df['gyro_y_smooth']**2 + df['gyro_z_smooth']**2)
    
    # Opcional: Velocidades (más rápidas que las aceleraciones de segundo orden)
    df['gyro_x_velocity'] = df['gyro_x_smooth'].diff() / time_step
    df['gyro_y_velocity'] = df['gyro_y_smooth'].diff() / time_step
    df['gyro_z_velocity'] = df['gyro_z_smooth'].diff() / time_step
    
    # Llenar valores NaN generados por diferencias
    df.fillna(0, inplace=True)
    
    return df

# Diccionario para almacenar estadísticas por clase
class_stats = {
    "Idle": {"acc_magnitude": [], "gyro_magnitude": []},
    "Flexion": {"acc_magnitude": [], "gyro_magnitude": []},
    "Extension": {"acc_magnitude": [], "gyro_magnitude": []},
}

# Función para calcular magnitudes
def calculate_magnitudes(df):
    acc_magnitude = np.sqrt(df['acc_x_smooth']**2 + df['acc_y_smooth']**2 + df['acc_z_smooth']**2)
    gyro_magnitude = np.sqrt(df['gyro_x_smooth']**2 + df['gyro_y_smooth']**2 + df['gyro_z_smooth']**2)
    return acc_magnitude, gyro_magnitude

# Leer datos y calcular estadísticas
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Mezcla"
for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        
        # Etiqueta según el archivo
        if "flexion" in filename.lower():
            label = "Flexion"
        elif "extension" in filename.lower():
            label = "Extension"
        elif "idle" in filename.lower():
            label = "Idle"
        else:
            continue  # Ignorar archivos sin etiqueta válida
        
        # Calcular magnitudes
        acc_magnitude, gyro_magnitude = calculate_magnitudes(df)
        
        # Guardar estadísticas
        class_stats[label]["acc_magnitude"].extend(acc_magnitude)
        class_stats[label]["gyro_magnitude"].extend(gyro_magnitude)

# Convertir listas en DataFrames para análisis
for label, stats in class_stats.items():
    class_stats[label]["acc_magnitude"] = pd.Series(stats["acc_magnitude"])
    class_stats[label]["gyro_magnitude"] = pd.Series(stats["gyro_magnitude"])

# Calcular estadísticas descriptivas
detailed_stats = {}
for label, stats in class_stats.items():
    detailed_stats[label] = {
        "acc_magnitude": stats["acc_magnitude"].describe(percentiles=[0.1, 0.25, 0.75, 0.9]),
        "gyro_magnitude": stats["gyro_magnitude"].describe(percentiles=[0.1, 0.25, 0.75, 0.9]),
    }

# Imprimir estadísticas detalladas
for label, stats in detailed_stats.items():
    print(f"\n=== {label} ===")
    print("Aceleración Magnitud:")
    print(stats["acc_magnitude"])
    print("\nGiroscopio Magnitud:")
    print(stats["gyro_magnitude"])