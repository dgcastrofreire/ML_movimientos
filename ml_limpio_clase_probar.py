from tensorflow.keras.models import load_model
import os
import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.random import set_seed


def read_data_from_csv(file_path):
    """
    Leer y procesar datos CSV incluyendo acelerómetro, giroscopio y etiquetas (bien/mal).
    """
    #print(f"Leyendo archivo {file_path}")
    df = pd.read_csv(file_path)

    # Asignar nombres de columnas
    '''if df.shape[1] == 9:
        df.columns = ['index', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']
        df = df.drop(columns=['index'])'''
        
    if df.shape[1] == 8:
        df.columns = ['timestamp_acc', 'acc_x', 'acc_y', 'acc_z','timestamp_gyro', 'gyro_x', 'gyro_y', 'gyro_z']
    else:
        raise ValueError(f"El archivo {file_path} no tiene el número esperado de columnas.")

    # Suavizado con Savitzky-Golay y normalización de todos los componentes
    window_size = 101
    df['acc_x_smooth'] = savgol_filter(df['acc_x'], window_length=window_size, polyorder=2)
    df['acc_y_smooth'] = savgol_filter(df['acc_y'], window_length=window_size, polyorder=2)
    df['acc_z_smooth'] = savgol_filter(df['acc_z'], window_length=window_size, polyorder=2)
    df['gyro_x_smooth'] = savgol_filter(df['gyro_x'], window_length=window_size, polyorder=2)
    df['gyro_y_smooth'] = savgol_filter(df['gyro_y'], window_length=window_size, polyorder=2)
    df['gyro_z_smooth'] = savgol_filter(df['gyro_z'], window_length=window_size, polyorder=2)

    # Calcular la magnitud del acelerómetro y giroscopio
    df['acc_magnitude'] = np.sqrt(df['acc_x_smooth']**2 + df['acc_y_smooth']**2 + df['acc_z_smooth']**2)
    df['gyro_magnitude'] = np.sqrt(df['gyro_x_smooth']**2 + df['gyro_y_smooth']**2 + df['gyro_z_smooth']**2)

    return df

def create_windows(data, window_size, step_size):
    """
    Divide datos en ventanas deslizantes.
    
    Args:
        data: DataFrame con las columnas de características (6 dimensiones).
        window_size: Longitud de cada ventana (timesteps).
        step_size: Número de muestras a deslizar la ventana.
    
    Returns:
        numpy array con forma (n_windows, window_size, n_features).
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
modelo_cargado = load_model("C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/modelo_lstm_alfa.h5")
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Mezcla"

for filename in os.listdir(csv_directory):
     if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        nuevas_ventanas = create_windows(df, window_size=100, step_size=50)

        # Realizar predicciones con el modelo cargado
        predicciones = modelo_cargado.predict(nuevas_ventanas, verbose=0)

        # Calcular un promedio ponderado (más confianza, más peso)
        pesos = predicciones.flatten()
        promedio_ponderado = np.dot(pesos, (pesos > 0.5).astype(int)) / np.sum(pesos)

        # Clasificación global
        resultado_global = "Extensión" if promedio_ponderado >= 0.5 else "Flexión"
        print(filename)
        print("Promedio ponderado:", promedio_ponderado)
        print("Predicción global (ponderada):", resultado_global)
        print("----------------------------")