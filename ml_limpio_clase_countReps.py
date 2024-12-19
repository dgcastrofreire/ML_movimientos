import os
import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, butter, lfilter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re

import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.random import set_seed
        
        

def read_data_from_csv(file_path):
    """
    Leer y procesar datos CSV incluyendo acelerómetro, giroscopio y etiquetas (bien/mal).
    """
    print(f"Leyendo archivo {file_path}")
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
    window_size = 220
    df['acc_x_smooth'] = savgol_filter(df['acc_x'], window_length=window_size, polyorder=1, mode="nearest")
    df['acc_y_smooth'] = savgol_filter(df['acc_y'], window_length=window_size, polyorder=1, mode="nearest")
    df['acc_z_smooth'] = savgol_filter(df['acc_z'], window_length=window_size, polyorder=1, mode="nearest")
    df['gyro_x_smooth'] = savgol_filter(df['gyro_x'], window_length=window_size,polyorder=1, mode="nearest")
    df['gyro_y_smooth'] = savgol_filter(df['gyro_y'], window_length=window_size, polyorder=1, mode="nearest")
    df['gyro_z_smooth'] = savgol_filter(df['gyro_z'], window_length=window_size, polyorder=1, mode="nearest")

    # Calcular la magnitud del acelerómetro y giroscopio
    df['acc_magnitude'] = np.sqrt(df['acc_x_smooth']**2 + df['acc_y_smooth']**2 + df['acc_z_smooth']**2)
    df['gyro_magnitude'] = np.sqrt(df['gyro_x_smooth']**2 + df['gyro_y_smooth']**2 + df['gyro_z_smooth']**2)
    
    T = 1/100
    fs = 100

    
# Gráfica del Acelerómetro y Giroscopio en X por separado
    plt.figure(figsize=(10, 12))

    plt.subplot(3, 2, 1)
    plt.plot(df['timestamp_acc'], df['acc_x_smooth'], label='Acelerómetro X', color='b')
    plt.title('Acelerómetro en X')
    plt.xlabel('Tiempo (timestamp)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(df['timestamp_gyro'], df['gyro_x_smooth'], label='Giroscopio X', color='r')
    plt.title('Giroscopio en X')
    plt.xlabel('Tiempo (timestamp)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)

    # Gráfica del Acelerómetro y Giroscopio en Y por separado
    plt.subplot(3, 2, 3)
    plt.plot(df['timestamp_acc'], df['acc_y_smooth'], label='Acelerómetro Y', color='g')
    plt.title('Acelerómetro en Y')
    plt.xlabel('Tiempo (timestamp)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(df['timestamp_gyro'], df['gyro_y_smooth'], label='Giroscopio Y', color='orange')
    plt.title('Giroscopio en Y')
    plt.xlabel('Tiempo (timestamp)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)

    # Gráfica del Acelerómetro y Giroscopio en Z por separado
    plt.subplot(3, 2, 5)
    plt.plot(df['timestamp_acc'], df['acc_z_smooth'], label='Acelerómetro Z', color='purple')
    plt.title('Acelerómetro en Z')
    plt.xlabel('Tiempo (timestamp)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(df['timestamp_gyro'], df['gyro_z_smooth'], label='Giroscopio Z', color='brown')
    plt.title('Giroscopio en Z')
    plt.xlabel('Tiempo (timestamp)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Graficar la magnitud del acelerómetro por separado
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp_acc'], df['acc_magnitude'], label='Magnitud del Acelerómetro', color='b', alpha=0.7)
    plt.title('Magnitud del Acelerómetro')
    plt.xlabel('Tiempo (timestamp)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Graficar la magnitud del giroscopio por separado
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp_gyro'], df['gyro_magnitude'], label='Magnitud del Giroscopio', color='r', alpha=0.7)
    plt.title('Magnitud del Giroscopio')
    plt.xlabel('Tiempo (timestamp)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
      
    return df

def extract_repetitions(filename):
    
    match = re.search(r'_(\d+)rep_', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No se encontraron repeticiones en el archivo {filename}")
    

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


# Función fijar semillas (para reproducibilidad de los resultados)
def fijar_semillas():
    set_seed(111)
    np.random.seed(111)
    random.seed(111)
    
    
    
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Mezcla"

# Acumuladores para los datos y etiquetas
all_windows = []
all_labels = []

# Parámetros para las ventanas
window_size = 150 # Número de timesteps por ventana
step_size = 50     # Desplazamiento entre ventanas

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)