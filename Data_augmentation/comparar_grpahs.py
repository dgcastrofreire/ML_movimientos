import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def moving_average(data, window_size):
    """Calcula el promedio móvil para suavizar datos."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def read_and_process_data(file_path):
    """Leer y procesar datos desde un archivo CSV."""
    df = pd.read_csv(file_path)
    if df.shape[1] == 8:
        df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']
    else:
        raise ValueError("El archivo CSV no tiene el número esperado de columnas.")
    
    # Aplicar filtros de suavizado
    window_sg = 101  # Ventana para Savitzky-Golay
    window_ma = 50   # Ventana para Promedio Móvil
    
    for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        df[f'{col}_savgol'] = savgol_filter(df[col], window_length=window_sg, polyorder=2)
        df[f'{col}_moving_avg'] = moving_average(df[col], window_ma)
    
    return df

def plot_signals(df, column):
    """Genera gráficos de la señal original, suavizada con Savitzky-Golay y con Promedio Móvil."""
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df[column], label='Original', alpha=0.5)
    plt.plot(df['timestamp'], df[f'{column}_savgol'], label='Savitzky-Golay', linestyle='dashed')
    plt.plot(df['timestamp'], df[f'{column}_moving_avg'], label='Moving Average', linestyle='dotted')
    
    plt.xlabel('Tiempo')
    plt.ylabel(column)
    plt.title(f'Filtro de Suavizado: {column}')
    plt.legend()
    plt.show()

# Cargar y procesar los datos
data_file = 'tu_archivo.csv'  # Reemplazar con la ruta del archivo CSV
df = read_and_process_data(data_file)

# Generar gráficos para cada componente
for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
    plot_signals(df, col)
