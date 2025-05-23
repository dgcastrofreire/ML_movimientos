import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split
import time

# Configuración de simulación
sampling_rate = 100  # Frecuencia de muestreo en Hz
time_step = 1 / sampling_rate  # Intervalo entre muestras
window_size = 100  # Ventana deslizante (número de timesteps por ventana)
features = 6  # Número de características (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)

# Configuración de semillas para reproducibilidad
def fijar_semillas():
    set_seed(111)
    np.random.seed(111)

fijar_semillas()

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
    window = 101
    df['acc_x_smooth'] = savgol_filter(df['acc_x'], window_length=window, polyorder=2)
    df['acc_y_smooth'] = savgol_filter(df['acc_y'], window_length=window, polyorder=2)
    df['acc_z_smooth'] = savgol_filter(df['acc_z'], window_length=window, polyorder=2)
    df['gyro_x_smooth'] = savgol_filter(df['gyro_x'], window_length=window, polyorder=2)
    df['gyro_y_smooth'] = savgol_filter(df['gyro_y'], window_length=window, polyorder=2)
    df['gyro_z_smooth'] = savgol_filter(df['gyro_z'], window_length=window, polyorder=2)

    return df

def simulate_real_time_data(df, window_size, step_size):
    """
    Genera datos en tiempo real simulados desde un DataFrame.
    """
    features = ['acc_x_smooth', 'acc_y_smooth', 'acc_z_smooth',
                'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth']
    data = df[features].values
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        yield window

# Crear el modelo LSTM
def create_model(input_shape):
    """
    Crear un modelo LSTM para la clasificación binaria.
    """
    entrada = Input(shape=input_shape)
    lstm_out = LSTM(16)(entrada)
    output_class = Dense(1, activation='sigmoid')(lstm_out)
    model = Model(inputs=entrada, outputs=output_class)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Directorio con los datos
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Mezcla"

# Entrenar con datos simulados en tiempo real
all_windows = []
all_labels = []

print("Simulando datos en tiempo real...")

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        
        # Etiqueta según el tipo de ejercicio
        if "flexion" in filename.lower():
            label = 0  # Flexión
        elif "extension" in filename.lower():
            label = 1  # Extensión
        else:
            raise ValueError(f"No se pudo asignar etiqueta al archivo {filename}")

        # Simulación en tiempo real
        for window in simulate_real_time_data(df, window_size, step_size=10):  # 10 pasos (100 ms)
            all_windows.append(window)
            all_labels.append(label)

# Convertir datos en arrays
all_windows = np.array(all_windows)
all_labels = np.array(all_labels)

print(f"Datos generados: {all_windows.shape}, Etiquetas: {all_labels.shape}")

# Dividir en conjunto de entrenamiento y prueba
train_data, test_data, train_labels, test_labels = train_test_split(all_windows, all_labels, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
input_shape = (window_size, features)
model = create_model(input_shape)

print("Entrenando el modelo...")
model.fit(
    train_data, train_labels,
    validation_data=(test_data, test_labels),
    epochs=10,
    batch_size=32
)

# Guardar el modelo entrenado
model.save("modelo_lstm_clasificacion_tiempo_real.h5")
print("Modelo guardado exitosamente.")
