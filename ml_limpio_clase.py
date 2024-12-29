import os
import random
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.random import set_seed

def read_data_from_csv(file_path):
    """
    Leer y procesar datos CSV incluyendo acelerómetro y giroscopio.
    """
    print(f"Leyendo archivo {file_path}")
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

# Configurar semillas para reproducibilidad
def fijar_semillas():
    set_seed(111)
    np.random.seed(111)
    random.seed(111)

# Directorio con los archivos CSV
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Mezcla"

# Acumuladores para los datos y etiquetas
all_windows = []
all_labels = []

# Parámetros para las ventanas
window_size = 100  # Número de timesteps por ventana
step_size = 50     # Desplazamiento entre ventanas

# Lectura y etiquetado
for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)

        # Crear ventanas
        windows = create_windows(df, window_size, step_size)
        all_windows.append(windows)

        # Etiquetas: 0 = flexión, 1 = extensión (basado en el nombre del archivo)
        if "flexion" in filename.lower():
            labels_class = np.zeros(shape=(windows.shape[0],))  # Etiquetas 0
        elif "extension" in filename.lower():
            labels_class = np.ones(shape=(windows.shape[0],))  # Etiquetas 1
        else:
            raise ValueError(f"No se pudo asignar etiqueta al archivo {filename}")
        
        all_labels.append(labels_class)

# Concatenar los datos y etiquetas
all_windows = np.concatenate(all_windows, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"Forma de los datos para LSTM: {all_windows.shape}")  # (n_samples, timesteps, features)
print(f"Forma de las etiquetas: {all_labels.shape}")

# Dividir en entrenamiento y prueba
train_data, test_data, train_labels, test_labels = train_test_split(
    all_windows, all_labels, test_size=0.2, random_state=42
)

# Crear el modelo LSTM
fijar_semillas()
entrada = Input(shape=(window_size, all_windows.shape[2]))
lstm_out = LSTM(64)(entrada)
output_class = Dense(1, activation='sigmoid', name='classification')(lstm_out)

modelo = Model(inputs=entrada, outputs=output_class)

# Compilar el modelo
modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
modelo.fit(
    train_data,
    train_labels,
    validation_data=(test_data, test_labels),
    epochs=10
)

# Guardar el modelo
modelo.save("modelo_lstm_clasificacion.h5")
print("Modelo guardado exitosamente.")
