import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Configuración de simulación
sampling_rate = 100  # Frecuencia de muestreo en Hz
time_step = 1 / sampling_rate  # Intervalo entre muestras
window_size = 50  # Ventana deslizante
features = 11  # Número de características

# Configuración de semillas para reproducibilidad
def fijar_semillas():
    set_seed(111)
    np.random.seed(111)

fijar_semillas()

def moving_average(data, window_size):
    """Calcula el promedio móvil para suavizar datos."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def read_data_from_csv(file_path):
    """Leer y procesar datos desde un archivo CSV."""
    print(f"Leyendo archivo {file_path}")
    df = pd.read_csv(file_path)
    if df.shape[1] == 8:
        df.columns = ['timestamp_acc', 'acc_x', 'acc_y', 'acc_z', 'timestamp_gyro', 'gyro_x', 'gyro_y', 'gyro_z']
    else:
        raise ValueError(f"El archivo {file_path} no tiene el número esperado de columnas.")
    
    # Suavizado con promedio móvil
    window = 50
    df['acc_x_smooth'] = moving_average(df['acc_x'], window)
    df['acc_y_smooth'] = moving_average(df['acc_y'], window)
    df['acc_z_smooth'] = moving_average(df['acc_z'], window)
    df['gyro_x_smooth'] = moving_average(df['gyro_x'], window)
    df['gyro_y_smooth'] = moving_average(df['gyro_y'], window)
    df['gyro_z_smooth'] = moving_average(df['gyro_z'], window)
    
    # Magnitudes
    df['acc_magnitude'] = np.sqrt(df['acc_x_smooth']**2 + df['acc_y_smooth']**2 + df['acc_z_smooth']**2)
    df['gyro_magnitude'] = np.sqrt(df['gyro_x_smooth']**2 + df['gyro_y_smooth']**2 + df['gyro_z_smooth']**2)
    
    # Velocidades angulares
    df['gyro_x_velocity'] = df['gyro_x_smooth'].diff() / time_step
    df['gyro_y_velocity'] = df['gyro_y_smooth'].diff() / time_step
    df['gyro_z_velocity'] = df['gyro_z_smooth'].diff() / time_step
    
    # Llenar valores NaN generados por diferencias
    df.fillna(0, inplace=True)
    
    return df

def simulate_real_time_data(df, window_size, step_size):
    """Genera datos en tiempo real simulados desde un DataFrame."""
    features = ['acc_x_smooth', 'acc_y_smooth', 'acc_z_smooth',
                'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth',
                'gyro_x_velocity', 'gyro_y_velocity', 'gyro_z_velocity',
                'acc_magnitude', 'gyro_magnitude']
    data = df[features].values
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        yield window

def create_gru_model(input_shape, n_classes):
    """Crear un modelo GRU para la clasificación multiclase."""
    entrada = Input(shape=input_shape)
    gru_out = GRU(16, return_sequences=True)(entrada)  # Aumentado a 16 unidades
    gru_out = GRU(8)(gru_out)  # Segunda capa GRU
    output_class = Dense(n_classes, activation='softmax')(gru_out)
    model = Model(inputs=entrada, outputs=output_class)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 📌 Directorio con los datos
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Mezcla_buena_0"

# Entrenar con datos simulados en tiempo real
all_windows = []
all_labels = []

print("Simulando datos en tiempo real...")

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        
        # Asignar etiquetas según el nombre del archivo
        if "flexion" in filename.lower():
            label = 0  # Flexión
        elif "flex_bajo" in filename.lower():
            label = 0  # Flexión
        elif "extension" in filename.lower():
            label = 1  # Extensión
        elif "ext_subo" in filename.lower():
            label = 1  # Extensión
        elif "idle" in filename.lower():
            label = 2  # Quieto
        elif "gir_flex_izqda" in filename.lower() or "gir_flex_dcha" in filename.lower():
            label = 3  # Girado flexión (Izquierda y Derecha combinados)
        else:
            raise ValueError(f"No se pudo asignar etiqueta al archivo {filename}")

        # Simulación en tiempo real
        for window in simulate_real_time_data(df, window_size, step_size=10):
            all_windows.append(window)
            all_labels.append(label)

# Convertir datos en arrays
all_windows = np.array(all_windows)
all_labels = np.array(all_labels)

print(f"Datos generados: {all_windows.shape}, Etiquetas: {all_labels.shape}")

# 📌 Dividir en conjunto de entrenamiento y prueba
train_data, test_data, train_labels, test_labels = train_test_split(all_windows, all_labels, test_size=0.2, random_state=42)

# 📌 Crear y entrenar el modelo con GRU
n_classes = 4  # 🔹 Ahora hay solo 4 clases
input_shape = (window_size, features)
gru_model = create_gru_model(input_shape, n_classes)

print("Entrenando el modelo con GRU...")
gru_model.fit(
    train_data, train_labels,
    validation_data=(test_data, test_labels),
    epochs=10,
    batch_size=32
)

# 📌 Convertir el modelo GRU a TensorFlow Lite
print("Convirtiendo el modelo GRU a TensorFlow Lite...")
gru_converter = tf.lite.TFLiteConverter.from_keras_model(gru_model)
gru_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
gru_converter.experimental_enable_resource_variables = True
gru_tflite_model = gru_converter.convert()

# 📌 Guardar el modelo en formato TensorFlow Lite
gru_tflite_model_path = "multiclase_final_error_7.tflite"
with open(gru_tflite_model_path, "wb") as f:
    f.write(gru_tflite_model)

print(f"✅ Modelo GRU en formato TensorFlow Lite guardado en {gru_tflite_model_path}.")
