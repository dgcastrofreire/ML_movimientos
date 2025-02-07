import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math

# 📌 Directorios de entrada y salida
input_folder = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Data_augmentation"
output_folder = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Data_augmentation_Aumentado"

# 📌 Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# 📌 Función para agregar ruido gaussiano
def add_gaussian_noise(df, noise_level=0.02):
    noisy_df = df.copy()
    sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in sensor_columns:
        noisy_df[col] += np.random.normal(0, noise_level, size=len(df))
    return noisy_df

# 📌 Función para jittering (pequeñas variaciones aleatorias)
def jitter(df, jitter_factor=0.01):
    sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    jittered_df = df.copy()
    for col in sensor_columns:
        jittered_df[col] += jitter_factor * np.std(df[col]) * np.random.randn(len(df))
    return jittered_df

# 📌 Función para rotación de los ejes
def rotate_data(df, angle_degrees=10):
    angle = math.radians(angle_degrees)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)

    rotated_df = df.copy()
    rotated_df['acc_x'] = cos_angle * df['acc_x'] - sin_angle * df['acc_y']
    rotated_df['acc_y'] = sin_angle * df['acc_x'] + cos_angle * df['acc_y']
    
    rotated_df['gyro_x'] = cos_angle * df['gyro_x'] - sin_angle * df['gyro_y']
    rotated_df['gyro_y'] = sin_angle * df['gyro_x'] + cos_angle * df['gyro_y']
    
    return rotated_df

# 📌 Función para interpolación de datos (manteniendo timestamps)
def interpolate_data(df, factor=2):
    """
    Interpola los datos para aumentar la cantidad de muestras manteniendo los timestamps originales.
    """
    sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    timestamps = df[['timestamp_acc', 'timestamp_gyro']].values  # Guardar los timestamps originales
    
    new_length = len(df) * factor
    interpolated_df = pd.DataFrame()

    for col in sensor_columns:
        interp = interp1d(np.linspace(0, 1, len(df)), df[col], kind="cubic")
        interpolated_df[col] = interp(np.linspace(0, 1, new_length))

    # Interpolar los timestamps también para mantener coherencia temporal
    timestamp_interp = interp1d(np.linspace(0, 1, len(timestamps)), timestamps, axis=0, kind="linear")
    interpolated_timestamps = timestamp_interp(np.linspace(0, 1, new_length))

    interpolated_df.insert(0, 'timestamp_acc', interpolated_timestamps[:, 0])
    interpolated_df.insert(4, 'timestamp_gyro', interpolated_timestamps[:, 1])

    return interpolated_df

# 📌 Procesar todos los archivos en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # Asegurar que las columnas sean correctas
        expected_columns = ['timestamp_acc', 'acc_x', 'acc_y', 'acc_z', 'timestamp_gyro', 'gyro_x', 'gyro_y', 'gyro_z']
        if list(df.columns) != expected_columns:
            print(f"⚠️ Error: {filename} tiene columnas incorrectas. Se omite este archivo.")
            continue

        # Aplicar Data Augmentation respetando timestamps
        df_noisy = add_gaussian_noise(df)
        df_jittered = jitter(df)
        df_rotated = rotate_data(df)
        df_interpolated = interpolate_data(df)

        # Guardar los archivos aumentados con nombres modificados
        df_noisy.to_csv(os.path.join(output_folder, f"noisy_{filename}"), index=False)
        df_jittered.to_csv(os.path.join(output_folder, f"jittered_{filename}"), index=False)
        df_rotated.to_csv(os.path.join(output_folder, f"rotated_{filename}"), index=False)
        df_interpolated.to_csv(os.path.join(output_folder, f"interpolated_{filename}"), index=False)

print("✅ Data Augmentation completado. Archivos guardados en:", output_folder)

