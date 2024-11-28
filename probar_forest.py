import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import joblib

def read_data_from_csv(file_path):
    """
    Leer y procesar datos CSV incluyendo acelerómetro, giroscopio y etiquetas (bien/mal).
    """
    print(f"Leyendo archivo {file_path}")
    df = pd.read_csv(file_path)

    # Asignar nombres de columnas
    if df.shape[1] == 9:
        df.columns = ['index', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']
        df = df.drop(columns=['index'])
    elif df.shape[1] == 8:
        df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']
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

def load_scaler(scaler_filename="scaler.pkl"):
    """
    Cargar el escalador para normalizar los datos.
    """
    scaler = joblib.load(scaler_filename)
    print(f"Escalador cargado desde {scaler_filename}")
    return scaler

def load_model(model_filename="random_forest_model.pkl"):
    """
    Cargar el modelo Random Forest desde un archivo.
    """
    rf_model = joblib.load(model_filename)
    print(f"Modelo cargado desde {model_filename}")
    return rf_model

def predict_and_print_label(rf_model, scaler, df):
    """
    Utilizar el modelo entrenado para predecir la etiqueta de todo el ejercicio y 
    imprimir la clase predicha (0 o 1).
    """
    # Aplicar la normalización usando el escalador cargado
    df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled', 'acc_magnitude_scaled', 'gyro_magnitude_scaled']] = scaler.transform(
        df[['acc_x_smooth', 'acc_y_smooth', 'acc_z_smooth', 'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth', 'acc_magnitude', 'gyro_magnitude']]
    )

    features = df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled', 'acc_magnitude_scaled', 'gyro_magnitude_scaled']]
    
    # Predecir la etiqueta para todos los datos del ejercicio
    predicted_labels = rf_model.predict(features)

    # Contar cuál es la clase más frecuente (votación de la mayoría)
    unique, counts = np.unique(predicted_labels, return_counts=True)
    majority_label = unique[np.argmax(counts)]

    # Imprimir la clase asignada al ejercicio
    print(f"El ejercicio ha sido clasificado como: {majority_label} (Label 0 o Label 1)")

# Especifica el directorio donde están los archivos CSV
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor"

# Cargar el modelo y el escalador desde los archivos .pkl
scaler = load_scaler("scaler.pkl")
rf_model = load_model("random_forest_model.pkl")

# Leer y procesar los datos desde los archivos CSV uno por uno y hacer predicciones
for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)

        # Predecir y mostrar la etiqueta asignada para el ejercicio completo
        predict_and_print_label(rf_model, scaler, df)
