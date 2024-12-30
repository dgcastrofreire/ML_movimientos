import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tensorflow as tf

def read_data_from_csv(file_path):
    """
    Leer y procesar datos CSV incluyendo acelerómetro, giroscopio y etiquetas.
    """
    try:
        df = pd.read_csv(file_path)
        if df.shape[1] == 8:
            df.columns = ['timestamp_acc', 'acc_x', 'acc_y', 'acc_z', 
                          'timestamp_gyro', 'gyro_x', 'gyro_y', 'gyro_z']
        else:
            raise ValueError("El archivo no tiene el número esperado de columnas.")

        # Suavizado con Savitzky-Golay
        window_size = 101
        for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            df[f"{col}_smooth"] = savgol_filter(df[col], window_length=window_size, polyorder=2)
        
        return df
    except Exception as e:
        print(f"Error al leer el archivo {file_path}: {e}")
        return None

def create_windows(data, window_size, step_size):
    """
    Divide datos en ventanas deslizantes.
    """
    features = ['acc_x_smooth', 'acc_y_smooth', 'acc_z_smooth', 
                'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth']
    data_array = data[features].values
    n_samples = len(data_array)

    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        window = data_array[start:start + window_size]
        windows.append(window)
    
    return np.array(windows)

def load_tflite_model(tflite_model_path):
    """
    Cargar un modelo TensorFlow Lite usando el intérprete.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def predict_with_tflite(interpreter, input_details, output_details, data):
    """
    Realizar predicciones usando un modelo TensorFlow Lite.
    """
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    predictions = []

    for window in data:
        input_data = np.expand_dims(window, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)  # Obtiene vector de probabilidades
        predictions.append(output_data[0])  # Vector de probabilidades para cada clase

    return np.array(predictions)

# Ruta del modelo .tflite
tflite_model_path = "modelo_gru_clasificacion_tiempo_real_multiclase.tflite"

# Cargar el modelo TensorFlow Lite
interpreter, input_details, output_details = load_tflite_model(tflite_model_path)

# Directorio con los datos
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/Mezcla"

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        if df is None:
            continue

        # Crear ventanas
        nuevas_ventanas = create_windows(df, window_size=100, step_size=50)

       # Realizar predicciones
        predicciones = predict_with_tflite(interpreter, input_details, output_details, nuevas_ventanas)

        # Calcular la probabilidad promedio para cada clase
        promedios_clases = np.mean(predicciones, axis=0)

        # Determinar la clase predominante
        clase_predominante = np.argmax(promedios_clases)

        # Asignar etiquetas a las clases
        etiquetas_clases = {0: "Flexión", 1: "Extensión", 2: "Idle"}
        resultado_clasificacion = etiquetas_clases[clase_predominante]

        # Imprimir resultados
        print(f"Archivo: {filename}")
        print(f"Promedios por clase: {promedios_clases}")
        print(f"Predicción global (clasificación): {resultado_clasificacion}")
        print("-" * 30)
