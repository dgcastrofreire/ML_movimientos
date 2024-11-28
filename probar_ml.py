import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import os

def read_data_for_testing(file_path):
    """
    Leer y procesar datos CSV para la evaluación del modelo.
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

    # Normalizar usando MinMaxScaler
    scaler = MinMaxScaler()
    df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled']] = scaler.fit_transform(
        df[['acc_x_smooth', 'acc_y_smooth', 'acc_z_smooth', 'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth']]
    )

    return df

def evaluate_saved_model(model_path, test_data_path):
    """
    Cargar el modelo guardado y evaluarlo con nuevos datos.
    """
    # Cargar el modelo
    model = tf.keras.models.load_model(model_path)
    print(f"Modelo cargado desde {model_path}")

    # Leer y procesar los datos de prueba
    df = read_data_for_testing(test_data_path)
    features = df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled']].to_numpy()
    labels = df['label'].to_numpy()

    # Crear dataset para evaluación
    seq_length = 30
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        features[:-seq_length],  # Alinear correctamente los features con las etiquetas
        targets=labels[seq_length:],
        sequence_length=seq_length,
        batch_size=32
    )

    # Evaluar el modelo en el nuevo conjunto de prueba
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f'Accuracy en el nuevo conjunto de prueba: {test_accuracy}')

    # Predicciones para verificar agrupamiento
    predictions = model.predict(test_ds)
    predictions = (predictions > 0.5).astype(int)

    # Ajustar el tamaño de las etiquetas para que coincida con las predicciones
    true_labels = labels[seq_length:len(predictions) + seq_length]

    # Agrupar y mostrar los resultados
    grouped_results = pd.DataFrame({'Predicted_Label': predictions.flatten(), 'True_Label': true_labels})
    grouped_results_sorted = grouped_results.sort_values(by='Predicted_Label')
    print("\nResultados agrupados por etiquetas predichas:\n")
    print(grouped_results_sorted)

# Ruta del modelo guardado
global_model_path = "lstm_model_final.h5"

# Ruta del archivo CSV con los nuevos datos para evaluar
test_data_path = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor/extension_cuello_12rep_1.csv"

evaluate_saved_model(global_model_path, test_data_path)