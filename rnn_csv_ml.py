import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def find_peaks_with_window(signal, window_size, threshold):
    peaks = []
    for i in range(len(signal) - window_size + 1):
        window = signal[i:i + window_size]
        peak_indices, _ = find_peaks(window, height=threshold)
        peaks.extend(peak_indices + i)
    return peaks

def read_data_from_csv(file_path):
    """
    Función que lee un archivo CSV y lo procesa
    """

    print(f"Leyendo archivo {file_path}")

    # Leer el archivo CSV y ajustar columnas
    df = pd.read_csv(file_path)

    # Comprueba el número de columnas para ajustar
    if df.shape[1] == 8:
        # Asignar nombres de columnas (asumiendo que la primera es un índice que no necesitamos)
        df.columns = ['index', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        df = df.drop(columns=['index'])  # Si no necesitas el índice, lo eliminamos
    elif df.shape[1] == 7:
        # Si el CSV ya tiene las 7 columnas correctas
        df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    else:
        raise ValueError(f"El archivo {file_path} no tiene el número esperado de columnas.")

    # Aplica el procesamiento que hacíamos anteriormente
    window_size = 100

    # Calcular la media móvil
    df['acc_x_ma'] = df['acc_x'].rolling(window=window_size).mean()
    df['acc_y_ma'] = df['acc_y'].rolling(window=window_size).mean()
    df['acc_z_ma'] = df['acc_z'].rolling(window=window_size).mean()

    # Desviación estándar móvil
    df['acc_x_std'] = df['acc_x'].rolling(window=window_size).std()
    df['acc_y_std'] = df['acc_y'].rolling(window=window_size).std()
    df['acc_z_std'] = df['acc_z'].rolling(window=window_size).std()

    # Suavizado con filtro Savitzky-Golay
    df['acc_x_smooth'] = savgol_filter(df['acc_x'], window_length=101, polyorder=2)

    # Normalizar los datos con MinMaxScaler
    scaler = MinMaxScaler()
    df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled']] = scaler.fit_transform(df[['acc_x_smooth', 'acc_y', 'acc_z']])

    return df

def train_lstm_model(df, file_name):
    """
    Función que entrena un modelo LSTM usando los datos procesados y realiza visualizaciones.
    """

    # Calcular el tamaño del dataset
    tamaño_acc_x = df["acc_x"].shape[0]

    # Paso 2: Calcular los índices para dividir los datos en 3/5, 1/5 y 1/5
    train_size = int(3/5 * tamaño_acc_x)  # 3/5 para el entrenamiento
    valid_size = int(1/5 * tamaño_acc_x)  # 1/5 para la validación

    # Paso 3: Dividir los datos en las tres particiones
    train_acc_x = df["acc_x"].to_numpy()[:train_size]  # 3/5 partes para entrenamiento
    valid_acc_x = df["acc_x"].to_numpy()[train_size:train_size + valid_size]  # 1/5 parte para validación
    test_acc_x = df["acc_x"].to_numpy()[train_size + valid_size:]  # 1/5 parte para test

    seq_length = 30
    tf.random.set_seed(42)

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        train_acc_x,
        targets=train_acc_x[seq_length:],
        sequence_length=seq_length,
        batch_size=32,
        shuffle=True,
        seed=42
    )

    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        valid_acc_x,
        targets=valid_acc_x[seq_length:],
        sequence_length=seq_length,
        batch_size=32
    )

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(1, input_shape=[None, 1])
    ])

    def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs=500):
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=70, restore_best_weights=True)

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=opt, metrics=["mae"])

        # Configurar gráfica en tiempo real
        plt.ion()
        fig, ax = plt.subplots()
        line_train, = ax.plot([], [], label='MAE Entrenamiento')
        line_val, = ax.plot([], [], label='MAE Validación')
        ax.set_xlim(0, epochs)
        ax.set_ylim(0, 1)  # Ajustar de acuerdo a tus valores de MAE
        ax.set_xlabel('Épocas')
        ax.set_ylabel('MAE')
        ax.set_title(f'Evolución del MAE durante el entrenamiento ({file_name})')
        ax.legend()
        
        history_mae = []
        history_val_mae = []

        for epoch in range(epochs):
            history = model.fit(train_set, validation_data=valid_set, epochs=1, verbose=0,
                                callbacks=[early_stopping_cb])

            history_mae.append(history.history['mae'][-1])
            history_val_mae.append(history.history['val_mae'][-1])

            # Actualizar líneas en el gráfico
            line_train.set_xdata(np.arange(len(history_mae)))
            line_train.set_ydata(history_mae)
            line_val.set_xdata(np.arange(len(history_val_mae)))
            line_val.set_ydata(history_val_mae)

            # Ajustar los límites del eje Y si es necesario
            ax.set_ylim(0, max(max(history_mae), max(history_val_mae)) * 1.1)

            # Actualizar la gráfica
            plt.pause(0.01)

            # Verificar si early stopping ha detenido el entrenamiento
            if early_stopping_cb.stopped_epoch > 0:
                break

        plt.ioff()  # Desactivar modo interactivo
        plt.show()

        # Evaluación final del modelo en el set de validación
        valid_loss, valid_mae = model.evaluate(valid_set)

        return model, history, valid_mae * 1e6

    # Entrenar el modelo y visualizar el MAE
    model, history, valid_mae = fit_and_evaluate(model, train_ds, valid_ds, learning_rate=0.05)

    # Predicciones en el set de test
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        test_acc_x,
        targets=test_acc_x[seq_length:],
        sequence_length=seq_length,
        batch_size=64
    )

    # Generar predicciones
    predictions = model.predict(test_ds)

    # Graficar las predicciones vs los valores reales del conjunto de prueba
    test_acc_x_plot = test_acc_x[seq_length:]  # Alinear con las predicciones
    plt.plot(test_acc_x_plot, label='Datos Reales')
    plt.plot(predictions.flatten(), label='Predicciones LSTM')
    plt.xlabel('Tiempo')
    plt.ylabel('Acc_X')
    plt.title(f'Predicciones vs Datos Reales ({file_name})')
    plt.legend()
    plt.show()

# Especifica el directorio donde están los archivos CSV
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor"

# Leer y procesar los datos desde los archivos CSV uno por uno
for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        # Entrenar el modelo LSTM con los datos leídos
        train_lstm_model(df, filename)
