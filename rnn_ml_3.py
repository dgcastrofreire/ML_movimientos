import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def find_peaks_with_window(signal, window_size, threshold):
    peaks = []
    for i in range(len(signal) - window_size + 1):
        window = signal[i:i + window_size]
        peak_indices, _ = find_peaks(window, height=threshold)
        peaks.extend(peak_indices + i)
    return peaks

def plot_learning_curves(history):
    # Graficar la pérdida
    plt.figure(figsize=(12, 4))
    
    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida Validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Exactitud
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Exactitud Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Exactitud Validación')
    plt.title('Exactitud durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Exactitud')
    plt.legend()
    
    plt.show()

def evaluate_model(model, test_ds):
    # Obtener predicciones del modelo
    predictions = model.predict(test_ds)
    predictions = (predictions > 0.5).astype("int32")  # Convertir predicciones de probabilidad a etiquetas

    # Extraer etiquetas verdaderas
    true_labels = np.concatenate([y for x, y in test_ds], axis=0)

    # Reporte de clasificación
    print(classification_report(true_labels, predictions, target_names=["Mal", "Bien"]))

    # Matriz de confusión
    cm = confusion_matrix(true_labels, predictions)
    print("Matriz de confusión:")
    print(cm)

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

    # Normalizar usando MinMaxScaler
    scaler = MinMaxScaler()
    df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled']] = scaler.fit_transform(
        df[['acc_x_smooth', 'acc_y_smooth', 'acc_z_smooth', 'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth']]
    )

    return df

def build_lstm_classifier(input_shape):
    """
    Construir un modelo LSTM para clasificación binaria.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Clasificación binaria
    ])
    
    model.compile(
        loss='binary_crossentropy',  # Clasificación binaria
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

def fit_and_evaluate(model, train_set, valid_set, learning_rate=0.001, epochs=500):
    """
    Entrenar el modelo con early stopping y evaluar.
    """
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    history = model.fit(train_set, validation_data=valid_set, epochs=epochs, callbacks=[early_stopping_cb])

    return model, history

def train_lstm_model(df, file_name):
    """
    Entrenar un modelo LSTM usando los datos procesados y realizar la clasificación binaria (bien/mal).
    """

    # Paso 1: Dividir los datos en entrenamiento, validación y prueba
    tamaño_acc_x = df["acc_x"].shape[0]
    train_size = int(3/5 * tamaño_acc_x)  # 3/5 para el entrenamiento
    valid_size = int(1/5 * tamaño_acc_x)  # 1/5 para la validación

    # Separar características (features) y etiquetas
    features = df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled']].to_numpy()
    labels = df['label'].to_numpy()

    # Dividir en entrenamiento, validación y prueba
    train_features = features[:train_size]
    valid_features = features[train_size:train_size + valid_size]
    test_features = features[train_size + valid_size:]

    train_labels = labels[:train_size]
    valid_labels = labels[train_size:train_size + valid_size]
    test_labels = labels[train_size + valid_size:]

    # Paso 2: Crear datasets para el modelo LSTM
    seq_length = 30

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        train_features,
        targets=train_labels[seq_length:],  # Etiquetas
        sequence_length=seq_length,
        batch_size=32,
        shuffle=True
    )

    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        valid_features,
        targets=valid_labels[seq_length:],  # Etiquetas
        sequence_length=seq_length,
        batch_size=32
    )

    # Paso 3: Crear y entrenar el modelo
    model = build_lstm_classifier(input_shape=(seq_length, train_features.shape[1]))

    # Entrenar el modelo con las etiquetas
    model, history = fit_and_evaluate(model, train_ds, valid_ds, learning_rate=0.001)

    # Evaluar en el conjunto de prueba
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        test_features,
        targets=test_labels[seq_length:],  # Etiquetas
        sequence_length=seq_length,
        batch_size=32
    )

    # Evaluar el modelo en el conjunto de prueba
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f'Accuracy en el conjunto de prueba: {test_accuracy}')
    #model, history = fit_and_evaluate(model, train_ds, valid_ds, learning_rate=0.001)
    plot_learning_curves(history)

    return model, history


# Especifica el directorio donde están los archivos CSV
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor"


# Leer y procesar los datos desde los archivos CSV uno por uno
for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        # Entrenar el modelo LSTM con los datos leídos
        train_lstm_model(df, filename)
       