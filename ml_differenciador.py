import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt

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

    # Normalizar usando MinMaxScaler
    scaler = MinMaxScaler()
    df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled', 'acc_magnitude_scaled', 'gyro_magnitude_scaled']] = scaler.fit_transform(
        df[['acc_x_smooth', 'acc_y_smooth', 'acc_z_smooth', 'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth', 'acc_magnitude', 'gyro_magnitude']]
    )

    return df

def train_random_forest(df):
    """
    Entrenar un clasificador Random Forest usando los datos procesados.
    """
    # Separar características (features) y etiquetas
    features = df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled', 'acc_magnitude_scaled', 'gyro_magnitude_scaled']]
    labels = df['label']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Entrenar el modelo Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = rf_clf.predict(X_test)
    print("Reporte de clasificación para Random Forest:")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    return rf_clf

def train_lstm_model(df):
    """
    Entrenar un modelo LSTM usando los datos procesados y realizar la clasificación binaria (bien/mal).
    """
    # Separar características (features) y etiquetas
    features = df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled', 'acc_magnitude_scaled', 'gyro_magnitude_scaled']].to_numpy()
    labels = df['label'].to_numpy()

    # Dividir los datos en entrenamiento, validación y prueba usando stratified sampling
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_features, train_labels, test_size=0.25, random_state=42, stratify=train_labels
    )

    # Crear datasets para el modelo LSTM
    seq_length = 30

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        train_features,
        targets=train_labels[seq_length:],
        sequence_length=seq_length,
        batch_size=32,
        shuffle=True
    )

    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        valid_features,
        targets=valid_labels[seq_length:],
        sequence_length=seq_length,
        batch_size=32
    )

    # Crear el modelo LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(seq_length, train_features.shape[1]), return_sequences=True),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Clasificación binaria
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Entrenar el modelo con early stopping
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    history = model.fit(train_ds, validation_data=valid_ds, epochs=200, callbacks=[early_stopping_cb])

    return model, history

# Especifica el directorio donde están los archivos CSV
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor"

# Leer y procesar los datos desde los archivos CSV uno por uno
for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        
        # Entrenar y evaluar el modelo Random Forest
        rf_model = train_random_forest(df)
        
        # Entrenar el modelo LSTM con los datos leídos
        lstm_model, history = train_lstm_model(df)

# Guardar el modelo LSTM entrenado al final del proceso
global_model_path = "lstm_model_final.h5"
lstm_model.save(global_model_path)
print(f"Modelo LSTM final guardado como {global_model_path}")