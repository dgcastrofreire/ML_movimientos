import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import firebase_admin
from firebase_admin import credentials, firestore
import joblib
from sklearn.preprocessing import MinMaxScaler

import scipy.signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from peakdetect import peakdetect

from decimal import Decimal

import pandas as pd
import matplotlib.pyplot as plt

# Initialize Firebase Admin SDK with credentials JSON
cred = credentials.Certificate(
    "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/DatosSensor/tfgdiego-40475-firebase-adminsdk-nmb6x-3d00b98ff7.json")
options = {"timeout": 10}  # Increase timeout to 10 seconds (default is 5)
firebase_admin.initialize_app(cred)
db = firestore.client()

def find_peaks_with_window(signal, window_size, threshold):
    
    peaks = []
    for i in range(len(signal) - window_size + 1):
        window = signal[i:i + window_size]
        peak_indices, _ = find_peaks(window, height=threshold)
        peaks.extend(peak_indices + i)
    return peaks


def read_data():

    # Fetch documents from the collection 'sensor_data'
    users_ref = db.collection("sensor_data")
    docs = users_ref.stream()

    # Process the data from the documents
    data = []

    # Column headers for DataFrame
    columns = ['timestamp_acc', 'acc_x', 'acc_y', 'acc_z',
               'timestamp_gyro', 'gyro_x', 'gyro_y', 'gyro_z']

    for doc in docs:
        doc_id = doc.id  # Obtener el ID del documento (nombre)
        content = doc.to_dict().get('content', '')
        lines = content.strip().split('\n')
        doc_data = []

        for line in lines:
            values = line.split(';')
            if len(values) == 2:
                try:
                    # Extract accelerometer data
                    acc_parts = values[0].split(',')
                    # Extract timestamp
                    timestamp_acc = [float(acc_parts[0].split('=')[1])]
                    acc_values = [float(acc_parts[i].split(
                        '=')[1] + '.' + acc_parts[i+1]) for i in range(1, len(acc_parts), 2)]

                    # Extract gyroscope data
                    gyro_parts = values[1].split(',')
                    timestamp_gyro = float(gyro_parts[0].split('=')[
                                           1])  # Extract timestamp
                    gyro_values = [float(gyro_parts[i].split(
                        '=')[1] + '.' + gyro_parts[i+1]) for i in range(1, len(gyro_parts), 2)]

                    # Ensure there are 3 values each for accelerometer and gyroscope
                    if len(acc_values) == 3 and len(gyro_values) == 3:
                        combined_data = timestamp_acc + acc_values + gyro_values
                        doc_data.append(combined_data)

                except (IndexError, ValueError) as e:
                    print(f"Error processing line: {line}. Error: {e}")
            else:
                print(
                    f"Malformed line (expecting two parts separated by ';'): {line}")

        # Convertir los datos a un DataFrame de pandas
        df = pd.DataFrame(doc_data, columns=[
                          'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])

        ''' 
        print(df.head())
        print(df.describe())
        print(df.isna().sum())
        
        '''
        # Define el tamaño de la ventana
        window_size = 100

        # Calcula la media móvil para las columnas de acelerómetro
        df['acc_x_ma'] = df['acc_x'].rolling(window=window_size).mean()
        df['acc_y_ma'] = df['acc_y'].rolling(window=window_size).mean()
        df['acc_z_ma'] = df['acc_z'].rolling(window=window_size).mean()

        # Calcula la desviación estándar móvil para las columnas de acelerómetro
        df['acc_x_std'] = df['acc_x'].rolling(window=window_size).std()
        df['acc_y_std'] = df['acc_y'].rolling(window=window_size).std()
        df['acc_z_std'] = df['acc_z'].rolling(window=window_size).std()

        # Calcula la media móvil para las columnas de giroscopio
        df['gyro_x_ma'] = df['gyro_x'].rolling(window=window_size).mean()
        df['gyro_y_ma'] = df['gyro_y'].rolling(window=window_size).mean()
        df['gyro_z_ma'] = df['gyro_z'].rolling(window=window_size).mean()

        # Calcula la desviación estándar móvil para las columnas de giroscopio
        df['gyro_x_std'] = df['gyro_x'].rolling(window=window_size).std()
        df['gyro_y_std'] = df['gyro_y'].rolling(window=window_size).std()
        df['gyro_z_std'] = df['gyro_z'].rolling(window=window_size).std()

        # Imprime algunas filas para ver los resultados
        
        print(df[['acc_x', 'acc_x_ma', 'acc_x_std', 'acc_y', 'acc_y_ma', 'acc_y_std',
                  'acc_z', 'acc_z_ma', 'acc_z_std', 'gyro_x', 'gyro_x_ma', 'gyro_x_std',
                  'gyro_y', 'gyro_y_ma', 'gyro_y_std', 'gyro_z', 'gyro_z_ma', 'gyro_z_std']].head(110))
      

        # Asegúrate de que los timestamps estén en formato numérico (no en formato de fecha)
        # Calcula la diferencia de tiempo entre muestras
        df['delta_t'] = df['timestamp'].diff()

        # Para los primeros datos donde no hay diferencia de tiempo, reemplaza NaN por 0
        df['delta_t'].fillna(0, inplace=True)

        # Integración para calcular la velocidad en cada eje (aproximación de la integral numérica)
        # Velocidad en el eje X
        df['vel_x'] = np.cumsum(df['acc_x'] * df['delta_t'])
        # Velocidad en el eje Y
        df['vel_y'] = np.cumsum(df['acc_y'] * df['delta_t'])
        # Velocidad en el eje Z
        df['vel_z'] = np.cumsum(df['acc_z'] * df['delta_t'])

        # Calcula la media móvil para las velocidades
        df['vel_x_ma'] = df['vel_x'].rolling(window=window_size).mean()
        df['vel_y_ma'] = df['vel_y'].rolling(window=window_size).mean()
        df['vel_z_ma'] = df['vel_z'].rolling(window=window_size).mean()

        # Calcula la desviación estándar móvil para las velocidades
        df['vel_x_std'] = df['vel_x'].rolling(window=window_size).std()
        df['vel_y_std'] = df['vel_y'].rolling(window=window_size).std()
        df['vel_z_std'] = df['vel_z'].rolling(window=window_size).std()
        
         # Calcula la skewness para el acelerometro
        df['skew_x_acc'] = df['acc_x'].rolling(window=window_size).skew()
        df['skew_y_acc'] = df['acc_y'].rolling(window=window_size).skew()
        df['skew_z_acc'] = df['acc_z'].rolling(window=window_size).skew()
        
        # Calcula la skewness para el gyroscopio
        df['skew_x_gyro'] = df['gyro_x'].rolling(window=window_size).skew()
        df['skew_y_gyro'] = df['gyro_y'].rolling(window=window_size).skew()
        df['skew_z_gyro'] = df['gyro_z'].rolling(window=window_size).skew()
        
        # Calcula la kurtosis para el acelerometro
        df['kurt_x_acc'] = df['acc_x'].rolling(window=window_size).kurt()
        df['kurt_y_acc'] = df['acc_y'].rolling(window=window_size).kurt()
        df['kurt_z_acc'] = df['acc_z'].rolling(window=window_size).kurt()
        
        # Calcula la kurtosis para el gyroscopio
        df['kurt_x_gyro'] = df['gyro_x'].rolling(window=window_size).kurt()
        df['kurt_y_gyro'] = df['gyro_y'].rolling(window=window_size).kurt()
        df['kurt_z_gyro'] = df['gyro_z'].rolling(window=window_size).kurt()

        
        # Imprime algunas filas para ver los resultados
        '''
        print(df[['timestamp', 'vel_x', 'vel_x_ma', 'vel_x_std',
                  'vel_y', 'vel_y_ma', 'vel_y_std',
                  'vel_z', 'vel_z_ma', 'vel_z_std']].head(110))
        '''
        
        
        # Calcula las diferencias absolutas entre valores consecutivos de aceleración
        df['acc_x_diff'] = df['acc_x'].diff().abs()
        df['acc_y_diff'] = df['acc_y'].diff().abs()
        df['acc_z_diff'] = df['acc_z'].diff().abs()

        # Imprime algunas filas para ver las diferencias absolutas calculadas
        #print(df[['timestamp', 'acc_x', 'acc_x_diff', 'acc_y', 'acc_y_diff', 'acc_z', 'acc_z_diff']].head(110))
        
       
        # Suaviza los datos de aceleración en el eje X
        df['acc_x_smooth'] = savgol_filter(df['acc_x'], window_length=101, polyorder=2)
        
        scaler = MinMaxScaler()
        df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled']] = scaler.fit_transform(df[['acc_x_smooth', 'acc_y', 'acc_z']])

        # Ajusta los parámetros para la detección de picos
        height_threshold = 0.6  # Ajusta según la magnitud esperada de los picos normalizados
        distance = 10  # Ajusta según la separación esperada entre picos

        # Detecta picos en los datos suavizados
        peaks, properties = find_peaks(df['acc_x_scaled'], prominence=0.1)

        # Imprime los índices de los picos detectados y sus alturas
        print(f"Peaks found at indices: {peaks}")
        #print(f"Peak heights: {properties['peak_heights']}")
        features = ['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled']
        df = pd.concat([df, df[features]], axis=1)
        

        # Visualización de los picos
        '''
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['acc_x'], label='Aceleración Normalizada (X)')
        plt.plot(df['timestamp'], df['acc_x_smooth'], label='Aceleración Suavizada (X)', linestyle='--')
        plt.plot(df['timestamp'][peaks], df['acc_x_smooth'][peaks], 'ro', label='Picos detectados')
        plt.xlabel('Timestamp')
        plt.ylabel('Aceleración Normalizada (X)')
        plt.title('Detección de Picos en la Aceleración Normalizada (X)')
        plt.legend()
        plt.show()
        '''
        
        

        #Building of the RNN LSTM neural network
        #testing_acc_x = 
        print("Tamaño acc_x: " , df['acc_x_ma'].to_numpy().shape)
        tamaño_acc_x = df["acc_x"].shape[0]

        # Paso 2: Calcular los índices para dividir los datos en 3/5, 1/5 y 1/5
        train_size = int(1/5 * tamaño_acc_x)  # 3/5 para el entrenamiento
        valid_size = int(3/5 * tamaño_acc_x)  # 1/5 para la validación

        # Paso 3: Dividir los datos en las tres particiones
        train_acc_x = df["acc_x"].to_numpy()[:train_size]  # 3/5 partes para entrenamiento
        valid_acc_x = df["acc_x"].to_numpy()[train_size:train_size + valid_size]  # 1/5 parte para validación
        test_acc_x = df["acc_x"].to_numpy()[train_size + valid_size:]  # 1/5 parte para test
       
        seq_length = 120
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
            sequence_length = seq_length,
            batch_size = 32
        )

        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(1, input_shape=[None,1])
        ])


        def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs=500):

            early_stopping_cb = tf.keras.callbacks.EarlyStopping(                       # Callback EarlingStopping, restaurando los pesos del mejor modelo entrenado
                monitor="val_mae", patience=70, restore_best_weights=True)

            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)    # Optimizador Decenso de Gradiente Estocástico

            model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"]) # Compilacion del modelo : Funcion de perdida Hubber,optimizador SGD y error absoluto medio (mae) como métrica

            history = model.fit(train_set, validation_data=valid_set, epochs=epochs,    # Entrenamiento del modelo (fit)
                                callbacks=[early_stopping_cb])

            valid_loss, valid_mae = model.evaluate(valid_set)                           # Perdia y mae de validacion

            return valid_mae * 1e6
        
        fit_and_evaluate(model, train_ds, valid_ds, learning_rate=0.02)                 # Se invoca la funcion fit_and_evaluate para hacer Compilacion , Entrenamiento etc

df = read_data()
