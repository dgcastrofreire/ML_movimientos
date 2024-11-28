import json
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import Sequential

from sklearn.discriminant_analysis import StandardScaler
from tensorflow import keras


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

        # Define el tamaño de la ventana
        window_size = 100
        
        # Limpiar los últimos dos registros que parecen ser incorrectos
        df = df.drop(len(df)-1, axis=0)
        df = df.drop(len(df)-1, axis=0)

        Obs = 20  # Número de observaciones para la ventana deslizante

        # Escalar los datos usando StandardScaler
        sc = StandardScaler()
        df['acc_x_trans'] = sc.fit_transform(np.array(df['acc_x']).reshape(-1, 1))        
        Data = np.asarray(df['acc_x_trans'])
        Data = np.atleast_2d(Data).T  # Transponer los datos para que tengan la forma correcta
        
        # Preparar las secuencias para entrenar la LSTM
        X = np.atleast_3d(np.array([Data[start:start + Obs] for start in range(0, Data.shape[0] - Obs)]))
        y = Data[Obs:]

        print(len(X), len(y))  # Verificar las dimensiones

        # Crear el modelo LSTM
        model = Sequential()
        model.add(LSTM(15, input_shape=(Obs, 1), return_sequences=True))  # Primera capa LSTM
        model.add(LSTM(15, return_sequences=False))  # Segunda capa LSTM
        model.add(Dense(1))  # Capa de salida
        model.add(Activation('linear'))  # Activación lineal
        
        # Compilar el modelo con 'mape' (mean absolute percentage error) y optimizador RMSprop
        model.compile(loss="mape", optimizer="rmsprop")
        
        # Entrenar el modelo
        NN = model.fit(X, y, epochs=200, batch_size=50, verbose=2, shuffle=False)
        
        # Guardar el historial de entrenamiento como un CSV
        pd.DataFrame(NN.history).to_csv("sensor_data_nominal_loss.csv")
        
        # Guardar los parámetros del modelo como JSON
        model_params = json.dumps(NN.params)
        with open("sensor_data_nominal_params.json", "w") as json_file:
            json_file.write(model_params)

        # Guardar el modelo en formato JSON
        model_json = model.to_json()
        with open("sensor_data_nominal.json", "w") as json_file:
            json_file.write(model_json)

        # Guardar los pesos del modelo
        model.save_weights("sensor_data_nominal_weights.h5")

            
        Predictions = [model.predict(np.asarray(df['Price'])[i:i+Obs].reshape(1, Obs, 1)) for i in range(len(df)-Obs)]

        Predictions = [df['Price'].iloc[0]]*Obs + [val[0][0] for val in Predictions]



        df['Predictions'] = Predictions

        df['acc_x_predicted'] = sc.inverse_transform(df['acc_x_trans'])

        df['Predictions'] = sc.inverse_transform(df['Predictions'])

        plt.xlabel('Date')

        plt.ylabel('Price')

        plt.plot(df['acc_x_predicted'], 'b', label='acc_x_predicted')

        plt.plot(df['Predictions'], 'r', label='Prediction')

        plt.legend(loc='upper left', shadow=True, fontsize='x-large')






df = read_data()




