import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

'''
def detect_peaks_derivative(data):
    """Detecta picos basados en la primera derivada.

    Args:
        data: Arreglo NumPy con los datos de la serie temporal.

    Returns:
        Lista de índices de los picos.
    """

    # Calcular la primera derivada
    derivative = np.gradient(data)

    # Encontrar los índices donde la derivada cambia de signo
    peaks = np.where(np.diff(np.sign(derivative)))[0]
    print("Número de picos detectados:", len(peaks))
    return peaks
'''


#oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from 
# different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
TF_ENABLE_ONEDNN_OPTS=0 

# Initialize Firebase Admin SDK with credentials JSON
cred = credentials.Certificate(
    "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/DatosSensor/tfgdiego-40475-firebase-adminsdk-nmb6x-3d00b98ff7.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# Fetch documents from the collection 'sensor_data'
users_ref = db.collection("sensor_data")
docs = users_ref.stream()

# Process the data from the documents
data = []

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
                timestamp_acc = float(acc_parts[0].split('=')[1])  # Extract timestamp
                acc_values = [float(acc_parts[i].split('=')[1] + '.' + acc_parts[i+1]) for i in range(1, len(acc_parts), 2)]
     
                # Extract gyroscope data
                gyro_parts = values[1].split(',')
                timestamp_gyro = float(gyro_parts[0].split('=')[1])  # Extract timestamp
                gyro_values = [float(gyro_parts[i].split('=')[1] + '.' + gyro_parts[i+1]) for i in range(1, len(gyro_parts), 2)]
     
                # Ensure there are 3 values each for accelerometer and gyroscope
                if len(acc_values) == 3 and len(gyro_values) == 3:
                    combined_data = acc_values + gyro_values
                    doc_data.append(combined_data)

            except (IndexError, ValueError) as e:
                print(f"Error processing line: {line}. Error: {e}")
        else:
            print(f"Malformed line (expecting two parts separated by ';'): {line}")
    
    # Convertir los datos a un DataFrame de pandas
    df = pd.DataFrame(doc_data, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    
     # Aplicar el filtro de Savitzky-Golay para suavizar los datos
    window_length = 100  # Debe ser un número impar
    polyorder = 2  # Orden del polinomio

    df_filtered = df.apply(lambda x: savgol_filter(x, window_length, polyorder))
    
    # Normalizar los datos
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_filtered)
    
    '''
    # Crear una figura y dos subgráficas
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Plotear los datos del acelerómetro en la subgráfica superior
    ax1.plot(df.index, data_scaled[:, 0], 'g-', label='Acc X')
    ax1.plot(df.index, data_scaled[:, 1], 'b-', label='Acc Y')
    ax1.plot(df.index, data_scaled[:, 2], 'r-', label='Acc Z')
    ax1.set_ylabel('Acelerómetro')
    ax1.legend(loc='upper right')
    ax1.set_title('Datos del Acelerómetro y Giroscopio')

    # Plotear los datos del giroscopio en la subgráfica inferior
    ax2.plot(df.index, data_scaled[:, 3], 'g--', label='Gyro X')
    ax2.plot(df.index, data_scaled[:, 4], 'b--', label='Gyro Y')
    ax2.plot(df.index, data_scaled[:, 5], 'r--', label='Gyro Z')
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Giroscopio')
    ax2.legend(loc='upper right')

    # Ajustar el espacio entre las subgráficas
    plt.tight_layout()
    '''
    # Mostrar la gráfica
    #plt.show()
    
    peaks_accX = find_peaks(data_scaled[:, 0], prominence=0.4)
    #print("Peaks position:", peaks[0])
    
    peaks_accY = find_peaks(data_scaled[:, 1], prominence=0.4)
    #print("Peaks position:", peaks[0])
    
    peaks_accZ = find_peaks(data_scaled[:, 2], prominence=0.4)
    #print("Peaks position:", peaks[0])
    
    peaks_gyroX = find_peaks(data_scaled[:, 3], prominence=0.4)
    #print("Peaks position:", peaks[0])
    
    peaks_gyroY = find_peaks(data_scaled[:, 4], prominence=0.4)
    #print("Peaks position:", peaks[0])
    
    peaks_gyroZ = find_peaks(data_scaled[:, 5], prominence=0.4)
    #print("Peaks position:", peaks[0])
    
    print (len(peaks_accX[0]), len(peaks_accY[0]), len(peaks_accZ[0]), len(peaks_gyroX[0]), len(peaks_gyroY[0]), len(peaks_gyroZ[0]))
    
    num_reps = (round(statistics.median([len(peaks_accX[0]), len(peaks_accY[0]), len(peaks_accZ[0]), len(peaks_gyroX[0]), len(peaks_gyroY[0]), len(peaks_gyroZ[0])])))
    
    print(num_reps)
    
     # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data_scaled[:, 0], label='Acc X')
    plt.title(f"Finding Peaks - Documento: {doc_id}")
    plt.xlabel('Tiempo')
    plt.ylabel('Aceleración (normalizada)')
    
    # Marcar los picos detectados
    [plt.axvline(p, c='C3', linewidth=0.3) for p in peaks_accX[0]]
    
    plt.legend()
    plt.show()

