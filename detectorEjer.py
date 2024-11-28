import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import joblib

# Inicializar Firebase Admin SDK con el archivo de credenciales JSON
cred = credentials.Certificate("C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/DatosSensor/tfgdiego-40475-firebase-adminsdk-nmb6x-3d00b98ff7.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Obtener documentos de la colección sensor_data
users_ref = db.collection("sensor_data")
docs = users_ref.stream()

# Procesar los datos de los documentos
data = []
file_names = []  # Lista para guardar los nombres de los archivos

for doc in docs:
    content = doc.to_dict().get('content', '')
    lines = content.strip().split('\n')
    doc_data = []
    for line in lines:
        values = line.split(';')
        if len(values) == 2:
            try:
                acc_values = [float(v.split('=')[1]) for v in values[0].split(',') if '=' in v]
                gyro_values = [float(v.split('=')[1]) for v in values[1].split(',') if '=' in v]
                if len(acc_values) == 3 and len(gyro_values) == 3:
                    doc_data.append(acc_values + gyro_values)
                    
            except (IndexError, ValueError) as e:
                print(f"Error procesando la línea: {line}. Error: {e}")
    print(doc_data)
    if doc_data:
        doc_data = np.mean(doc_data, axis=0)
        data.append(doc_data)
        file_names.append(doc.id)  # Guardar el nombre del archivo

# Convertir los datos a un array de numpy
data = np.array(data)

# Verificar si hay suficientes datos para el clustering
if len(data) < 2:
    print("No hay suficientes datos para realizar el clustering.")
else:
    # Escalar los datos
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Crear el modelo K-means para 2 clusters con múltiples inicializaciones
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0, n_init=14)

    # Ajustar el modelo a los datos
    kmeans.fit(data)

    # Predecir las etiquetas de los datos
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_

    # Evaluar la calidad del clustering utilizando el Silhouette Score
    silhouette_avg = silhouette_score(data, labels)
    print("Silhouette Score: ", silhouette_avg)

    # Imprimir las etiquetas y centroides
    print("Etiquetas de los ejercicios:", labels)
    print("Centroides:", centroids)

    # Mostrar los nombres de archivos agrupados por etiquetas
    cluster_0_files = [file_names[i] for i in range(len(labels)) if labels[i] == 0]
    cluster_1_files = [file_names[i] for i in range(len(labels)) if labels[i] == 1]

    print("\nArchivos en el Cluster 0:")
    for file_name in cluster_0_files:
        print(file_name)

    print("\nArchivos en el Cluster 1:")
    for file_name in cluster_1_files:
        print(file_name)

    # Reducir a 2D para visualización con PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    reduced_centroids = pca.transform(centroids)

    # Visualizar los resultados
    colors = ['red', 'blue']
    for i in range(len(reduced_data)):
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color=colors[labels[i]])
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], marker='x', color='black', s=100, linewidths=3, zorder=10)
    
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('K-means clustering de dos ejercicios')
    plt.show()
