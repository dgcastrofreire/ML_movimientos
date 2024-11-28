from pyexpat import model
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.preprocessing import scale

# Inicializar Firebase
cred = credentials.Certificate("C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/DatosSensor/tfgdiego-40475-firebase-adminsdk-nmb6x-3d00b98ff7.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Función para analizar datos en tiempo real
def analyze_data_in_real_time():
    # Recuperar el ID del documento más reciente
    doc_ref = db.collection('current_session').document('latest')
    doc = doc_ref.get()
    if doc.exists:
        document_id = doc.to_dict()['document_id']
        print(f"Listening to document ID: {document_id}")
        listen_to_document(document_id)
    else:
        print("No document found")

def listen_to_document(document_id):
    doc_ref = db.collection('sensor_data').document(document_id)
    doc_watch = doc_ref.on_snapshot(on_snapshot)

def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        data = doc.to_dict()
        sensor_data = data['data'].split(';')
        features = np.array(sensor_data).reshape(1, -1)
        features = scale.transform(features)
        prediction = model.predict(features)
        if prediction == 1:
            print("Movimiento detectado")
        else:
            print("No hay movimiento")

# Iniciar la escucha en tiempo real
analyze_data_in_real_time()