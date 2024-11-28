import csv
import firebase_admin
from firebase_admin import credentials, firestore
import unicodedata

# Inicializar la conexión a Firestore
cred = credentials.Certificate("C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/DatosSensor/tfgdiego-40475-firebase-adminsdk-nmb6x-3d00b98ff7.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def remove_accents(input_str):
    # Normalizar la cadena eliminando los acentos
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def export_documents_to_csv():
    # Referencia a la colección en Firestore
    users_ref = db.collection("sensor_data")
    docs = users_ref.stream()

    for doc in docs:
        doc_id = doc.id  # Obtener el ID del documento

        # Eliminar acentos del nombre del archivo
        doc_id_normalized = remove_accents(doc_id)

        content = doc.to_dict().get('content', '')
        lines = content.strip().split('\n')

        # Crear un archivo CSV para cada documento con el nombre sin acentos
        with open(f'{doc_id_normalized}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Escribir encabezados en el archivo CSV
            writer.writerow(['timestamp_acc', 'acc_x', 'acc_y', 'acc_z', 'timestamp_gyro', 'gyro_x', 'gyro_y', 'gyro_z'])

            for line in lines:
                values = line.split(';')
                if len(values) == 2:
                    try:
                        acc_parts = values[0].split(',')
                        gyro_parts = values[1].split(',')

                        timestamp_acc = float(acc_parts[0].split('=')[1])
                        acc_values = [float(acc_parts[i].split('=')[1] + '.' + acc_parts[i+1]) for i in range(1, len(acc_parts), 2)]
                        timestamp_gyro = float(gyro_parts[0].split('=')[1])
                        gyro_values = [float(gyro_parts[i].split('=')[1] + '.' + gyro_parts[i+1]) for i in range(1, len(gyro_parts), 2)]

                        # Combina los valores y los escribe en el archivo CSV
                        writer.writerow([timestamp_acc] + acc_values + [timestamp_gyro] + gyro_values)

                    except (IndexError, ValueError) as e:
                        print(f"Error processing line: {line}. Error: {e}")

    print("Documentos exportados exitosamente a archivos CSV separados.")

# Ejecutar la función
export_documents_to_csv()
