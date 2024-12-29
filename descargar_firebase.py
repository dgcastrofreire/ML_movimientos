import csv
import firebase_admin
from firebase_admin import credentials, firestore
import unicodedata

# Initialize the Firestore connection
cred = credentials.Certificate("C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/DatosSensor/tfgdiego-40475-firebase-adminsdk-nmb6x-3d00b98ff7.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def remove_accents(input_str):
    # Normalize the string by removing accents
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def parse_data(line):
    """Parse a line of data and return a dictionary with the parsed values."""
    try:
        acc_part, gyro_part = line.split(';')

        def process_part(part):
            result = {}
            key_values = part.split(',')
            current_key = None
            for item in key_values:
                if '=' in item:
                    key, value = item.split('=')
                    current_key = key
                    result[key] = value
                else:
                    if current_key:
                        result[current_key] = f"{result[current_key]},{item}"
                        result[current_key] = float(result[current_key].replace(',', '.'))
            return result

        acc_data = process_part(acc_part)
        gyro_data = process_part(gyro_part)

        return {**acc_data, **gyro_data}
    except Exception as e:
        print(f"Error parsing line: {line}. Error: {e}")
        return None

def export_documents_to_csv():
    # Reference the Firestore collection
    users_ref = db.collection("sensor_data")
    docs = users_ref.stream()

    for doc in docs:
        doc_id = doc.id  # Get the document ID

        # Normalize the document ID for file naming
        doc_id_normalized = remove_accents(doc_id)

        content = doc.to_dict().get('content', '')
        lines = content.strip().split('\n')

        # Create a CSV file for each document
        with open(f'{doc_id_normalized}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write headers to the CSV file
            writer.writerow(['timestamp_acc', 'acc_x', 'acc_y', 'acc_z', 'timestamp_gyro', 'gyro_x', 'gyro_y', 'gyro_z'])

            for line in lines:
                parsed_data = parse_data(line)
                if parsed_data:
                    # Write the parsed data to the CSV file
                    writer.writerow([
                        parsed_data.get('timestamp_acc', ''),
                        parsed_data.get('acc_x', ''),
                        parsed_data.get('acc_y', ''),
                        parsed_data.get('acc_z', ''),
                        parsed_data.get('timestamp_gyro', ''),
                        parsed_data.get('gyro_x', ''),
                        parsed_data.get('gyro_y', ''),
                        parsed_data.get('gyro_z', '')
                    ])

    print("Documents exported successfully to separate CSV files.")

# Run the function
export_documents_to_csv()
