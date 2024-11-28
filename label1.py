import os
import pandas as pd

def add_label_to_csv_files(directory):
    """
    Añadir una columna 'label' a todos los archivos CSV en el directorio con un valor determinado,
    dependiendo del nombre del archivo.
    
    :param directory: Ruta del directorio que contiene los archivos CSV.
    """
    # Recorrer todos los archivos CSV en el directorio
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            print(f"Procesando archivo: {file_path}")
            
            # Leer el archivo CSV
            df = pd.read_csv(file_path)
            
            # Determinar el valor de la etiqueta según el nombre del archivo
            if filename.startswith("extension_cuello"):
                label_value = 1
            elif filename.startswith("flexion_cuello"):
                label_value = 0
            else:
                print(f"Nombre de archivo desconocido, omitiendo: {file_path}")
                continue
            
            # Añadir la columna 'label' con el valor especificado
            df['label'] = label_value
            
            # Guardar el archivo CSV actualizado, sobrescribiendo el original
            df.to_csv(file_path, index=False)
            print(f"Etiqueta {label_value} añadida y archivo guardado: {file_path}")

# Ejemplo de uso
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor"
add_label_to_csv_files(csv_directory)