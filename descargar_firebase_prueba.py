import csv

data = "timestamp_acc=1735486798649,acc_x=1,1914062500,acc_y=0,7799072266,acc_z=-0,7509765625;timestamp_gyro=1735486798654,gyro_x=28,3612804413,gyro_y=-99,5198211670,gyro_z=83,0945129395"

# Dividimos la cadena en las dos partes: acelerómetro y giroscopio
acc_part, gyro_part = data.split(';')

# Función para procesar una parte y convertirla en un diccionario
def parse_data(part):
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
                # Concatenar la parte adicional
                result[current_key] = f"{result[current_key]},{item}"
                # Convertir a número flotante
                result[current_key] = float(result[current_key].replace(',', '.'))
    return result

# Procesar ambas partes
acc_data = parse_data(acc_part)
gyro_data = parse_data(gyro_part)

# Combinar las dos partes en un solo diccionario
combined_data = {**acc_data, **gyro_data}

# Guardar los datos en un archivo CSV
with open('sensor_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Escribir encabezados
    writer.writerow(combined_data.keys())
    # Escribir valores
    writer.writerow(combined_data.values())

print("Datos guardados en 'sensor_data.csv'")
