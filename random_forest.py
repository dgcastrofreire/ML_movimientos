import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

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

    return df

def save_scaler(scaler, scaler_filename="scaler.pkl"):
    """
    Guardar el escalador utilizado para normalizar los datos.
    """
    joblib.dump(scaler, scaler_filename)
    print(f"Escalador guardado en {scaler_filename}")

def train_random_forest(df):
    """
    Entrenar un clasificador Random Forest usando los datos procesados.
    """
    # Normalizar usando MinMaxScaler y guardar el escalador
    scaler = MinMaxScaler()
    df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled', 'acc_magnitude_scaled', 'gyro_magnitude_scaled']] = scaler.fit_transform(
        df[['acc_x_smooth', 'acc_y_smooth', 'acc_z_smooth', 'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth', 'acc_magnitude', 'gyro_magnitude']]
    )

    # Guardar el escalador para usarlo posteriormente
    save_scaler(scaler, "scaler.pkl")

    # Separar características (features) y etiquetas
    features = df[['acc_x_scaled', 'acc_y_scaled', 'acc_z_scaled', 'gyro_x_scaled', 'gyro_y_scaled', 'gyro_z_scaled', 'acc_magnitude_scaled', 'gyro_magnitude_scaled']]
    labels = df['label']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Ajustar hiperparámetros con GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],  # Número de árboles en el bosque
        'max_depth': [10, 20, None],      # Profundidad máxima de los árboles
        'min_samples_split': [2, 5],      # Número mínimo de muestras requeridas para dividir un nodo
        'min_samples_leaf': [1, 2],       # Número mínimo de muestras en una hoja
        'class_weight': ['balanced']      # Balanceo de clases
    }

    rf_clf = RandomForestClassifier(random_state=42)

    # Validación cruzada y búsqueda de hiperparámetros óptimos
    grid_search = GridSearchCV(rf_clf, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Mostrar mejores parámetros
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

    # Entrenar el mejor modelo
    best_rf_clf = grid_search.best_estimator_
    best_rf_clf.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = best_rf_clf.predict(X_test)
    print("Reporte de clasificación para Random Forest:")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    return best_rf_clf

def save_model(rf_model, model_filename="random_forest_model.pkl"):
    """
    Guardar el modelo Random Forest en un archivo para su uso posterior.
    """
    joblib.dump(rf_model, model_filename)
    print(f"Modelo guardado en {model_filename}")

# Especifica el directorio donde están los archivos CSV
csv_directory = "C:/Users/Diego Castro/Documents/Uvigo/Uvigo/4º/TFG/MLsensor"

# Leer y combinar los datos desde los archivos CSV en un solo DataFrame
combined_df = pd.DataFrame()

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = read_data_from_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Entrenar y evaluar el modelo Random Forest usando todos los datos combinados
rf_model = train_random_forest(combined_df)

# Guardar el modelo entrenado
save_model(rf_model, "random_forest_model.pkl")
