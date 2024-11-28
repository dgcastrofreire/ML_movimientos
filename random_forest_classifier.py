import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
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
    window_size = 51  # Tamaño de ventana reducido para preservar detalles
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

def count_repetitions(df, column='acc_magnitude', height_threshold=0.5, distance=100):
    """
    Contar el número de repeticiones de un ejercicio usando la señal del acelerómetro.
    """
    # Detectar los picos en la señal de magnitud para identificar repeticiones
    peaks, _ = find_peaks(df[column], height=height_threshold, distance=distance)
    num_reps = len(peaks)
    print(f"Número de repeticiones detectadas: {num_reps}")
    return num_reps

def extract_features(df):
    """
    Extrae características estadísticas y temporales del DataFrame original para mejorar la precisión del modelo.
    """
    features = {}
    
    # Características del acelerómetro
    features['acc_x_mean'] = df['acc_x_smooth'].mean()
    features['acc_x_std'] = df['acc_x_smooth'].std()
    features['acc_y_mean'] = df['acc_y_smooth'].mean()
    features['acc_y_std'] = df['acc_y_smooth'].std()
    features['acc_z_mean'] = df['acc_z_smooth'].mean()
    features['acc_z_std'] = df['acc_z_smooth'].std()
    
    # Características del giroscopio
    features['gyro_x_mean'] = df['gyro_x_smooth'].mean()
    features['gyro_x_std'] = df['gyro_x_smooth'].std()
    features['gyro_y_mean'] = df['gyro_y_smooth'].mean()
    features['gyro_y_std'] = df['gyro_y_smooth'].std()
    features['gyro_z_mean'] = df['gyro_z_smooth'].mean()
    features['gyro_z_std'] = df['gyro_z_smooth'].std()
    
    # Características de la magnitud
    features['acc_magnitude_mean'] = df['acc_magnitude'].mean()
    features['acc_magnitude_std'] = df['acc_magnitude'].std()
    features['gyro_magnitude_mean'] = df['gyro_magnitude'].mean()
    features['gyro_magnitude_std'] = df['gyro_magnitude'].std()
    
    return pd.Series(features)

def create_segments_and_labels(df, window_size=100):
    """
    Crea ventanas de datos y sus etiquetas correspondientes.
    """
    segments = []
    labels = []

    for start in range(0, len(df) - window_size, window_size):
        end = start + window_size
        segment = df.iloc[start:end]
        
        # Crear las características de la ventana actual
        features = extract_features(segment)
        segments.append(features)
        
        # Etiqueta de la ventana (usamos la etiqueta más común en la ventana)
        label = segment['label'].mode()[0]
        labels.append(label)
    
    return pd.DataFrame(segments), labels

def save_scaler(scaler, scaler_filename="scaler_predict.pkl"):
    """
    Guardar el escalador utilizado para normalizar los datos.
    """
    joblib.dump(scaler, scaler_filename)
    print(f"Escalador guardado en {scaler_filename}")

def train_random_forest(X, y):
    """
    Entrenar un clasificador Random Forest usando los datos procesados.
    """
    # Normalizar usando MinMaxScaler y guardar el escalador
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar el escalador para usarlo posteriormente
    save_scaler(scaler, "scaler_predict.pkl")

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

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

# Contar el número de repeticiones detectadas utilizando detección directa de picos
count_repetitions(combined_df, column='acc_magnitude', height_threshold=0.5, distance=100)

# Crear segmentos de datos y entrenar el modelo con estos segmentos
X, y = create_segments_and_labels(combined_df)
rf_model = train_random_forest(X, y)

# Guardar el modelo entrenado
save_model(rf_model, "random_forest_model.pkl")