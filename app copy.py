from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
from datetime import datetime
import os.path
import re # Necesario para la función get_image_urls_from_path

# Importa la función de predicción.
try:
    from predict_new import predict_volume_and_save_images 
except ImportError:
    print("ERROR: No se pudo importar 'predict.py'. Usando función mock de seguridad.")
    def predict_volume_and_save_images(uploaded_top_path, uploaded_side_path, output_dir, cell_name):
        # Devuelve un volumen mock y rutas mock (predichas)
        predicted_top_path = os.path.join(output_dir, f"{cell_name}_TOP_predicted.jpg")
        predicted_side_path = os.path.join(output_dir, f"{cell_name}_SIDE_predicted.jpg")
        return 0.0, predicted_top_path, predicted_side_path


app = Flask(__name__)
# Configuración CORS: permite la comunicación con el frontend de React
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CARPETAS
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploaded_images')
PREDICTED_FOLDER = os.path.join(BASE_DIR, 'predicted_images')
# El archivo log está en la carpeta raíz del proyecto
LOG_FILE = os.path.join(os.path.dirname(BASE_DIR), 'data_log.csv') 

# Variables globales para las rutas de la API (usadas en get_image_urls_from_path)
IMAGE_API_ROUTE = "/api/predicted_images"
UPLOAD_API_ROUTE = "/api/uploaded_images"

# Asegurar que las carpetas existan
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

# ==============================================================================
# 2. FUNCIONES AUXILIARES
# ==============================================================================

def load_data_log():
    """Carga el archivo log (CSV) o crea un DataFrame vacío."""
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            return pd.read_csv(LOG_FILE)
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame(columns=['Cell Name', 'Measurement ID', 'Upload Date', 'Estimated Volume (mL)', 'Image Location (Top)', 'Image Location (Side)'])

def get_image_urls_from_path(row):
    """
    Convierte los nombres de archivo en el log CSV a URLs accesibles por el frontend.
    Retorna 4 URLs: TOP_predicted, SIDE_predicted, TOP_uploaded, SIDE_uploaded.
    """
    try:
        date_str = str(row['Upload Date']).split(' ')[0]
        cell_name = row['Cell Name'] 

        top_predicted_filename = f"{cell_name}_TOP_predicted.jpg"
        side_predicted_filename = f"{cell_name}_SIDE_predicted.jpg"
        top_uploaded_filename = f"{cell_name}_TOP_uploaded.jpg" 
        side_uploaded_filename = f"{cell_name}_SIDE_uploaded.jpg"
        
        base_path = os.path.join(date_str, cell_name)
        
        def build_url(api_route, path, filename):
            full_path = os.path.join(api_route, path, filename)
            # Garantiza el uso de barras diagonales para URLs y evita dobles barras
            return re.sub(r'//+', '/', full_path.replace(os.path.sep, '/'))

        url_predicted_top = build_url(IMAGE_API_ROUTE, base_path, top_predicted_filename)
        url_predicted_side = build_url(IMAGE_API_ROUTE, base_path, side_predicted_filename)
        
        url_uploaded_top = build_url(UPLOAD_API_ROUTE, base_path, top_uploaded_filename)
        url_uploaded_side = build_url(UPLOAD_API_ROUTE, base_path, side_uploaded_filename)
        
        # Retorna las 4 URLs en una serie de Pandas
        return pd.Series([
            url_predicted_top,
            url_predicted_side,
            url_uploaded_top,
            url_uploaded_side
        ])
        
    except KeyError:
        return pd.Series([None, None, None, None])
    
    except Exception as e:
        measurement_id = row.get('Measurement ID', 'N/A')
        print(f"Error general al reconstruir URLs para la fila {measurement_id}: {e}")
        return pd.Series([None, None, None, None])

# ==============================================================================
# 3. ENDPOINTS DE LECTURA DE DATOS
# ==============================================================================

@app.route('/api/estimations', methods=['GET'])
def list_estimations():
    """Lista todas las estimaciones realizadas, incluyendo las 4 URLs de imagen."""
    df = load_data_log()
    if df.empty:
        return jsonify([]), 200

    df_output = df[['Cell Name', 'Measurement ID', 'Upload Date', 'Estimated Volume (mL)']].copy()
    
    # Capturar las 4 columnas de URL
    df_output[['predicted_image_top_url', 'predicted_image_side_url', 
               'uploaded_image_top_url', 'uploaded_image_side_url']] = df.apply(
        get_image_urls_from_path, 
        axis=1, 
        result_type='expand'
    )

    df_output['Upload Date'] = pd.to_datetime(df_output['Upload Date'])
    results = df_output.sort_values(by='Upload Date', ascending=False).to_dict(orient='records')
    
    return jsonify(results), 200

@app.route('/api/estimations/summary', methods=['GET'])
def get_estimations_summary():
    """Devuelve un resumen de la última estimación y el conteo por célula."""
    df = load_data_log()
    if df.empty:
        return jsonify([]), 200

    df['Upload Date'] = pd.to_datetime(df['Upload Date'])
    
    last_estimation = df.sort_values(by='Upload Date', ascending=False).drop_duplicates(subset=['Cell Name'], keep='first')
    count_estimations = df.groupby('Cell Name')['Measurement ID'].count().reset_index().rename(columns={'Measurement ID': 'Estimated Count'})

    summary = last_estimation[['Cell Name', 'Estimated Volume (mL)', 'Upload Date', 'Measurement ID']].copy()
    
    summary.rename(columns={
        'Estimated Volume (mL)': 'Last Estimated Volume (mL)',
        'Upload Date': 'Last Estimation Date',
        'Measurement ID': 'Last Measurement ID'
    }, inplace=True)

    summary = pd.merge(summary, count_estimations, on='Cell Name', how='left')

    # Añadir las 4 URLs de imagen
    summary[['predicted_image_top_url', 'predicted_image_side_url','uploaded_image_top_url', 'uploaded_image_side_url']] = summary.apply(get_image_urls_from_path, axis=1, result_type='expand')

    summary['Last Estimation Date'] = summary['Last Estimation Date'].dt.strftime("%Y-%m-%d %H:%M:%S")

    results = summary.to_dict(orient='records')
    
    return jsonify(results), 200

@app.route('/api/estimations/latest', methods=['GET'])
def list_latest_estimations():
    """Lista las últimas 10 estimaciones realizadas."""
    df = load_data_log()
    if df.empty:
        return jsonify([]), 200

    df_latest = df.sort_values(by='Upload Date', ascending=False).head(10)
    
    df_output = df_latest[['Cell Name', 'Measurement ID', 'Upload Date', 'Estimated Volume (mL)']].copy()
    
    # Capturar las 4 columnas de URL
    df_output[['predicted_image_top_url', 'predicted_image_side_url', 
               'uploaded_image_top_url', 'uploaded_image_side_url']] = df_latest.apply(
        get_image_urls_from_path, 
        axis=1, 
        result_type='expand'
    )

    results = df_output.to_dict(orient='records')
    
    return jsonify(results), 200

@app.route('/api/estimations/<string:cell_name>', methods=['GET'])
def get_cell_history(cell_name):
    """Obtiene el historial de estimaciones para una célula específica."""
    df = load_data_log()
    if df.empty:
        return jsonify([]), 200

    cell_data = df[df['Cell Name'] == cell_name].copy()
    
    if cell_data.empty:
        return jsonify({"message": f"No estimations found for cell: {cell_name}"}), 404

    cell_data['Upload Date'] = pd.to_datetime(cell_data['Upload Date'])
    cell_data = cell_data.sort_values(by='Upload Date', ascending=True)
    
    df_output = cell_data[['Cell Name', 'Measurement ID', 'Upload Date', 'Estimated Volume (mL)']].copy()
    
    # Capturar las 4 URLs
    df_output[['predicted_image_top_url', 'predicted_image_side_url', 
               'uploaded_image_top_url', 'uploaded_image_side_url']] = cell_data.apply(
        get_image_urls_from_path, 
        axis=1, 
        result_type='expand'
    )

    results = df_output.to_dict(orient='records')

    return jsonify(results), 200


# ==============================================================================
# 4. ENDPOINTS PARA SERVIR IMÁGENES
# ==============================================================================

@app.route('/api/predicted_images/<path:filename>')
def serve_predicted_image(filename):
    """Sirve archivos estáticos desde la carpeta de imágenes predichas."""
    return send_from_directory(PREDICTED_FOLDER, filename)

@app.route('/api/uploaded_images/<path:filename>')
def serve_uploaded_image(filename):
    """Sirve archivos estáticos desde la carpeta de imágenes subidas (originales)."""
    return send_from_directory(UPLOAD_FOLDER, filename)


# ==============================================================================
# 5. ENDPOINT PRINCIPAL DE PROCESAMIENTO
# ==============================================================================

@app.route('/api/analyze', methods=['POST'])
def analyze_cell():
    """Endpoint principal para recibir imágenes, procesar y registrar datos."""
    
    if 'cell_name' not in request.form or 'image_top' not in request.files or 'image_side' not in request.files:
        return jsonify({"error": "Missing required data (cell_name, image_top, or image_side)."}), 400

    cell_name = request.form['cell_name']
    file_top = request.files['image_top']
    file_side = request.files['image_side']
    
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y%m%d%H%M%S")
    measurement_id = f"{cell_name}-{timestamp}"
    
    # Rutas para guardar (Estructura: Date/Cell)
    day_upload_dir = os.path.join(UPLOAD_FOLDER, date_str)
    cell_upload_dir = os.path.join(day_upload_dir, cell_name) 
    os.makedirs(cell_upload_dir, exist_ok=True) 

    day_predicted_dir = os.path.join(PREDICTED_FOLDER, date_str)
    cell_predicted_dir = os.path.join(day_predicted_dir, cell_name)
    os.makedirs(cell_predicted_dir, exist_ok=True)

    # Nombres de archivo que esperan las utilidades de URL
    uploaded_filename_top = f"{cell_name}_TOP_uploaded.jpg" 
    uploaded_filename_side = f"{cell_name}_SIDE_uploaded.jpg"

    path_top = os.path.join(cell_upload_dir, uploaded_filename_top)
    path_side = os.path.join(cell_upload_dir, uploaded_filename_side)
    
    # Guardar las imágenes originales
    file_top.save(path_top)
    file_side.save(path_side)

    # Ejecutar Predicción
    try:
        estimated_volume, predicted_top_path, predicted_side_path = predict_volume_and_save_images(
            path_top, 
            path_side, 
            cell_predicted_dir, 
            cell_name
        )
    except Exception as e:
        print(f"Error fatal durante la predicción para {cell_name}: {e}")
        return jsonify({"error": f"Internal prediction model error: {str(e)}"}), 500

    # Registrar en CSV
    df = load_data_log()
    new_row = {
        'Cell Name': cell_name,
        'Measurement ID': measurement_id,
        'Upload Date': now.strftime("%Y-%m-%d %H:%M:%S"),
        'Estimated Volume (mL)': estimated_volume,
        'Image Location (Top)': path_top, 
        'Image Location (Side)': path_side, 
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

    # Devolver respuesta con las 4 URLs (CRÍTICO para el frontend)
    return jsonify({
        "status": "success",
        "measurement_id": measurement_id,
        "cell_name": cell_name,
        "estimated_volume": estimated_volume,
        
        "predicted_image_top_url": f"{IMAGE_API_ROUTE}/{date_str}/{cell_name}/{os.path.basename(predicted_top_path)}",
        "predicted_image_side_url": f"{IMAGE_API_ROUTE}/{date_str}/{cell_name}/{os.path.basename(predicted_side_path)}",
        
        "uploaded_image_top_url": f"{UPLOAD_API_ROUTE}/{date_str}/{cell_name}/{uploaded_filename_top}",
        "uploaded_image_side_url": f"{UPLOAD_API_ROUTE}/{date_str}/{cell_name}/{uploaded_filename_side}",
    }), 200

# ==============================================================================
# 6. EJECUCIÓN DEL SERVIDOR
# ==============================================================================

if __name__ == '__main__':
    print(f"Data log saved to: {LOG_FILE}")
    print("Iniciando servidor Flask en http://localhost:5000...")
    app.run(debug=True, port=5000)