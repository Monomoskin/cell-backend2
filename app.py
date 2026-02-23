from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
from datetime import datetime
import os.path
import re # Necesario para la función get_image_urls_from_path
import json
from predict_new import process_pair_for_backend
import uuid  

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
    """Obtiene el historial de estimaciones para una célula específica, con URLs reales de todas las imágenes."""
    df = load_data_log()
    if df.empty:
        return jsonify([]), 200

    cell_data = df[df['Cell Name'] == cell_name].copy()
    
    if cell_data.empty:
        return jsonify({"message": f"No estimations found for cell: {cell_name}"}), 404

    cell_data['Upload Date'] = pd.to_datetime(cell_data['Upload Date'])
    cell_data = cell_data.sort_values(by='Upload Date', ascending=True)
    
    results = []
    
    for _, row in cell_data.iterrows():
        measurement_id = row['Measurement ID']
        date_str = str(row['Upload Date']).split(' ')[0]
        base_path = os.path.join(date_str, cell_name)
        
        # Ruta del JSON de detalle (donde están las imágenes reales)
        json_path = os.path.join(PREDICTED_FOLDER, date_str, cell_name, f"{measurement_id}_prediction.json")
        
        # URLs base
        base_pred_url = f"{IMAGE_API_ROUTE}/{base_path}"
        base_upload_url = f"{UPLOAD_API_ROUTE}/{base_path}"
        
        # URLs originales (siempre existen, basadas en nombres fijos)
        uploaded_top_url = f"{base_upload_url}/{cell_name}_TOP_uploaded.jpg"
        uploaded_side_url = f"{base_upload_url}/{cell_name}_SIDE_uploaded.jpg"
        
        # URLs predichas (por defecto null, se llenan si hay JSON)
        pred_top_clean = pred_top_text = pred_side_clean = pred_side_text = None
        
        # Intentar leer el JSON de detalle
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    pred_data = json.load(f)
                
                images = pred_data.get("images", {})
                
                # Construir URLs reales usando los nombres guardados en el JSON
                if 'top_clean' in images:
                    pred_top_clean = f"{base_pred_url}/{os.path.basename(images['top_clean'])}"
                if 'top_with_text' in images:
                    pred_top_text = f"{base_pred_url}/{os.path.basename(images['top_with_text'])}"
                if 'side_clean' in images:
                    pred_side_clean = f"{base_pred_url}/{os.path.basename(images['side_clean'])}"
                if 'side_with_text' in images:
                    pred_side_text = f"{base_pred_url}/{os.path.basename(images['side_with_text'])}"
            except Exception as e:
                print(f"Error leyendo JSON para {measurement_id}: {e}")
                # No rompemos el endpoint, solo dejamos las URLs predichas en null
        
        # Construir el registro completo
        record = {
            'Cell Name': row['Cell Name'],
            'Measurement ID': measurement_id,
            'Upload Date': row['Upload Date'].strftime("%Y-%m-%d %H:%M:%S"),
            'Estimated Volume (mL)': row['Estimated Volume (mL)'],
            # URLs predichas (ahora reales)
            'predicted_image_top_clean_url': pred_top_clean,
            'predicted_image_top_with_text_url': pred_top_text,
            'predicted_image_side_clean_url': pred_side_clean,
            'predicted_image_side_with_text_url': pred_side_text,
            # URLs originales
            'uploaded_image_top_url': uploaded_top_url,
            'uploaded_image_side_url': uploaded_side_url,
        }
        
        results.append(record)

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
    
    # Validación básica de campos requeridos
    if 'cell_name' not in request.form or 'image_top' not in request.files or 'image_side' not in request.files:
        return jsonify({"error": "Faltan datos requeridos (cell_name, image_top o image_side)."}), 400

    # Sanitizar cell_name para evitar problemas en paths/URLs
    import re
    cell_name_raw = request.form['cell_name'].strip()
    cell_name = re.sub(r'[^a-zA-Z0-9_-]', '_', cell_name_raw)
    if not cell_name or len(cell_name) > 100:
        return jsonify({"error": "Nombre de célula inválido o demasiado largo."}), 400

    file_top = request.files['image_top']
    file_side = request.files['image_side']

    # Validación simple de extensiones
    allowed_ext = {'.jpg', '.jpeg', '.png'}
    if not file_top.filename.lower().endswith(tuple(allowed_ext)) or \
       not file_side.filename.lower().endswith(tuple(allowed_ext)):
        return jsonify({"error": "Solo se permiten archivos .jpg, .jpeg o .png"}), 400

    # Fecha: por defecto ahora real, pero permite override para pruebas
    now = datetime.now()
    test_date_str = request.form.get('test_date')  # opcional: "2025-02-10" o "2025-02-10 14:30:00"

    if test_date_str:
        try:
            if ' ' in test_date_str:  # con hora
                now = datetime.strptime(test_date_str, "%Y-%m-%d %H:%M:%S")
            else:  # solo fecha → combina con hora actual
                base_date = datetime.strptime(test_date_str, "%Y-%m-%d")
                current_time = datetime.now()
                now = base_date.replace(
                    hour=current_time.hour,
                    minute=current_time.minute,
                    second=current_time.second,
                    microsecond=current_time.microsecond
                )
            print(f"[TEST MODE] Fecha manual activada: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        except ValueError as ve:
            print(f"[TEST MODE] Formato de test_date inválido '{test_date_str}': {ve} → usando fecha real")
            now = datetime.now()
        except Exception as e:
            print(f"[TEST MODE] Error al parsear test_date: {e} → usando fecha real")
            now = datetime.now()

    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y%m%d%H%M%S")

    # Generación de ID legible y único (formato que te gusta)
    unique_suffix = uuid.uuid4().hex[:6]  # 6 caracteres hexadecimales únicos
    measurement_id = f"{cell_name}-{timestamp}-{unique_suffix}"

    # Rutas para guardar (estructura: Date/Cell)
    day_upload_dir = os.path.join(UPLOAD_FOLDER, date_str)
    cell_upload_dir = os.path.join(day_upload_dir, cell_name)
    os.makedirs(cell_upload_dir, exist_ok=True)

    day_predicted_dir = os.path.join(PREDICTED_FOLDER, date_str)
    cell_predicted_dir = os.path.join(day_predicted_dir, cell_name)
    os.makedirs(cell_predicted_dir, exist_ok=True)

    # Nombres de archivo originales (usando cell_name para mantener consistencia con CSV)
    uploaded_filename_top = f"{cell_name}_TOP_uploaded.jpg"
    uploaded_filename_side = f"{cell_name}_SIDE_uploaded.jpg"

    path_top = os.path.join(cell_upload_dir, uploaded_filename_top)
    path_side = os.path.join(cell_upload_dir, uploaded_filename_side)

    # Guardar imágenes originales
    file_top.save(path_top)
    file_side.save(path_side)

    # Ejecutar Predicción (usamos measurement_id como sample_key para que los archivos internos sean únicos)
    try:
        prediction_result = process_pair_for_backend(
            top_image_path=path_top,
            side_image_path=path_side,
            sample_key=measurement_id,  # ← ID único y legible aquí
            output_dir=cell_predicted_dir
        )
    except Exception as e:
        print(f"Error fatal durante la predicción para {cell_name}: {e}")
        return jsonify({"error": f"Error interno en el modelo de predicción: {str(e)}"}), 500

    # Verificar si hubo error en la predicción
    if "error" in prediction_result:
        return jsonify({"error": prediction_result["error"]}), 400

    # Extraer valores principales
    estimated_volume = prediction_result.get("total_volume_ml", 0.0)
    height_mm = prediction_result.get("height_mm")

    # Rutas de imágenes predichas (priorizamos with_text si existe, sino clean)
    images = prediction_result.get("images", {})
    predicted_top = images.get("top_with_text") or images.get("top_clean")
    predicted_side = images.get("side_with_text") or images.get("side_clean")

    # Guardar JSON completo de la predicción
    json_path = os.path.join(cell_predicted_dir, f"{measurement_id}_prediction.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_result, f, indent=2, ensure_ascii=False)

    # Registrar en CSV (versión más completa)
    df = load_data_log()
    new_row = {
        'Cell Name': cell_name,
        'Measurement ID': measurement_id,  # ← ID único y legible guardado aquí
        'Upload Date': now.strftime("%Y-%m-%d %H:%M:%S"),
        'Estimated Volume (mL)': round(estimated_volume, 4),
        'Height (mm)': round(height_mm, 2) if height_mm is not None else None,
        'Num Cells Detected': len(prediction_result.get("cells", [])),
        'Image Location (Top)': path_top,
        'Image Location (Side)': path_side,
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

    # Construir URLs
    base_pred_url = f"{IMAGE_API_ROUTE}/{date_str}/{cell_name}"
    base_upload_url = f"{UPLOAD_API_ROUTE}/{date_str}/{cell_name}"

    response = {
        "status": prediction_result["status"],
        "measurement_id": measurement_id,  # ← ID único y legible devuelto al frontend
        "cell_name": cell_name,
        "estimated_volume": round(estimated_volume, 4),
        "height_mm": height_mm,
        "num_cells_detected": len(prediction_result.get("cells", [])),
        "processed_date": now.strftime("%Y-%m-%d %H:%M:%S"),  # fecha usada (útil para confirmar en pruebas)
        
        # Imágenes predichas
        "predicted_image_top_clean_url": f"{base_pred_url}/{os.path.basename(images.get('top_clean', ''))}" if 'top_clean' in images else None,
        "predicted_image_top_with_text_url": f"{base_pred_url}/{os.path.basename(images.get('top_with_text', ''))}" if 'top_with_text' in images else None,
        "predicted_image_side_clean_url": f"{base_pred_url}/{os.path.basename(images.get('side_clean', ''))}" if 'side_clean' in images else None,
        "predicted_image_side_with_text_url": f"{base_pred_url}/{os.path.basename(images.get('side_with_text', ''))}" if 'side_with_text' in images else None,
        
        # Imágenes originales
        "uploaded_image_top_url": f"{base_upload_url}/{uploaded_filename_top}",
        "uploaded_image_side_url": f"{base_upload_url}/{uploaded_filename_side}",
        
        # Resumen de células (opcional, pero muy útil)
        "cells_summary": prediction_result.get("cells", []),
    }

    return jsonify(response), 200


# SESSION


# Diccionario temporal para sesiones
sessions = {}

@app.route("/api/wechat-login/<session_id>", methods=["GET"])
def wechat_login_page(session_id):
    """
    Página que se abre desde el móvil. Usuario introduce su nombre.
    """
    return render_template("wechat_login.html", session_id=session_id)

@app.route("/api/wechat-login-submit", methods=["POST"])
def wechat_login_submit():
    """
    El formulario envía nombre y session_id
    """
    data = request.json
    session_id = data.get("session_id")
    username = data.get("username")
    
    if session_id and username:
        sessions[session_id] = username
        return jsonify({"status": "ok"}), 200
    return jsonify({"status": "error"}), 400

@app.route("/api/wechat-poll/<session_id>", methods=["GET"])
def wechat_poll(session_id):
    """
    Polling desde Electron para ver si usuario ya envió su nombre
    """
    username = sessions.get(session_id)
    if username:
        return jsonify({"username": username}), 200
    return jsonify({"username": None}), 200


# ==============================================================================
# 6. EJECUCIÓN DEL SERVIDOR
# ==============================================================================

if __name__ == '__main__':
    print(f"Data log saved to: {LOG_FILE}")
    print("Iniciando servidor Flask en http://localhost:5000...")
    app.run(debug=True, host='0.0.0.0', port=5000)