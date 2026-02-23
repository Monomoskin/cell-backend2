# predict.py

import cv2
import numpy as np
import os
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.model_zoo import get_config_file
import json
import pandas as pd
from detectron2.structures import Instances 
import csv

# --- CONSTANTES DE CALIBRACIÓN ---
FRASCO_DIAMETER_MM = 90.0
FRASCO_HEIGHT_MM = 18.0

# --- Variables Globales para Carga Única del Modelo ---
PREDICTOR = None
METADATA = None
CATEGORY_NAMES = None

# --- 1. Helper function (Sin cambios) ---
def find_highest_score_instance(instances, class_id):
    """
    Returns the instance with the highest score for a given class ID.
    """
    if len(instances) == 0:
        return None
    scores = instances.scores
    classes = instances.pred_classes
    class_indices = (classes == class_id).nonzero(as_tuple=True)[0]
    if len(class_indices) == 0:
        return None
    best_idx = class_indices[torch.argmax(scores[class_indices])]
    
    best_instance = Instances(instances.image_size)
    best_instance.pred_masks = instances.pred_masks[best_idx:best_idx+1]
    best_instance.pred_classes = instances.pred_classes[best_idx:best_idx+1]
    best_instance.scores = instances.scores[best_idx:best_idx+1]
    
    return best_instance

# --------------------------------------------------------------------------
# --- FUNCIÓN DE CARGA ÚNICA (NUEVO) ---
# --------------------------------------------------------------------------
def _setup_predictor():
    """Inicializa y carga el modelo solo una vez."""
    global PREDICTOR, METADATA, CATEGORY_NAMES
    
    # Si ya está cargado, no hacemos nada.
    if PREDICTOR is not None:
        return
        
    print("🚀 Cargando modelo Detectron2...")

    # Rutas relativas a la ubicación de predict.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(current_dir) 
      # Asume: /volume_estimator/backend/output_train/model_final.pth
    model_path = os.path.join(current_dir, "final_model", "model_final.pth") 
    
    # Ahora apunta a: /volume_estimator/annotations/coco_annotations.json
    json_path = os.path.join(base_path, "annotations", "coco_annotations.json")
    
    # Ahora apunta a: /volume_estimator/images
    image_dir = os.path.join(base_path, "images") 
    
    dataset_name = "celulas_frascos_flask" 

    # --- Dataset and Metadata Setup ---
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    CATEGORY_NAMES = [cat['name'] for cat in coco_data['categories']]
    try:
        register_coco_instances(dataset_name, {}, json_path, image_dir)
    except AssertionError:
        pass
        
    MetadataCatalog.get(dataset_name).thing_classes = CATEGORY_NAMES
    METADATA = MetadataCatalog.get(dataset_name)
    
    # --- Predictor Configuration ---
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORY_NAMES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.70 
    cfg.MODEL.DEVICE = "cpu"
    PREDICTOR = DefaultPredictor(cfg)
    
    print("✅ Modelo Detectron2 cargado exitosamente.")

# --------------------------------------------------------------------------
# --- FUNCIÓN CENTRAL DE PREDICCIÓN PARA FLASK (NUEVO) ---
# --------------------------------------------------------------------------

def predict_volume_and_save_images(uploaded_top_path, uploaded_side_path, output_dir, cell_name):
    """
    Función que envuelve la lógica principal para ser llamada desde app.py.

    Args:
        uploaded_top_path (str): Ruta temporal de la imagen TOP subida.
        uploaded_side_path (str): Ruta temporal de la imagen SIDE subida.
        output_dir (str): Directorio donde guardar las imágenes y resultados predichos.
        cell_name (str): Nombre de la célula/muestra (usado como sample_key).

    Returns:
        tuple: (estimated_volume, predicted_top_path, predicted_side_path)
    """
    
    # 1. Asegurar que el modelo esté cargado
    _setup_predictor()
    
    predictor = PREDICTOR
    metadata = METADATA
    category_names = CATEGORY_NAMES
    
    # Crear la carpeta de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # --- IDs de Clases ---
    try:
        frasco_id = category_names.index("container")
        cell_id = category_names.index("cell") 
        cell_profile_id = category_names.index("cell_profile")
    except ValueError:
        cell_profile_id = cell_id 
    
    # -------------------------------------------------------------
    # 2. PROCESAMIENTO VISTA SIDE (Altura Z y Calibración Z)
    # -------------------------------------------------------------
    im_side = cv2.imread(uploaded_side_path)
    if im_side is None:
        return 0.0, "Error de lectura SIDE", "Error de lectura SIDE"

    outputs_side = predictor(im_side)
    instances_side = outputs_side["instances"].to("cpu")
    
    frasco_side_instance = find_highest_score_instance(instances_side, frasco_id)
    if frasco_side_instance is None:
        return 0.0, "Error de detección (Side)", "Error de detección (Side)"
        
    frasco_side_mask = frasco_side_instance.pred_masks[0].cpu().numpy().astype(bool)
    y_coords_side, _ = np.where(frasco_side_mask)
    height_pixels_frasco = y_coords_side.max() - y_coords_side.min()
    if height_pixels_frasco == 0:
        return 0.0, "Error de calibración Z (altura 0)", "Error de calibración Z (altura 0)"
    factor_z_mm_per_pixel = FRASCO_HEIGHT_MM / height_pixels_frasco
    
    cell_profile_instance = find_highest_score_instance(instances_side, cell_profile_id)
    if cell_profile_instance is None:
        return 0.0, "Error de detección (Side)", "Error de detección (Side)"
        
    cell_profile_mask = cell_profile_instance.pred_masks[0].cpu().numpy().astype(bool)
    y_coords_cell_profile, _ = np.where(cell_profile_mask)
    height_pixels_cell = y_coords_cell_profile.max() - y_coords_cell_profile.min()
    height_real_mm = height_pixels_cell * factor_z_mm_per_pixel


    # -------------------------------------------------------------
    # 3. PROCESAMIENTO VISTA TOP (Área XY y Calibración XY)
    # -------------------------------------------------------------
    im_top = cv2.imread(uploaded_top_path)
    if im_top is None:
        return 0.0, "Error de lectura TOP", "Error de lectura TOP"
        
    outputs_top = predictor(im_top)
    instances_top = outputs_top["instances"].to("cpu")
    
    frasco_top_instance = find_highest_score_instance(instances_top, frasco_id)
    if frasco_top_instance is None:
        return 0.0, "Error de detección (Top)", "Error de detección (Top)"
        
    frasco_top_mask = frasco_top_instance.pred_masks[0].cpu().numpy().astype(bool)
    y_coords_top, x_coords_top = np.where(frasco_top_mask)
    width_pixels_frasco = x_coords_top.max() - x_coords_top.min()
    if width_pixels_frasco == 0:
        return 0.0, "Error de calibración XY (ancho 0)", "Error de calibración XY (ancho 0)"
        
    pixels_per_mm = width_pixels_frasco / FRASCO_DIAMETER_MM
    pixels_to_mm2 = 1 / (pixels_per_mm ** 2)

    # -------------------------------------------------------------
    # 4. CÁLCULO FINAL DEL VOLUMEN
    # -------------------------------------------------------------
    
    best_cell_instance = find_highest_score_instance(instances_top, cell_id)
    if best_cell_instance is None:
        return 0.0, "Célula no detectada", "Célula no detectada"
        
    cell_mask = best_cell_instance.pred_masks[0].numpy().astype(bool)
            
    area_mm2 = np.sum(cell_mask) * pixels_to_mm2
    volumen_ml = (area_mm2 * height_real_mm) / 1000 # mm³ a mL
    
    y_c, x_c = np.where(cell_mask)
    center_x, center_y = int(np.mean(x_c)), int(np.mean(y_c))
    
    # -------------------------------------------------------------
    # 5. VISUALIZACIÓN y GUARDADO DE IMÁGENES
    # -------------------------------------------------------------
    
    # === A. VISUALIZACIÓN Y GUARDADO DE IMAGEN TOP ===
    output_image_top_path = os.path.join(output_dir, f"{cell_name}_TOP_predicted.jpg")
    v_top = Visualizer(im_top[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    out_top = v_top.draw_instance_predictions(instances_top) 
    final_image_top = cv2.cvtColor(out_top.get_image(), cv2.COLOR_RGB2BGR)
    
    text = f"Vol: {volumen_ml:.3f} mL | Area: {area_mm2:.1f} mm2 | Alt: {height_real_mm:.1f} mm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    
    # Dibuja el texto en la imagen TOP
    cv2.putText(final_image_top, text, (center_x - 50, center_y + 10), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(final_image_top, text, (center_x - 50, center_y + 10), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_image_top_path, final_image_top)

    # === B. VISUALIZACIÓN Y GUARDADO DE IMAGEN SIDE ===
    output_image_side_path = os.path.join(output_dir, f"{cell_name}_SIDE_predicted.jpg")
    v_side = Visualizer(im_side[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    out_side = v_side.draw_instance_predictions(instances_side)
    final_image_side = cv2.cvtColor(out_side.get_image(), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_side_path, final_image_side)

    # -------------------------------------------------------------
    # 6. RETORNO DE RESULTADOS
    # -------------------------------------------------------------
    
    # Devuelve el volumen y las rutas de guardado
    return volumen_ml, output_image_top_path, output_image_side_path

# --------------------------------------------------------------------------
# --- Lógica de la Función Principal (Deshabilitada/Comentada) ---
# --------------------------------------------------------------------------

# Tu función process_sample_pair original y main() ya no se usan directamente,
# ya que Flask ahora llama a predict_volume_and_save_images con un solo par de imágenes.

# Las funciones main y consolidate_results de tu script original pueden ser 
# eliminadas o mantenidas comentadas, ya que no son necesarias para el backend Flask.
# Si quieres mantenerlas para uso offline, solo asegúrate de no llamarlas
# al importar el módulo.

# Llama a la configuración inicial para cargar el modelo cuando app.py lo importa
try:
    _setup_predictor()
except Exception as e:
    # Esto te permitirá ver errores de carga del modelo en la consola de Flask
    print(f"ERROR: No se pudo cargar el modelo Detectron2 al importar predict.py. Detalles: {e}")