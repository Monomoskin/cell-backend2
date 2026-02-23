# predict_pipeline.py
import os
import cv2
import numpy as np
import torch
import json
import pandas as pd
from datetime import datetime
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.model_zoo import get_config_file
from detectron2.structures import Instances


# Importar tu custom ROI Heads (ajusta la ruta según tu proyecto)
from roi_heads import CallusROIHeads   # <--- asegúrate que este archivo exista y esté en PYTHONPATH

# Constantes de calibración (puedes mover a config más adelante)
FRASCO_DIAMETER_MM = 90.0
FRASCO_HEIGHT_MM = 12.0

# Variables globales para singleton
PREDICTOR = None
METADATA = None
CATEGORY_NAMES = None


def setup_predictor():
    global PREDICTOR, METADATA, CATEGORY_NAMES
    
    if PREDICTOR is not None:
        return
    
    print("[Backend] Cargando modelo Detectron2 (una sola vez)...")
    
    # Rutas - ajústalas según tu estructura real
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "output_train_attr", "model_final.pth")
    json_path = os.path.join(current_dir, "..", "annotations", "coco_annotations_multiattr.json")
    image_dir = os.path.join(current_dir, "..", "images")  # solo para register
    
    dataset_name = "celulas_frascos_attr"
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    CATEGORY_NAMES = [cat['name'] for cat in coco_data['categories']]
    
    try:
        register_coco_instances(dataset_name, {}, json_path, image_dir)
    except AssertionError:
        pass
    
    MetadataCatalog.get(dataset_name).thing_classes = CATEGORY_NAMES
    metadata = MetadataCatalog.get(dataset_name)
    
    # Atributos (deben coincidir con entrenamiento)
    metadata.species_classes = ["moso", "other"]
    metadata.quality_classes = ["poor", "medium", "good"]
    metadata.stage_classes = ["non_embryogenic", "embryogenic"]
    
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NAME = "CallusROIHeads"
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORY_NAMES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.30
    cfg.MODEL.DEVICE = "cpu"   # cambia a "cuda" si tienes GPU
    
    PREDICTOR = DefaultPredictor(cfg)
    METADATA = metadata
    
    print("[Backend] Modelo cargado correctamente.")


def find_highest_score_instance(instances, class_id):
    # (copiado tal cual de tu primer código - sin cambios)
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


def process_pair_for_backend(
    top_image_path: str,
    side_image_path: str = None,
    sample_key: str = None,
    output_dir: str = None
) -> dict:
    """
    Procesa un par TOP (+ opcional SIDE) y retorna resultados estructurados para API.
    Guarda imágenes y CSV/JSON en output_dir.
    """
    if sample_key is None:
        sample_key = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if output_dir is None:
        output_dir = os.path.join("output_backend", sample_key)
    os.makedirs(output_dir, exist_ok=True)
    
    setup_predictor()  # asegura que el modelo esté listo
    predictor = PREDICTOR
    metadata = METADATA
    category_names = CATEGORY_NAMES
    
    # IDs de clases (como en tu primer código)
    frasco_top_id = category_names.index("container_top")
    frasco_side_id = category_names.index("container_side")
    top_classes_of_interest = [category_names.index(n) for n in ["callus", "potato"] if n in category_names]
    cell_profile_id = category_names.index("cell_profile") if "cell_profile" in category_names else None
    defective_region_id = category_names.index("defective_region") if "defective_region" in category_names else None
    
    predicted_attributes = {}
    height_real_mm = None
    
    # ────────────────────────────────────────────────
    # PROCESO SIDE → altura + visualización
    # ────────────────────────────────────────────────
    has_side = side_image_path is not None and os.path.exists(side_image_path)
    height_real_mm = None
    cell_profile_center = None  # Guardaremos el centro de cell_profile para colocar texto de altura

    if has_side:
        im_side = cv2.imread(side_image_path)
        if im_side is not None:
            inst_side = predictor(im_side)["instances"].to("cpu")
            frasco_side = find_highest_score_instance(inst_side, frasco_side_id)

            if frasco_side is not None:
                m_f = frasco_side.pred_masks[0].numpy().astype(bool)
                ys = np.where(m_f)[0]
                if len(ys) > 10:
                    factor_z = FRASCO_HEIGHT_MM / (ys.max() - ys.min() + 1e-6)

                    if cell_profile_id is not None:
                        cell_profile = find_highest_score_instance(inst_side, cell_profile_id)
                        if cell_profile is not None:
                            m_cp = cell_profile.pred_masks[0].numpy().astype(bool)
                            ys_cp, xs_cp = np.where(m_cp)
                            if len(ys_cp) > 10:
                                height_real_mm = (ys_cp.max() - ys_cp.min()) * factor_z
                                # Calcular centro de la máscara de cell_profile
                                cx = int(np.mean(xs_cp))
                                cy = int(np.mean(ys_cp))
                                cell_profile_center = (cx, cy)

    # ────────────────────────────────────────────────
    # Guardar visualización SIDE (si existe)
    # ────────────────────────────────────────────────
    side_clean_path = None
    side_with_text_path = None

    if has_side and im_side is not None and 'inst_side' in locals():
        # 1. Versión limpia (solo máscaras)
        side_clean_path = os.path.join(output_dir, f"{sample_key}_SIDE_clean.jpg")
        v_side_clean = Visualizer(im_side[:, :, ::-1], metadata, instance_mode=ColorMode.SEGMENTATION)
        for i in range(len(inst_side)):
            mask = inst_side.pred_masks[i].numpy()
            v_side_clean.draw_binary_mask(mask, alpha=0.35)
        cv2.imwrite(side_clean_path, v_side_clean.output.get_image()[:, :, ::-1])

        # 2. Versión con texto informativo
        side_with_text_path = os.path.join(output_dir, f"{sample_key}_SIDE_with_text.jpg")
        v_side_text = Visualizer(im_side[:, :, ::-1], metadata, instance_mode=ColorMode.SEGMENTATION)

        # Dibujar todas las máscaras primero
        for i in range(len(inst_side)):
            mask = inst_side.pred_masks[i].numpy()
            v_side_text.draw_binary_mask(mask, alpha=0.35)

        # Texto principal: altura (en inglés)
        height_text_lines = []
        if height_real_mm is not None:
            height_text_lines.append(f"Estimated Height: {height_real_mm:.2f} mm")
        else:
            height_text_lines.append("Height: not calculated")

        if 'factor_z' in locals():
            height_text_lines.append(f"Factor Z: {factor_z:.4f} mm/px")

        # Decidir posición del texto de altura
        if cell_profile_center is not None:
            # Centrado arriba del centro de cell_profile
            text_pos = (cell_profile_center[0], cell_profile_center[1] - 140)  # -60 píxeles arriba para que quede visible
            alignment = "center"
        else:
            # Fallback: posición fija arriba-izquierda
            text_pos = (50, 80)
            alignment = "left"

        v_side_text.draw_text(
            "\n".join(height_text_lines),
            text_pos,
            font_size=22,
            color="cyan",
            horizontal_alignment=alignment
        )

        # Etiqueta "CELL PROFILE" sobre la máscara (si existe)
        if cell_profile_id is not None:
            cell_profile_inst = find_highest_score_instance(inst_side, cell_profile_id)
            if cell_profile_inst is not None:
                m_cp = cell_profile_inst.pred_masks[0].numpy().astype(bool)
                ys_cp, xs_cp = np.where(m_cp)
                if len(ys_cp) > 0:
                    cx = int(np.mean(xs_cp))
                    cy = int(np.mean(ys_cp))
                    v_side_text.draw_text(
                        "CELL PROFILE",
                        (cx, cy - 50),  # arriba del centro
                        font_size=18,
                        color="yellow",
                        horizontal_alignment="center"
                    )

        cv2.imwrite(side_with_text_path, v_side_text.output.get_image()[:, :, ::-1])

   
    # ────────────────────────────────────────────────
    # PROCESO TOP → área + atributos + defectos
    # ────────────────────────────────────────────────
    im_top = cv2.imread(top_image_path)
    if im_top is None:
        return {"error": "No se pudo leer la imagen TOP"}
    
    outputs = predictor(im_top)
    inst_top = outputs["instances"].to("cpu")
    
    # Filtrado básico de score (como en tu código)
    if defective_region_id is not None:
        keep = torch.ones(len(inst_top), dtype=torch.bool)
        for i in range(len(inst_top)):
            cls = inst_top.pred_classes[i].item()
            score = inst_top.scores[i].item()
            if cls == defective_region_id:
                keep[i] = score >= 0.1
            else:
                keep[i] = score >= 0.7
        inst_top = inst_top[keep]
    
    frasco_top = find_highest_score_instance(inst_top, frasco_top_id)
    if frasco_top is None:
        return {"error": "No se detectó el contenedor en TOP"}
    
    m_f = frasco_top.pred_masks[0].numpy().astype(bool)
    ys_f, xs_f = np.where(m_f)
    if len(xs_f) < 10:
        return {"error": "Máscara de frasco inválida en TOP"}
    
    px_per_mm = (xs_f.max() - xs_f.min()) / FRASCO_DIAMETER_MM
    px_to_mm2 = 1 / (px_per_mm ** 2)
    
    volume_calculable = has_side and height_real_mm is not None and height_real_mm > 0
    
    cells_results = []
    
    for i in range(len(inst_top)):
        cid = inst_top.pred_classes[i].item()
        if cid not in top_classes_of_interest:
            continue
        cell_mask = inst_top.pred_masks[i].numpy().astype(bool)
        if np.sum(cell_mask & m_f) / np.sum(cell_mask) < 0.9:
            continue
        area_mm2 = np.sum(cell_mask) * px_to_mm2
        if area_mm2 <= 0:
            continue
        
        ys, xs = np.where(cell_mask)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        
        volume_ml = (area_mm2 * height_real_mm) / 1000 if volume_calculable else None
        
        # Lógica de defectos (copiada/adaptada)
        defective_percent = 0.0
        defective_area_mm2 = 0.0
        if defective_region_id is not None:
            defective_mask_total = np.zeros_like(cell_mask, dtype=bool)
            for j in range(len(inst_top)):
                if inst_top.pred_classes[j] != defective_region_id:
                    continue
                defect_mask = inst_top.pred_masks[j].numpy().astype(bool)
                overlap = np.sum(defect_mask & cell_mask) / np.sum(defect_mask) if np.sum(defect_mask) > 0 else 0
                if overlap > 0.8:
                    defective_mask_total |= defect_mask
            if defective_mask_total.sum() > 0:
                defective_area_mm2 = np.sum(defective_mask_total) * px_to_mm2
                defective_percent = (defective_area_mm2 / area_mm2) * 100
        
        final_quality = max(0.0, min(100.0, 100.0 - defective_percent))
        
        # Atributos extras (species, stage, quality)
        species = stage = quality_from_model = None
        cls_name = category_names[cid]
        if cls_name == "callus":
            if hasattr(inst_top, "species") and i < len(inst_top.species):
                species = metadata.species_classes[inst_top.species[i].item()]
            if hasattr(inst_top, "stage") and i < len(inst_top.stage):
                stage = metadata.stage_classes[inst_top.stage[i].item()]
            if hasattr(inst_top, "quality") and i < len(inst_top.quality):
                quality_from_model = metadata.quality_classes[inst_top.quality[i].item()]
        
        cell_key = f"{sample_key}_cell{i+1}"
        predicted_attributes[cell_key] = {
            "class": cls_name,
            "species": species,
            "stage": stage,
            "quality_from_model": quality_from_model,
            "score": float(inst_top.scores[i]),
            "volume_ml": round(volume_ml, 4) if volume_ml else None,
            "area_mm2": round(area_mm2, 2),
            "height_mm": round(height_real_mm, 2) if height_real_mm else None,
            "defective_area_mm2": round(defective_area_mm2, 2),
            "defective_percent": round(defective_percent, 1),
            "final_quality_percent": round(final_quality, 1)
        }
        
        cells_results.append({
            "cell_id": cell_key,
            "class": cls_name,
            "volume_ml": predicted_attributes[cell_key]["volume_ml"],
            "area_mm2": predicted_attributes[cell_key]["area_mm2"],
            "height_mm": predicted_attributes[cell_key]["height_mm"],
            "defective_percent": predicted_attributes[cell_key]["defective_percent"],
            "quality_percent": predicted_attributes[cell_key]["final_quality_percent"],
            "species": species,
            "stage": stage,
            "quality_label": quality_from_model,
            "score": round(float(inst_top.scores[i]), 3),
            "center": [cx, cy]
        })
    

    
    # ────────────────────────────────────────────────
    # Guardar visualizaciones TOP (clean + with text)
    # ────────────────────────────────────────────────
    
    top_clean_path = os.path.join(output_dir, f"{sample_key}_TOP_clean.jpg")
    v_clean = Visualizer(im_top[:, :, ::-1], metadata, instance_mode=ColorMode.SEGMENTATION)
    for i in range(len(inst_top)):
        mask = inst_top.pred_masks[i].numpy()
        class_name = category_names[inst_top.pred_classes[i].item()]
        if class_name == "defective_region":
            v_clean.draw_binary_mask(mask, alpha=0.7, color=(1.0, 0.0, 0.0))
        else:
            v_clean.draw_binary_mask(mask, alpha=0.35)
    cv2.imwrite(top_clean_path, v_clean.output.get_image()[:, :, ::-1])

    top_with_text_path = os.path.join(output_dir, f"{sample_key}_TOP_with_text.jpg")
    v_text = Visualizer(im_top[:, :, ::-1], metadata, instance_mode=ColorMode.SEGMENTATION)
    
    for i in range(len(inst_top)):
        mask = inst_top.pred_masks[i].numpy()
        class_name = category_names[inst_top.pred_classes[i].item()]
        if class_name == "defective_region":
            v_text.draw_binary_mask(mask, alpha=0.7, color=(1.0, 0.0, 0.0))
        else:
            v_text.draw_binary_mask(mask, alpha=0.35)
    
    for cell in cells_results:
        text_lines = [
            f"Class: {cell['class']}",
            f"Vol: {cell['volume_ml'] or 'N/A'} mL",
            f"Area: {cell['area_mm2']} mm²",
            f"Defect: {cell['defective_percent']}%",
            f"Quality: {cell['quality_percent']}%",
            f"Score: {cell['score']}",
        ]
        if cell.get('species'): text_lines.append(f"Species: {cell['species']}")
        if cell.get('stage'): text_lines.append(f"Stage: {cell['stage']}")
        if cell.get('quality_label'): text_lines.append(f"Quality label: {cell['quality_label']}")
        
        cx, cy = cell["center"]
        v_text.draw_text(
            "\n".join(text_lines),
            (cx, cy - 80),
            font_size=18,
            color="yellow",
            horizontal_alignment="center"
        )
    
    cv2.imwrite(top_with_text_path, v_text.output.get_image()[:, :, ::-1])

    # ────────────────────────────────────────────────
    # Resultado para API
    # ────────────────────────────────────────────────
    total_volume = sum(c["volume_ml"] or 0 for c in cells_results)
    
    result = {
        "sample_key": sample_key,
        "height_mm": round(height_real_mm, 2) if height_real_mm else None,
        "total_volume_ml": round(total_volume, 4),
        "cells": cells_results,
        "predicted_attributes": predicted_attributes,
        "images": {
            "top_clean": top_clean_path,
            "top_with_text": top_with_text_path,
        },
        "status": "success" if cells_results else "warning_no_cells"
    }
    
    # Agregar SIDE si existen
    if side_clean_path:
        result["images"]["side_clean"] = side_clean_path
    if side_with_text_path:
        result["images"]["side_with_text"] = side_with_text_path
    
    # Guardar JSON completo
    json_output_path = os.path.join(output_dir, f"{sample_key}_result.json")
    with open(json_output_path, "w", encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result

# Carga inicial al importar el módulo
try:
    setup_predictor()
except Exception as e:
    print(f"[Backend ERROR] Falló carga inicial del modelo:\n{e}")