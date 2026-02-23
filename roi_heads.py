import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import get_config_file
from detectron2.data import DatasetMapper, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.data import detection_utils as utils
from detectron2.structures import Instances
SPECIES = ["moso", "other"]
QUALITY = ["poor", "medium", "good"]
STAGE = ["non_embryogenic", "embryogenic"]
@ROI_HEADS_REGISTRY.register()
class CallusROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        dim = self.box_head.output_shape.channels
        self.species_head = AttributeHead(dim, len(SPECIES))
        self.quality_head = AttributeHead(dim, len(QUALITY))
        self.stage_head   = AttributeHead(dim, len(STAGE))

    def _shared_roi_features(self, features, instances):
        # Extrae features para cada ROI usando el box_pooler y box_head
        box_features = self.box_pooler(
            [features[f] for f in self.in_features],
            [x.proposal_boxes for x in instances]
        )
        box_features = self.box_head(box_features)
        return box_features

    def forward(self, images, features, proposals, targets=None):
        # Llamar a la implementación base
        instances, losses = super().forward(images, features, proposals, targets)

        # Obtener features de los ROIs
        box_features = self._shared_roi_features(features, proposals)

        # Calcular logits de atributos
        species_logits = self.species_head(box_features)
        quality_logits = self.quality_head(box_features)
        stage_logits   = self.stage_head(box_features)

        # Distribuir logits por imagen
        start_idx = 0
        for i, inst in enumerate(instances):
            num_inst = len(inst)
            inst.species_logits = species_logits[start_idx:start_idx + num_inst]
            inst.quality_logits = quality_logits[start_idx:start_idx + num_inst]
            inst.stage_logits   = stage_logits[start_idx:start_idx + num_inst]

            # Solo calcular clase predicha en modo inferencia
            if not self.training:
                inst.species = inst.species_logits.argmax(dim=-1)
                inst.quality = inst.quality_logits.argmax(dim=-1)
                inst.stage   = inst.stage_logits.argmax(dim=-1)

            start_idx += num_inst

        # Durante entrenamiento, calcular losses si hay targets
        if self.training and targets is not None:
            callus_targets = [t for t in targets if hasattr(t, "species")]
            if callus_targets:
                start_idx = 0
                for i, t in enumerate(targets):
                    num_inst = len(t)
                    if hasattr(t, "species"):
                        losses.update({
                            "loss_species": F.cross_entropy(
                                species_logits[start_idx:start_idx + num_inst],
                                t.species
                            ),
                            "loss_quality": F.cross_entropy(
                                quality_logits[start_idx:start_idx + num_inst],
                                t.quality
                            ),
                            "loss_stage": F.cross_entropy(
                                stage_logits[start_idx:start_idx + num_inst],
                                t.stage
                            ),
                        })
                    start_idx += num_inst

        return instances, losses

class AttributeHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

