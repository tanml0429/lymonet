
from copy import copy
from pathlib import Path
import torch

from lymonet.apis import YOLO
from lymonet.apis.yolov8_api import (
    DEFAULT_CFG, DEFAULT_CFG_DICT, IterableSimpleNamespace,
    DetectionTrainer, DetectionValidator, DetectionPredictor, RANK,
    de_parallel,
    )

from lymonet.apis.lymo_api import LYMO_DEFAULT_CFG

from .nn.tasks import LymoDetectionModel
from .val import LymoDetectionValidator
from .predict import LymoDetectionPredictor
from .dataset import LYMODataset, build_lymo_dataset
from .utils.utils import preprocess_correspondence
from .fine_cls_model.classification_validator import LymoClassificationValidator
from .fine_cls_model.clssification_model import LymoClassificationModel
from .fine_cls_model.classify_trainner import LymoClassificationTrainer


class LYMO(YOLO):
    
    def __init__(self, model: str | Path = 'yolov8n.pt', task=None) -> None:
        super().__init__(model, task)
    
    @property
    def task_map(self):
        task_map = super().task_map
        task_map['detect']['model'] = LymoDetectionModel
        task_map['detect']['trainer'] = LymoDetectionTrainer
        task_map['detect']['validator'] = LymoDetectionValidator
        task_map['detect']['predictor'] = LymoDetectionPredictor

        task_map["classify"]["model"] = LymoClassificationModel
        task_map["classify"]["trainer"] = LymoClassificationTrainer
        task_map["classify"]["validator"] = LymoClassificationValidator
        return task_map
    
    
    @staticmethod
    def apply_improvements():
        from .nn.nn import ResBlock_CBAM, CBAM, ChannelAttentionModule, SpatialAttentionModule, RecoveryBlock
        # globals().update(locals())
        globals()['RecoveryBlock'] = RecoveryBlock

        from lymonet.apis.yolov8_api import BaseValidator
        # from ..ultralytics.ultralytics.engine import validator
        from .cfg import get_cfg
        # validator.get_cfg = get_cfg

        # print('apply_improvements')



class LymoDetectionTrainer(DetectionTrainer):

    def __init__(self, cfg=LYMO_DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = LymoDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights) 
        return model
    
    def get_validator(self):
        # super().get_validator()
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'centent_loss', "texture_loss"
        return LymoDetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    
    def build_dataset(self, img_path, mode='train', batch=None):
        """Build LYMO Dataset"""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        # rect = mode == 'val'  # val时，rect=True
        rect = self.args.rect  # 在验证时，设置rect会报错

        return build_lymo_dataset(self.args, 
                                  img_path, batch, 
                                  self.data, mode=mode, 
                                  rect=rect,
                                  stride=gs)
    
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess_batch(batch)
        # batch = preprocess_correspondence(batch, self)
        if self.args.load_correspondence:
            cors = batch.get('correspondence', None) 
            assert len(cors) == len(batch['img']), "correspondence length should be equal to batch size"
            cors_img = torch.zeros_like(batch['img'])
            for i, cor in enumerate(cors):
                if cor is None:  # 没有对应的CDFI图像，就使用原图作为目标
                    cor_img = batch['img'][i]  
                else:
                    cor_img = cor['img']
                    cor_img = cor_img.to(self.device, non_blocking=True).float() / 255
                cors_img[i] = cor_img  # 保存对应的CDFI图像
            # 连续
            cors_img = cors_img.to(self.device, non_blocking=True).float()
            batch['cors_img'] = cors_img
        else:
            batch['cors_img'] = None
        
        return batch