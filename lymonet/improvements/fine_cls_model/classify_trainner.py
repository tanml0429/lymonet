
import torch

from .clssification_model import LymoClassificationModel
from lymonet.apis.yolov8_api import (
    ClassificationTrainer,
    RANK, DEFAULT_CFG,
)
from lymonet.apis.lymo_api import LYMO_DEFAULT_CFG

from .classification_validator import LymoClassificationValidator

class LymoClassificationTrainer(ClassificationTrainer):

    def __init__(self, cfg=LYMO_DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = ["cls_loss", "echo_loss"]
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        model = LymoClassificationModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        model.args = self.args
        
        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model
    

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))

    def get_validator(self):
        return LymoClassificationValidator(self.test_loader, self.save_dir, _callbacks=self.callbacks)
