
import torch
from copy import deepcopy
import damei as dm

from lymonet.apis.yolov8_api import (
    ClassificationValidator, ClassificationModel,
    yaml_model_load, 
)
from ..nn.tasks import parse_model


LOGGER = dm.get_logger("classification_model.py")

class LymoClassificationModel(ClassificationModel):

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)


    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return LymoClassificationLoss(self)



class LymoClassificationLoss:

    def __init__(self, model):
        self.hyp = model.args

    def __call__(self, preds, batch):
        """
        Compute the classification loss between predictions and true labels.
        :preds: [tensor(batch_size, 3), tensor(batch_size, 2)]
            element 0: original 3 classification for inflammtary, metastatic, normal
            elemnet 1: 2 classification for with or without high echo area 
        """
        if isinstance(preds, list):  # multiple predictions，包含cls和echo两个head的输出
            loss = torch.zeros(2, device=preds[0].device)
            cls_loss = torch.nn.functional.cross_entropy(preds[0], batch["cls"], reduction="mean")
            echo_loss = self.compute_echo_loss(preds[1], batch, reduction="mean")
            loss[0] = cls_loss * self.hyp.cls_loss_gain
            loss[1] = echo_loss * self.hyp.echo_loss_gain
            return loss.sum(), loss.detach()
        else:
            loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
            loss_items = loss.detach()
            return loss, loss_items

    def compute_echo_loss(self, x, batch, reduction="mean"):
        """
        x: tensor(batch_size, 2)
        batch: dict
        """
        # 认为inflammetary和normal都有高回声区域，为第0类，metastatic都无高回声区域，为第1类
        # i.e. 0 and 2 → 0, 1 → 1
        gt_labels = batch["cls"].clone()
        gt_labels[gt_labels == 2] = 0

        loss = torch.nn.functional.cross_entropy(x, gt_labels, reduction=reduction)
        # loss_items = loss.detach()
        return loss