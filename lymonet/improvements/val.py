
import torch
from lymonet.apis.yolov8_api import (
    DetectionValidator, RANK, DetMetrics,
    get_save_dir, check_imgsz, callbacks,
    )
from lymonet.apis.lymo_api import get_cfg
from .nn.tasks import LymoDetectionModel
from .utils.utils import preprocess_correspondence
# from ..ResNet50xxxx import ResNet50xxxx
from .fine_cls_model.fine_cls_model import FineClsModel



class LymoDetectionValidator(DetectionValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        # super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        # BaseValidator的初始化
        self.args = get_cfg(overrides=args)  # 注，此处替代成了lymo的默认配置
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        
        # DetectionValidator的初始化
        self.nt_per_class = None
        self.is_coco = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        self._fine_cls_model = None

    @property
    def fine_cls_model(self):
        if self._fine_cls_model is None:
            self._fine_cls_model = FineClsModel()
        return self._fine_cls_model

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = LymoDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def postprocess(self, preds):
        preds = super().postprocess(preds)
        return preds
    
    def update_metrics(self, preds, batch):

        if self.args.fine_cls:  # 非极大值抑制后，更新矩阵之前，使用额外的精细分类模型对每个预测框进行分类，替换preds
            preds = self.fine_cls_model(preds, batch)
        super().update_metrics(preds, batch)
    
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        # batch = preprocess_correspondence(batch, self)
        cors = batch.get('correspondence', None)
        if cors:
            assert len(cors) == len(batch['img']), "correspondence length should be equal to batch size"
            cors_img = torch.zeros_like(batch['img'])
            for i, cor in enumerate(cors):
                if cor is None:  # 没有对应的CDFI图像，就使用原图作为目标
                    cor_img = batch['img'][i]  
                else:
                    cor_img = cor['img']
                    cor_img = cor_img.to(self.device, non_blocking=True).float() / 255
                cors_img[i] = cor_img  # 保存对应的CDFI图像
            
            if self.args.half:
                cors_img = cors_img.to(self.device, non_blocking=True).half()
            else:
                cors_img = cors_img.to(self.device, non_blocking=True).float()
            batch['cors_img'] = cors_img
        return batch
    

