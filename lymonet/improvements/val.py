
import torch
from lymonet.apis.yolov8_api import DetectionValidator, RANK
from .nn.tasks import LymoDetectionModel
from .utils.utils import preprocess_correspondence
# from ..ResNet50xxxx import ResNet50xxxx

class LymoDetectionValidator(DetectionValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self._fine_cls_model = None

    @property
    def fine_cls_model(self):
        if self._fine_cls_model is None:
            self._fine_cls_model = self.load_fine_cls_model()
        return self._fine_cls_model

    def load_fine_cls_model(self):
        # fine_cls_model = ResNet50xxxx()
        fine_cls_model = None
        return fine_cls_model

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
        # 非极大值抑制后，更新矩阵之前，使用额外的分类模型对每个预测框进行分类，替换preds
        preds = self.fine_classify(preds, batch)
        super().update_metrics(preds, batch)
    
    def fine_classify(self, preds, batch):
        """
        精细分类
        preds: list(tensor(num_targets, 6), ...) bs个元素
        for循环，bbox, 从batch(origin_img) -> clip 目标部分:target_img
        送入分类器，得到新的分类，替代preds中的分类
        """

        # for xxx:  # batch
        #     for yyy:  # target
        #         target_img
        #         target_cls = self.fine_cls_model(target_img)
        #         preds[xxx][yyy][5] = target_cls
        
        return preds
    
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