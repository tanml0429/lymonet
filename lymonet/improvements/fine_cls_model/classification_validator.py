

from lymonet.apis.yolov8_api import (
    ClassificationValidator, ClassifyMetrics, ConfusionMatrix, 
    ap_per_class, compute_ap, plot_mc_curve, plot_pr_curve,
)
import damei as dm
import copy
import numpy as np
import torch

from .lymo_confusion_matrix import LymoConfusionMatrix
from lymonet.improvements.utils import utils

class LymoClassificationValidator(ClassificationValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(
            dataloader=dataloader, save_dir=save_dir, pbar=pbar, args=args, _callbacks=_callbacks
        )

        self.lymo_cm = LymoConfusionMatrix(nc=3, conf=0.001, task="classify")
        self.static = dict()


    def get_stats(self):
        cls_stats =  super().get_stats()

        self.lymo_cm.process_cls_preds(self.pred, self.targets)  # update confusion matrix
        merge_type = None
        # merge_type = "normal_vs_abnormals"
        if merge_type is not None:
            self.lymo_cm.merge_classes(merge_type)

        self.lymo_cm.confusion2score(  # 根据cm计算score
            names=self.names,
            return_type="dict")  # compute score
        cm_stats = self.lymo_cm.metrics

        if self.training:
            cm_stats.pop("classes")
            cm_stats.pop("precision")
            cm_stats.pop("recall")
            cm_stats.pop("F1")

        stats = {**cls_stats, **cm_stats}
        return stats
    
    def finalize_metrics(self, *args, **kwargs):
        super().finalize_metrics(*args, **kwargs)
        # 把lymo_cm的结果写入self.metrics(ClassifyMetrics)
        if not self.training:
            self.metrics.precision = self.lymo_cm.metrics["precision"]
            self.metrics.recall = self.lymo_cm.metrics["recall"]
            self.metrics.f1 = self.lymo_cm.metrics["F1"]
        self.metrics.accuracy = self.lymo_cm.metrics["accuracy"]
        self.metrics.mean_precision = self.lymo_cm.metrics["mean_precision"]
        self.metrics.mean_recall = self.lymo_cm.metrics["mean_recall"]
        self.metrics.mean_f1 = self.lymo_cm.metrics["mean_F1"]


    def print_results(self):
        super().print_results()
        metrics = self.lymo_cm.metrics
        if not self.training:
            info = dm.misc.dict2info(metrics)
            print(info)
        return metrics


    def postprocess(self, preds):
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):

        if self.training: 
            n5 = min(len(self.names), 5)
            if isinstance(preds, list):  # cls和echo两个head的输出取出cls
                preds = preds[0]
            self.pred.append(preds.argsort(1, descending=True)[:, :n5])  # (16, 3)  [[0, 1, 2], [2, 1, 0]...]
            self.targets.append(batch["cls"])
            
        else:  # inference

            # 医学知识融合筛选
            preds = self.knowledge_embed_aspect_ratio(preds, batch)
            super().update_metrics(preds, batch)

    
    def knowledge_embed_aspect_ratio(self, preds, batch):
        """
        淋巴结呈椭圆形，长短轴比例在>=2(或<=0.5)时，为正常，否则为三种都可能
        :param preds: (batchsz, 3)
        :param batch: dict
            img: (batchsz, 3, 640, 640)
            cls: (batchsz, )        
        """
        if preds is None:
            return
        
        imgs = batch["img"]
        imgs = imgs.cpu().detach().numpy()
        imgs = imgs.transpose(0, 2, 3, 1) * 255
        imgs = imgs.astype(np.uint8)
        preds_np = copy.deepcopy(preds.cpu().detach().numpy())

        batchsz = preds_np.shape[0]
        for i in range(batchsz):
            img = imgs[i]
            h, w, _ = img.shape  # 由于img是resize后的，所以h, w不是原始的
            org_img = utils.restore_org_img(img)
            org_h, org_w, _ = org_img.shape

            aspect_ratio = org_w / org_h

            cls_prob = preds_np[i]
            cls = np.argmax(cls_prob)
            pd_cls_name = self.names[cls]
            gt_cls = batch["cls"][i].cpu().detach().numpy()

            # 领域知识：长宽比<1.9时，一定是不正常
            if aspect_ratio < 1.9:
                if pd_cls_name == "normal":
                    # print(f"img{i} gt_cls: {gt_cls}, pd_cls: {cls}, since it should not be abnormal, cls=0 or 1")
                    # 领域知识：nomral和inflamatory接近，与metastatic差异大，所以这里直接设置为inflamatory，而不是依赖概率判断
                    new_cls = 0
                    new_prob = torch.zeros_like(preds[i])
                    new_prob[new_cls] = 1
                    preds[i] = new_prob
                    value = {"pd_cls": cls, "gt_cls": int(gt_cls), "to_be_cls": new_cls, "aspect_ratio": aspect_ratio, "reason": "aspect_ratio<1.6", "pd_cls_prob": list(cls_prob)}
                    self.update_static("knowledge_1", value)
                    

            # 领域知识：长宽比大于5时，一定是正常
            if aspect_ratio >= 6:
                if pd_cls_name != "normal":
                    new_cls = 2
                    new_prob = torch.zeros_like(preds[i])
                    new_prob[new_cls] = 1
                    preds[i] = new_prob
                    value = {"pd_cls": cls, "gt_cls": int(gt_cls), "to_be_cls": new_cls, "aspect_ratio": aspect_ratio, "reason": "aspect_ratio>=5", "pd_cls_prob": list(cls_prob)}
                    self.update_static("knowledge_2", value)
                    # 统计发现，测试集中 > 5的图像已经全部被正确预测为正常，其实不需要再做这个判断了
            pass
        return preds
    

    def update_static(self, key, value):
        if key not in self.static:
            self.static[key] = list()
        self.static[key].append(value)





    
   
    



    
    

