import numpy as np
import copy 
import torch

from lymonet.improvements.utils import utils



def knowledge_embed_aspect_ratio(preds, batch, names):
    """
    淋巴结呈椭圆形，长短轴比例在>=2(或<=0.5)时，为正常，否则为三种都可能
    :param preds: (batchsz, 3)
    :param batch: dict
        img: (batchsz, 3, 640, 640)
        cls: (batchsz, )        
    """
    if preds is None:
        return
    
    for i, pred in enumerate(preds):
        bbox_pd = pred[:, :4]  # tensor(num_targets, 4)  x1y1x2y2
        conf_pd = pred[:, 4]  # tensor(num_targets)
        cls_pd = pred[:, 5]  # tensor(num_targets)
        batch_idx = batch['batch_idx'] == i
        cls_gt = batch['cls'][batch_idx]
        
        # aspect_ratios = []
        for j, bbox in enumerate(bbox_pd):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = w / h

            cls_p = cls_pd[j]
            cls_pd_name = names[int(cls_p)]

            # aspect_ratios.append(aspect_ratio)

            # 领域知识：长宽比<1.9时，一定不是正常
            if aspect_ratio < 1.4:
                if cls_pd_name == "normal":
                    # 领域知识：nomral和inflamatory接近，与metastatic差异大，所以这里直接设置为inflamatory，而不是依赖概率判断
                    new_cls_name = "inflamatory"
                    new_cls = names.index(new_cls_name)
                    value = {"pd_cls": int(cls_p), "gt_cls": list(cls_gt), "to_be_cls": new_cls, "aspect_ratio": aspect_ratio, "reason": "aspect_ratio<1.6"}
                    # print(value)
                    preds[i][j, 5] = new_cls
                    # self.update_static("knowledge_1", value)
                    pass
            
            if aspect_ratio >= 4.10:
                if cls_pd_name != "normal":
                    new_cls_name = "normal"
                    new_cls = names.index(new_cls_name)
                    value = {"pd_cls": int(cls_p), "gt_cls": list(cls_gt), "to_be_cls": new_cls, "aspect_ratio": aspect_ratio, "reason": "aspect_ratio>=5"}
                    # print(value)
                    preds[i][j, 5] = new_cls
                    pass


        """
        # 领域知识：长宽比大于5时，一定是正常
        if aspect_ratio >= 6:
            if pd_cls_name != "normal":
                new_cls = 2
                new_prob = torch.zeros_like(preds[i])
                new_prob[new_cls] = 1
                preds[i] = new_prob
                value = {"pd_cls": cls, "gt_cls": int(gt_cls), "to_be_cls": new_cls, "aspect_ratio": aspect_ratio, "reason": "aspect_ratio>=5", "pd_cls_prob": list(cls_prob)}
                # self.update_static("knowledge_2", value)
                # 统计发现，测试集中 > 5的图像已经全部被正确预测为正常，其实不需要再做这个判断了
        """
        # print(aspect_ratios)
    return preds