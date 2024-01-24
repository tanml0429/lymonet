
import torch
from torch import nn
from torchvision.models import vgg19, resnet50, VGG19_Weights

from lymonet.apis.yolov8_api import (
    make_anchors, dist2bbox,
    TaskAlignedAssigner,
    BboxLoss,
    xywh2xyxy,
)


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        # self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:9])
        # self.feature_extractor = self.feature_extractor.to('cuda')
        # # self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()
        # self.criterion = self.criterion.to('cuda')
        self.feature_extractor = self.feature_extractor.half()
        self.feature_extractor.eval()

    def forward(self, x):
        return self.feature_extractor(x)

class LymoDetectionLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.model = model

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        self.MSEpercept = nn.MSELoss().to(device)  #  for content loss
        self.L1loss = nn.L1Loss().to(device)  # For texture loss
        self.feature_extractor = VGGFeatureExtractor().to(device)  


    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        :param preds: (features from P3 P4 P5, restored feature from RB) since the DetectWithRecoveryBlock returns two values
        """

        """
        无RB时：
            train: preds为list([P3, P4, P5])
            val: preds为tuple(loss, list([P3, P4, P5]))
        有RB时：
            train: preds为tuple(restored_feature, list([P3, P4, P5]))
            val: preds为tuple(restored_feature, tuple(loss, list([P3, P4, P5]))
        """
        cors_img = batch.get('cors_img', None)
        has_recovery_block = False if cors_img is None else True  # 是否带有RB模块
        if has_recovery_block:  # 没有corespoding_img时，preds一定是[P3, P4, P5]
            assert len(preds) == 2, "preds should be (restored_feature, [P3, P4, P5]) when have corresponding image"
            restored_feature, preds = preds
        else:
            assert len(preds) in [2, 3], "preds should be [P3, P4, P5] when no corresponding image"
            preds = preds
            

        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, content loss, texture loss

        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain  7.5
        loss[1] *= self.hyp.cls  # cls gain  0.5
        loss[2] *= self.hyp.dfl  # dfl gain  1.5

        
        if cors_img is not None:
            loss[3] = self.compute_content_loss(restored_feature, batch['cors_img'])  # TODO: 应该是对应CDFI进入RB后的Feature
            loss[4] = self.L1loss(restored_feature, batch['cors_img'])
            loss[3] *= self.hyp.content_loss_gain
            loss[4] *= self.hyp.texture_loss_gain
        else:
            loss[3] = 0
            loss[4] = 0

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    def compute_content_loss(self, restored_feature, cors_img):
        f1 = self.feature_extractor(restored_feature)
        f2 = self.feature_extractor(cors_img)
        return self.MSEpercept(f1, f2)
    
    def load_corresponding_batch(self, batch):
        """Deprecated, Process in LymoDetectionTrainer.preprocess_batch()"""
        """根据当前的batch信息，加载US图对应的CDFI图，用于计算content loss和texture loss"""
        raise DeprecationWarning
        cors = batch['correspondence']
        assert len(cors) == len(batch['img']), "correspondence length should be equal to batch size"
        cors_img = torch.zeros_like(batch['img'])
        for i, cor in enumerate(cors):
            if cor is None:
                cor_img = batch['img'][i]  # 没有对应的CDFI图像，就使用原图作为目标
            else:
                cor_img = cor['img']
            cors_img[i] = cor_img  # 保存对应的CDFI图像
        # 连续
        cors_img = cors_img.to(self.device)
        # cors_img = cors_img.half()
        return cors_img


    
    


    


