


from pathlib import Path
import os, sys
import cv2
import numpy as np
import torch
import copy

here = Path(__file__).parent

from lymonet.apis.yolov8_api import ops
from .medical_knowledge import knowledge_embed_aspect_ratio


class FineClsModel:

    def __init__(self) -> None:
        self.model = self.load_fine_cls_model()

        self.count = 0
        self.static = dict()

    def load_fine_cls_model(self):
        # fine_cls_model = ResNet50xxxx()
        from lymonet.apis import LYMO
        # model: str =  '/home/tml/VSProjects/lymonet/runs/classify/yolov8s_aug1_valsquare1/weights/best.pt'
        model: str = '/home/tml/VSProjects/lymonet/lymonet/ultralytics/runs/classify/yolov8s_class10/weights/best.pt'
        fine_cls_model = LYMO(model)
        return fine_cls_model

    def __call__(self, preds, batch, **kwargs):
        """
        精细分类
        preds: list(tensor(num_targets, 6), ...) bs个元素
        for循环，bbox, 从batch(origin_img) -> clip 目标部分:target_img
        送入分类器，得到新的分类，替代preds中的分类
        """
        sreen_by_threshold = kwargs.get('sreen_by_threshold', 0.25)
        medical_knowledge_embed = kwargs.get('medical_knowledge_embed', False)
        if sreen_by_threshold:
            for i, pred in enumerate(preds):
                pred = pred[pred[:, 4] > sreen_by_threshold]
                preds[i] = pred
        if medical_knowledge_embed:
            preds = knowledge_embed_aspect_ratio(preds, batch, names=['metastatic', 'inflamatory', 'normal'])
            return preds
        
        img_size = 640
        # return preds
        slice_imgs, target_idx_in_batch = self.get_slice_img(preds, batch)  # list(numpy(切片))  list(0,0,0,0,1,1,2,3,3,...) 代表图片中的目标在batch中的索引
        slice_imgs = self.preprocess(slice_imgs, img_sz=img_size)  # list(tensor(3, 640, 640), ...
        slice_imgs = slice_imgs.to(batch['img'].device, non_blocking=True).float()

        result = self.model.predict(source=slice_imgs, stream=False, verbose=False)
        # probs = [x.probs for x in result]
        # 获取精细分类结果
        fine_cls_preds = [x.probs.top1 for x in result]  # list() 有16张图每张图n个目标，共约16*n个元素 
        fine_pd_confs = [x.probs.top1conf.cpu().detach().numpy() for x in result]  # list()
        fine_pd_probs = [x.probs.data.cpu().detach().numpy() for x in result]  # list()
        fine_pd_probs = [np.array((x[1], x[0], x[2])) for x in fine_pd_probs]  # 由于fine_cls_model的输出顺序和原来的不一样，所以要调整一下
        # 把file_cls_preds list(16*n)整理成list(16, n)
        batch_size = len(preds)
        fine_cls_preds_new = []  # list(list(n1), list(n2), list(n3), ...)
        for x in fine_cls_preds:
            if x == 0:
                fine_cls_preds_new.append(1)
            elif x == 1:
                fine_cls_preds_new.append(0)
            else:
                fine_cls_preds_new.append(x)
        fine_cls_preds = fine_cls_preds_new
        assert len(target_idx_in_batch) == len(fine_cls_preds), f'len(target_idx_in_batch): {len(target_idx_in_batch)}, len(fine_cls_preds): {len(fine_cls_preds)}'
        
        for im_idx in range(batch_size):  # 遍历batch_size次
            fine_cls_pred_in_im = np.array(fine_cls_preds)[(np.array(target_idx_in_batch) == im_idx)]
            fine_pd_conf = np.array(fine_pd_confs)[(np.array(target_idx_in_batch) == im_idx)]
            fine_pd_prob = np.array(fine_pd_probs)[(np.array(target_idx_in_batch) == im_idx)]
            batch_idx = batch['batch_idx'] == im_idx
            gt_cls = batch['cls'][batch_idx]
            # fine_cls_preds_new.append(fine_cls_pred_in_im.tolist())
            fine_pd_cls = fine_cls_pred_in_im.tolist()
            some_pd_cls = self.error_analysis(preds, batch, im_idx, fine_pd_cls, fine_pd_conf, fine_pd_prob)
            
            pd_cls = preds[im_idx][:, 5]  # n个值
            pd_cls = pd_cls.clone().int().flatten().cpu().detach().numpy()
            pd_conf = preds[im_idx][:, 4].clone().cpu().detach().numpy()  # n个值 (n, )
            # 分为一类的置信度，剩余部分开置于两类
            pd_prob = np.zeros((len(pd_conf), 3))
            pd_conf = pd_conf * 2  # pd_conf既包含bbox的置信度，又包含分类的置信度，所以要乘以2
            pd_conf = np.clip(pd_conf, 0, 1)
            for i, c in enumerate(pd_cls):
                pd_prob[i, :] = (1 - pd_conf[i]) / 2  # 三个位置都赋值，剩余的一半
                pd_prob[i, c] = pd_conf[i]  # 该位置赋值为置信度

            use_aspect_ratio_weight = True  # 是否使用根据aspect_ratio的敏感性分析得到的权重
            if use_aspect_ratio_weight:
                bbox_pd = preds[im_idx][:, :4]  # tensor(num_targets, 4)
                aspect_ratios = [(b[2]-b[0])/(b[3]-b[1]) for b in bbox_pd]  # w/h
                aspect_ratios = torch.tensor(aspect_ratios).cpu().detach().numpy()
                aspect_ratios = [round(x, 2) for x in aspect_ratios]
                ws = self.get_weight(aspect_ratios)
            else:
                ws = 0.5 * np.ones(len(pd_prob))
            fuse_prob0 = ws[:, None] * pd_prob + (1-ws)[:, None] * fine_pd_prob  # (n, 3)
            fuse_prob1 = np.multiply(pd_prob, fuse_prob0) / np.sum(np.multiply(pd_prob, fuse_prob0), axis=1)[:, None]  # (n, )
            fuse_pd_cls = np.argmax(fuse_prob1, axis=1)

            # preds[im_idx][:, 5] = torch.from_numpy(fuse_pd_cls).float().to(preds[im_idx].device)
            # preds[im_idx][:, 5] = torch.from_numpy(fine_cls_pred_in_im).float().to(preds[im_idx].device)
            # preds[im_idx][:, 5] = gt_cls[0, :]  # 只取gt的第一个值替换，不太对
            preds[im_idx][:, 5] = torch.from_numpy(some_pd_cls).float().to(preds[im_idx].device)
        
        print(self.static)
        pd_mc = self.static['pd_cls more correct asp_r']
        fine_pd_mc = self.static['fine_pd_cls more correct asp_r']
        pd_mc = np.array([x for x in pd_mc]).flatten()
        fine_pd_mc = np.array([x for x in fine_pd_mc]).flatten()
        bins = np.arange(0.5, 8.5, 0.5)
        hist, bin_edges = np.histogram(pd_mc, bins=bins)
        hist2, bin_edges2 = np.histogram(fine_pd_mc, bins=bins)
        for i in range(len(hist)):
            prcent = hist[i] / (hist[i] + hist2[i])
            print(f"Aspect Ratio range: {bin_edges[i]} to {bin_edges[i+1]}, Count: {hist[i]:>3} vs {hist2[i]:>3}, precent: {prcent:.2f}")
        return preds
    
    def error_analysis(self, preds, batch, im_idx, fine_pd_cls, fine_pd_conf, fine_pd_prob):
        batch_idx = batch['batch_idx'] == im_idx
        gt_cls = batch['cls'][batch_idx]
        gt_cls = gt_cls.clone().int().flatten().cpu().detach().numpy()
        pd_cls = preds[im_idx][:, 5]  # n个值
        pd_cls = pd_cls.clone().int().flatten().cpu().detach().numpy()
        pd_conf = preds[im_idx][:, 4].cpu().detach().numpy()  # n个值 (n, )

        fine_pd_cls = np.array(fine_pd_cls)
        fine_pd_conf = np.array(fine_pd_conf)
        fine_pd_prob = np.array(fine_pd_prob)  # (n, 3)

        # 统计按aspect_ratio分类的错误情况
        bbox_pd = preds[im_idx][:, :4]  # tensor(num_targets, 4)
        aspect_ratios = [(b[2]-b[0])/(b[3]-b[1]) for b in bbox_pd]  # w/h
        aspect_ratios = torch.tensor(aspect_ratios).cpu().detach().numpy()
        aspect_ratios = [round(x, 2) for x in aspect_ratios]

        # 分为一类的置信度，剩余部分开置于两类
        pd_prob = np.zeros((len(pd_conf), 3))
        for i, c in enumerate(pd_cls):
            pd_prob[i, :] = (1 - pd_conf[i]) / 2  # 三个位置都赋值，剩余的一半
            pd_prob[i, c] = pd_conf[i]  # 该位置赋值为置信度
        
        # 概率乘积法
        fuse_prob = np.multiply(pd_prob, fine_pd_prob) / np.sum(np.multiply(pd_prob, fine_pd_prob), axis=1)[:, None]  # (n, )
        # 加权平均概率乘积法
        use_aspect_ratio_weight = True  # 是否使用根据aspect_ratio的敏感性分析得到的权重
        ws = 0.7 * np.ones(len(pd_prob)) if not use_aspect_ratio_weight else self.get_weight(aspect_ratios)  # weights

        fuse_prob0 = ws[:, None] * pd_prob + (1-ws)[:, None] * fine_pd_prob  # (n, 3)
        fuse_prob1 = np.multiply(pd_prob, fuse_prob0) / np.sum(np.multiply(pd_prob, fuse_prob0), axis=1)[:, None]  # (n, )
        fuse_pd_cls = np.argmax(fuse_prob1, axis=1)
        # 计算是否更改分类
        pd_fit_fuse = np.array([pd_cls == x for x in fuse_pd_cls]).flatten()
        fine_pd_fit_fuse = np.array([fine_pd_cls == x for x in fuse_pd_cls]).flatten()
        if np.sum(pd_fit_fuse) >= np.sum(fine_pd_fit_fuse):  # 不修改
            is_changed = False
        elif np.sum(pd_fit_fuse) < np.sum(fine_pd_fit_fuse):  # 修改
            is_changed = True

        # 分析检测模型的分类错误，但精细模型分类正确的情况。
        pd_correct = np.array([pd_cls == x for x in gt_cls]).flatten()
        fine_pd_correct = np.array([fine_pd_cls == x for x in gt_cls]).flatten()
        if np.sum(pd_correct) < np.sum(fine_pd_correct):  # 应该用fine_cls替换
            self.count += 1
            print(f'gt_cls: {gt_cls}, pd_cls: {pd_cls}, conf: {pd_conf}, fine_pd_cls: {fine_pd_cls}, conf: {fine_pd_conf} count: {self.count}')
            self.extend_to_static(f'fine_pd_cls more correct asp_r', aspect_ratios)
            self.add_to_static(f'should True and replaced: {is_changed}', 1)
            # return fine_pd_cls
        elif np.sum(pd_correct) > np.sum(fine_pd_correct):  # 不应该用fine_cls替换
            # self.count += 1
            # print(f'gt_cls: {gt_cls}, pd_cls: {pd_cls}, conf: {pd_conf}, fine_pd_cls: {fine_pd_cls}, conf: {fine_pd_conf} count: {self.count}')
            self.extend_to_static(f'pd_cls more correct asp_r', aspect_ratios)
            self.add_to_static(f'should False and replaced: {is_changed}', 1)
            # return pd_cls
        else:  # 无所谓
            pass
        
        # aspect_ratio 大于4时，用fine_cls替换
        if len(aspect_ratios) > 0 and np.max(aspect_ratios) > 4:
            for i, ar in enumerate(aspect_ratios):
                if ar > 4:
                    pd_cls[i] = fine_pd_cls[i]
        return pd_cls
        
    def extend_to_static(self, key, value):
        if key not in self.static:
            self.static[key] = []
        self.static[key].extend(value)
    
    def append_to_static(self, key, value):
        if key not in self.static:
            self.static[key] = []
        self.static[key].append(value)

    def add_to_static(self, key, value):
        if key not in self.static:
            self.static[key] = 0
        self.static[key] += value
    
    def preprocess(self, imgs, img_sz=640, save_img=False):
        """
        imgs: list(tensor(3, h, w), ...)  list(num_targets, ...) bs个元素
        """
        for i, img in enumerate(imgs):
            img = self.letterbox_image(img, (img_sz, img_sz), need_scaleup=False)
            img_tensor = torch.from_numpy(img).float() / 255  # (h, w, 3)
            if save_img and i==0:
                self.save_tensor_img(img_tensor, name=f'preprocess_img_slice_{i}.png')
            img_tensor = img_tensor.permute(2, 0, 1)  # (3, h, w)
            imgs[i] = img_tensor
        imgs = torch.stack(imgs)  # tensor(bs, 3, h, w)
        return imgs

    
    def get_slice_img(self, preds, batch, save_img=False, which_orgin_imgs='file'):
        """
        根据预测的结果和原始图像，截取目标部分
        返回：tensor(bs*n, 3, 640, 640)  2表示每张图最多截取2个目标
        """
        if which_orgin_imgs == 'tensor':  # 
            imgs = batch['img'].clone()  # tensor(bs, 3, h, w)
            imgs = imgs.permute(0, 2, 3, 1)  # tensor(bs, h, w, 3)
            # for i, img in enumerate(imgs):
            #     if save_img:
            #         self.save_tensor_img(img, name=f'origin_img_{i}.png')

        num_targets_list = [len(pred) for pred in preds]  # list(num_targets, ...) bs个元素
        
        # slice_imgs = torch.zeros((sum(num_targets_list), img_sz, img_sz, 3))  # tensor(bs, 2, 3, 640, 640)
        slice_imgs = []
        target_idx_in_batch = []   # 每个目标在第几个batch中

        for i, pred in enumerate(preds):  # 每次循环是一个batch的里一张图
            if which_orgin_imgs == 'tensor':
                img = imgs[i].cpu().detach().numpy()
                img = img * 255 if img.max() <= 1 else img
                predn = pred.clone()
            else:
                img = batch['im_file'][i]  # tensor(3, h, w)
                img = cv2.imread(img)
                pbatch = self._prepare_batch(i, batch)
                predn = self._prepare_pred(pred, pbatch)  # bbox缩放到原图大小
            h, w = img.shape[:2]
            bbox_pd = predn[:, :4]  # tensor(num_targets, 4)
            conf_pd = predn[:, 4]  # tensor(num_targets)
            cls_pd = predn[:, 5]  # tensor(num_targets)
            num_targets = len(predn)

            for j, bbox in enumerate(bbox_pd):
                bbox = bbox.int()  # x1, y1, x2, y2
                # 预测的bbox可能会超出原图边界
                assert bbox[0] < bbox[2] and bbox[1] < bbox[3], f'bbox error: {bbox}'
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(w, bbox[2])
                bbox[3] = min(h, bbox[3])
                slice_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]  # tensor(3, h, w)
                if save_img and i==0 and j==0:
                    self.save_tensor_img(slice_img, name=f'slice_img_{i}_{j}-{len(bbox_pd)}_conf{conf_pd[j]:.2f}.png')
                # slice_img = slice_img.cpu().detach().numpy()
                slice_imgs.append(slice_img)
                target_idx_in_batch.append(i)
            pass
        return slice_imgs, target_idx_in_batch
    
    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=batch['img'].device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn
    
    def save_tensor_img(self, tensor, name):
        if isinstance(tensor, torch.Tensor):
            img_np = tensor.cpu().detach().numpy()
        else:
            img_np = tensor
        img_np = img_np * 255 if img_np.max() <= 1 else img_np
        img_np = img_np.astype('uint8')
        cv2.imwrite(f'{here}/{name}', img_np)

    def letterbox_image(self, image, size, need_scaleup=False):
        is_tensor = False
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
            is_tensor = True
        if need_scaleup:
            image = image * 255
            image = image.astype('uint8')
        # BGR RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ih, iw = image.shape[0:2]
        h, w = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'{here}/image_resized.png', image_resized)
        new_image = np.ones((h, w, 3), dtype=np.uint8) * 128 # 128 for gray
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        new_image[dy:dy+nh, dx:dx+nw] = image_resized
        if is_tensor:
            new_image = torch.from_numpy(new_image)
        return new_image
    
    def get_weight(self, aspect_ratios):
        
        ws = []
        # 根据敏感性分析得到的权重
        for ar in aspect_ratios:
            if ar < 0.5:
                w1 = 1.0
            elif ar < 1:
                w1 = 0.50
            elif ar < 1.5:
                w1 = 0.56
            elif ar < 2:
                w1 = 0.55
            elif ar < 2.5:
                w1 = 0.54
            elif ar < 3:
                w1 = 0.51
            elif ar < 3.5:
                w1 = 0.56
            elif ar < 4:
                w1 = 0.71
            elif ar < 4.5:
                w1 = 0.20
            else:
                w1 = 0.0
            ws.append(w1)
        return np.array(ws)


        