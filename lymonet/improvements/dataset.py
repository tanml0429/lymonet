

import json
from pathlib import Path
import torch
import cv2
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment

from lymonet.apis.yolov8_api import (
    YOLODataset,
    colorstr,
    Compose,
    LetterBox,
    Format,
    bbox_iou,
    bbox_ioa,
)

from .augment import (
    lymo_transforms, 
    LymoFormat,
    LymoLetterBox,
)

from .utils import utils

class LYMODataset(YOLODataset):
    """LYMO训练时需要额外加载US对应的CDFI图，因此需要重写YOLODataset
    """
    def __init__(self, *args, data=None, task="detect", **kwargs):
        mode = kwargs.pop('mode', 'train')  # train val test, pop it to avoid error
        self.load_correspondence = kwargs.pop('load_correspondence', False)
        super().__init__(*args, data=data, task=task, **kwargs)
        
        if self.load_correspondence:  # 加载correspondence信息
            metadata_path = data.get("metadata", None)
            assert metadata_path is not None, "metadata path should not be None when load_correspondence is True, plese check your data config (.yaml)"
            self.metadata = self.load_metadata(data["metadata"])

    @property
    def correspondence(self):
        return self.metadata["correspondence"]  # 一个dict，key是US图像的文件名，value是CDFI图像的文件名组成的列表

    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    def get_coresponding_file(self, im_file):
        """Find the corresponding CDFI image file for the given US image file"""
        imf = Path(im_file)
        cors_stems = self.correspondence.get(imf.stem, [])  # 训练的图片本身为CDFI时，没有对应的CDFI图像
        if len(cors_stems) == 0:
            return None
        # 有多个CDFI图像，随机选一个, TODO: 选择方法
        cors_stem = cors_stems[0]
        cors_file = imf.parent / f'{cors_stem}{imf.suffix}'
        return str(cors_file)
    
    def get_correspondence(self, label):
        """Find the all correspondence info for the given US image index"""
        cors_file = self.get_coresponding_file(label["im_file"])  # 对应的CDFI图像文件路径，可能是None
        # if cors_file is not None:
        #     if "patient0435_node_2&3&4_CDFI_3319.png" in cors_file:  # 这东西会自动加载为(3, 672, 672)，测试一下怎么回事
        #         pass
        if cors_file is None:
            return None
        cors_idx = self.im_files.index(cors_file)
        correspondence = super().get_image_and_label(cors_idx)
        
        return correspondence
    
    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        if self.load_correspondence:
            label["correspondence"] = self.get_correspondence(label)
            self.align_correspondence(label, save_img=False)  # 对齐B超和CDFI图像
        return label
    
    def align_correspondence(self, label, save_img=False):
        """将corresponding的bbox和img的bbox对齐"""
        if label["correspondence"] is None:
            return
        img1 = label["img"]
        bboxes_obj = label["instances"]._bboxes
        bboxes1 = copy.deepcopy(bboxes_obj.bboxes)  # (n, 4)
        format1 = bboxes_obj.format
        assert format1 == "xywh", "bbox1 should be in xywh format"
        bboxes1 = utils.denormalize(img1, bboxes1)  # (n, 4)  in pixel
        if save_img:
            img1 = utils.plot_bboxes(img1, bboxes1)
            cv2.imwrite("img1.png", img1)

        cors = label["correspondence"]
        img2 = cors["img"]
        im2_h, im2_w = img2.shape[:2]
        instances2 = cors["instances"]
        bboxes2_obj = instances2._bboxes
        bboxes2 = copy.deepcopy(bboxes2_obj.bboxes)
        assert bboxes2_obj.format == "xywh", "bbox2 should be in xywh format"
        bboxes2 = utils.denormalize(img2, bboxes2)  # (n, 4)  x1y1x2y2 in pixel

        # bbox1和bbox2是多对多的关系，计算IoU
        bboxes1 = torch.from_numpy(bboxes1)
        bboxes2 = torch.from_numpy(bboxes2)
        ious = self.compute_iou(bboxes1, bboxes2, xywh=True)  # (n1, n2)
        # 根据iou分配bbox的对应关系 # 使用 linear_sum_assignment 函数找到最优分配
        cost_matrix = 1 - ious
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        src_points = []
        dst_points = []
        for i, j in zip(row_ind, col_ind):
            dst_bbox = bboxes1[i]  # xywh, 注意：这里是反的，是b2映射到b1
            src_bbox = bboxes2[j]  # xywh
            # bbox是xc yc w h的形式，转换成4个点，左上，右上，右下，左下
            src_4points = utils.xywh2four_points(src_bbox)
            dst_4points = utils.xywh2four_points(dst_bbox)
            src_points.append(src_4points)
            dst_points.append(dst_4points)
        src_points = np.array(src_points)  # (n, 4, 2)
        dst_points = np.array(dst_points)  # (n, 4, 2)
        # (n, 4, 2) → (n*4, 2)
        src_points = src_points.reshape(-1, 2)
        dst_points = dst_points.reshape(-1, 2)

        # 计算单应性矩阵
        homography_matrix, status = cv2.findHomography(src_points, dst_points)  # status: 每个个点是否是内点
        if save_img:  # 对齐前的图像
            img2 = utils.plot_bboxes(img2, bboxes2)
            cv2.imwrite("unalign_img2.png", img2)
        # 使用单应性矩阵变换图像
        transformed_img2 = cv2.warpPerspective(img2, homography_matrix, (im2_w, im2_h))
        transformed_bboxes2 = utils.warp_bboxes(bboxes2, homography_matrix)  # xywh to xywh in pixel
        if save_img:
            transformed_img2 = utils.plot_bboxes(transformed_img2, transformed_bboxes2)
            cv2.imwrite("aligned_img2.png", transformed_img2)
        transformed_bboxes2_norm = utils.normalize(img2, transformed_bboxes2)  # xywh to xywh in norm

        # 写回cors里
        cors["img"] = transformed_img2
        instances2.update(bboxes=transformed_bboxes2_norm)
        cors["instances"] = instances2

    def compute_iou(self, bboxes1, bboxes2, **kwargs):
        ious = np.zeros((len(bboxes1), len(bboxes2)))
        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                ious[i, j] = bbox_iou(bbox1, bbox2, **kwargs)
        return ious

    def __getitem__(self, index):
        label = self.get_image_and_label(index)  # 这里增加了correspondence信息
        label = self.transforms(label)
        return label

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            # transforms = v8_transforms(self, self.imgsz, hyp)
            # 改为lymo_transforms
            transforms = lymo_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LymoLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            LymoFormat(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms
    
    # @staticmethod
    # def collate_fn(batch):
    #     """Collates data samples into batches. include corresponding image"""
    #     new_batch = {}
    #     keys = batch[0].keys()
    #     values = list(zip(*[list(b.values()) for b in batch]))
    #     for i, k in enumerate(keys):
    #         value = values[i]
    #         if k == 'img':
    #             value = torch.stack(value, 0)
    #         if k in ['masks', 'keypoints', 'bboxes', 'cls']:
    #             value = torch.cat(value, 0)
    #         # if k == 'correspondence':
    #         #     value 
    #         new_batch[k] = value
    #     new_batch['batch_idx'] = list(new_batch['batch_idx'])
    #     for i in range(len(new_batch['batch_idx'])):
    #         new_batch['batch_idx'][i] += i  # add target image index for build_targets()
    #     new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
    #     return new_batch

   
def build_lymo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset"""

    augment_in_training = cfg.augment_in_training if hasattr(cfg, 'augment_in_training') else True
    if mode == 'train':  # train时，根据cfg.augment_in_training决定是否做增强
        augment = augment_in_training
    else:  # val test不会做增强
        augment = False

    rect = cfg.rect or rect  # rectangular batches

    return LYMODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=augment,  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        mode=mode,
        load_correspondence=cfg.load_correspondence if hasattr(cfg, 'load_correspondence') else False,
        )



