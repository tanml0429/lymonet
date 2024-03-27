

import torch
import numpy as np
import damei as dm

def preprocess_correspondence(batch, self):
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
        # 连续
        cors_img = cors_img.to(self.device, non_blocking=True).float()
        batch['cors_img'] = cors_img
    return batch

def warp_bbox(bbox, H):
    """
    bbox的格式是xc, yc, w, h
    """
    x_center, y_center, w, h = bbox
    half_w, half_h = w / 2, h / 2
    points = np.array([
            [x_center - half_w, y_center - half_h, 1],
            [x_center + half_w, y_center - half_h, 1],
            [x_center + half_w, y_center + half_h, 1],
            [x_center - half_w, y_center + half_h, 1]
        ], dtype=np.float32).T  # 转置以便于矩阵乘法
    # 应用单应性矩阵变换
    transformed_points = H.dot(points)  # 4个点

    # 将齐次坐标转换回 (x, y) 坐标
    transformed_points /= transformed_points[2]

    # 找到变换后的边界框的新坐标
    min_x = np.min(transformed_points[0])
    min_y = np.min(transformed_points[1])
    max_x = np.max(transformed_points[0])
    max_y = np.max(transformed_points[1])
    # transformed_bbox_xyxy = [min_x, min_y, max_x, max_y]

    transformed_bbox_xywh = [
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
        max_x - min_x,
        max_y - min_y,
    ]
    return transformed_bbox_xywh

def warp_bboxes(bboxes, H):
    """
    对BBoxes应用单应性矩阵进行变换
    """
    new_bboxes = []
    for bbox in bboxes:
        new_bboxes.append(warp_bbox(bbox, H))
    return np.array(new_bboxes)


def xywh2two_points(bbox):
    """
    将bbox的格式从xywh转换为两个点的坐标
    """
    x_center, y_center, w, h = bbox
    half_w, half_h = w / 2, h / 2
    lt = (x_center - half_w, y_center - half_h)  # Left Top
    rb = (x_center + half_w, y_center + half_h)  # Right Bottom
    return [lt, rb]


def xywh2four_points(bbox):
    """
    将bbox的格式从xywh转换为四个点的坐标
    """
    x_center, y_center, w, h = bbox
    half_w, half_h = w / 2, h / 2
    lt = (x_center - half_w, y_center - half_h)  # Left Top
    rt = (x_center + half_w, y_center - half_h)  # Right Top
    rb = (x_center + half_w, y_center + half_h)  # Right Bottom
    lb = (x_center - half_w, y_center + half_h)  # Left Bottom
    return [lt, rt, rb, lb]

def denormalize(img, bboxes):
    h, w = img.shape[:2]
    scale = np.array([w, h, w, h], dtype=np.float32)
    bboxes[:, 0] *= scale[0]
    bboxes[:, 1] *= scale[1]
    bboxes[:, 2] *= scale[2]
    bboxes[:, 3] *= scale[3]
    return bboxes

def normalize(img, bboxes):
    h, w = img.shape[:2]
    scale = np.array([w, h, w, h], dtype=np.float32)
    bboxes[:, 0] /= scale[0]
    bboxes[:, 1] /= scale[1]
    bboxes[:, 2] /= scale[2]
    bboxes[:, 3] /= scale[3]
    return bboxes

def plot_bboxes(img, bboxes):
    """
    输入: bboxes in xywh
    """
    bboxes_xyxy = dm.general.xywh2xyxy(bboxes)
    for bbox in bboxes_xyxy:
        img = dm.general.plot_one_box_trace_pose_status(bbox, img)
    return img
            

def restore_org_img(img, grey=128):
    """从使用灰色背景的图片中恢复出原始的图片"""
    mask = np.all(img != [grey, grey, grey], axis=-1)  # 必须是三个通道都不是128才是mask
    coords = np.column_stack(np.where(mask))

    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)

    org_img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
    return org_img
            