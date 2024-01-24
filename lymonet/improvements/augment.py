
import numpy as np
from lymonet.apis.yolov8_api import (
    Compose,
    Mosaic,
    CopyPaste,
    RandomPerspective,
    LetterBox,
    MixUp,
    Albumentations,
    RandomHSV,
    RandomFlip,
    Instances,
    Format,
    v8_transforms,
    )

import damei as dm
LOGGER = dm.get_logger('augment.py')


class LymoMosaic(Mosaic):
    """修改Mosaic，保存corresponding image的信息"""

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        super().__init__(dataset, imgsz, p, n)
    
    # 重写此方法
    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        final_labels = super()._cat_labels(mosaic_labels)
        final_labels['correspondence'] = mosaic_labels[0].get('correspondence', None)
        return final_labels
    
class LymoFormat(Format):
    
    def __call__(self, labels):
        labels = super().__call__(labels)
        cors = labels.get('correspondence', None)
        if cors:
            cors_labels = super().__call__(cors)
            labels['correspondence'] = cors_labels
        return labels

class LymoLetterBox(LetterBox):
    
    def __call__(self, labels=None, image=None):
        labels = super().__call__(labels, image)
        cors = labels.get('correspondence', None)
        if cors:
            cors_labels = super().__call__(cors)
            labels['correspondence'] = cors_labels
        return labels

## 对应于data.autoaugment.py: v8_transforms, 为适配RecoveryBlock修改
def lymo_transforms(dataset, imgsz, hyp, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose([
        LymoMosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
        CopyPaste(p=hyp.copy_paste),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
        )])
    flip_idx = dataset.data.get('flip_idx', [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    return Compose([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])  # transforms