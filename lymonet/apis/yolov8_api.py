


try:
    from ultralytics import YOLO
    backend = 'ultralytics'
except:
    
    backend = 'local'

if backend == 'local':
    print('Use local Ultralytics backend. Please install ultralytics in /lymonet. Run: pip install https://github.com/ultralytics/ultralytics.git')
    import os, sys
    from pathlib import Path
    here = Path(__file__).parent
    yolov8_path = f'{here.parent}/ultralytics'
    if yolov8_path not in sys.path:
        sys.path.append(yolov8_path)

    from ..ultralytics.ultralytics import YOLO
    from ..ultralytics.ultralytics.nn.tasks import (
        parse_model, 
        DetectionModel,
        BaseModel,
        v8DetectionLoss,
        yaml_model_load,
        initialize_weights,
        make_divisible,
        colorstr,
        scale_img,
        ClassificationModel,
        )

    from ..ultralytics.ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                        Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                        Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                        RTDETRDecoder, Segment)

    from ..ultralytics.ultralytics.models.yolo.detect.train import (
        DetectionTrainer,
        )

    from ..ultralytics.ultralytics.models.yolo.detect.val import (
        DetectionValidator,
        )

    from ..ultralytics.ultralytics.models.yolo.detect.predict import (
        DetectionPredictor,
        )

    from ..ultralytics.ultralytics.utils import (
        RANK,
        DEFAULT_CFG,
        DEFAULT_CFG_DICT,
        IterableSimpleNamespace,
        callbacks,
        ops,
        )

    from ..ultralytics.ultralytics.utils.torch_utils import (
        de_parallel,
        torch_distributed_zero_first,
    )

    from ..ultralytics.ultralytics.utils.tal import (
        make_anchors,
        dist2bbox,
        TaskAlignedAssigner,
    )

    from ..ultralytics.ultralytics.utils.ops import (
        crop_mask, xywh2xyxy, xyxy2xywh
    )

    from ..ultralytics.ultralytics.utils.loss import (
        BboxLoss,
    )

    from ..ultralytics.ultralytics.utils.instance import (
        Instances,
    )

    from ..ultralytics.ultralytics.data.dataset import (
        YOLODataset,
        BaseDataset,
    )

    from ..ultralytics.ultralytics.data.augment import (
        Compose,
        LetterBox,
        Format,
        Mosaic,
        CopyPaste,
        RandomPerspective,
        MixUp,
        Albumentations,
        RandomHSV,
        RandomFlip,
        v8_transforms,
    )

    from ..ultralytics.ultralytics.utils.metrics import (
        bbox_iou,
        bbox_ioa,
        ClassifyMetrics,
        ConfusionMatrix,
        ap_per_class,
        compute_ap,
        plot_mc_curve,
        plot_pr_curve,
        smooth,
        DetMetrics,
    )


    from ..ultralytics.ultralytics.models.yolo.classify.val import (
        ClassificationValidator,
        )

    from ..ultralytics.ultralytics.models.yolo.classify.train import (
        ClassificationTrainer
        )

    from ..ultralytics.ultralytics.models.yolo.classify.predict import (
        ClassificationPredictor
        )

    from ..ultralytics.ultralytics.engine.validator import (
        BaseValidator,
    )

    from ..ultralytics.ultralytics.cfg import (
        get_save_dir, cfg2dict, check_dict_alignment, LOGGER,
        CFG_FLOAT_KEYS, CFG_FRACTION_KEYS, CFG_INT_KEYS, CFG_BOOL_KEYS,
    )
    from ..ultralytics.ultralytics.utils.checks import check_imgsz

else:
    print('Use Ultralytics backend')
    from ultralytics import YOLO
    from ultralytics.nn.tasks import (
        parse_model, 
        DetectionModel,
        BaseModel,
        v8DetectionLoss,
        yaml_model_load,
        initialize_weights,
        make_divisible,
        colorstr,
        scale_img,
        ClassificationModel,
        )
    from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                        Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                        Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                        RTDETRDecoder, Segment)
    from ultralytics.models.yolo.detect.train import (
        DetectionTrainer,
        )
    from ultralytics.models.yolo.detect.val import (
        DetectionValidator,
        )
    from ultralytics.models.yolo.detect.predict import (
        DetectionPredictor,
        )
    
    from ultralytics.utils import (
        RANK,
        DEFAULT_CFG,
        DEFAULT_CFG_DICT,
        IterableSimpleNamespace,
        callbacks,
        ops,
        )
    from ultralytics.utils.torch_utils import (
        de_parallel,
        torch_distributed_zero_first,
    )
    from ultralytics.utils.tal import (
        make_anchors,
        dist2bbox,
        TaskAlignedAssigner,
    )
    from ultralytics.utils.ops import (
        crop_mask, xywh2xyxy, xyxy2xywh
    )
    from ultralytics.utils.loss import (
        BboxLoss,
    )
    from ultralytics.utils.instance import (
        Instances,
    )
    from ultralytics.data.dataset import (
        YOLODataset,
        BaseDataset,
    )
    from ultralytics.data.augment import (
        Compose,
        LetterBox,
        Format,
        Mosaic,
        CopyPaste,
        RandomPerspective,
        MixUp,
        Albumentations,
        RandomHSV,
        RandomFlip,
        v8_transforms,
    )
    from ultralytics.utils.metrics import (
        bbox_iou,
        bbox_ioa,
        ClassifyMetrics,
        ConfusionMatrix,
        ap_per_class,
        compute_ap,
        plot_mc_curve,
        plot_pr_curve,
        smooth,
        DetMetrics,
    )
    from ultralytics.models.yolo.classify.val import (
        ClassificationValidator,
        )
    from ultralytics.models.yolo.classify.train import (
        ClassificationTrainer
        )
    from ultralytics.models.yolo.classify.predict import (
        ClassificationPredictor
        )
    from ultralytics.engine.validator import (
        BaseValidator,
    )
    from ultralytics.cfg import (
        get_save_dir, cfg2dict, check_dict_alignment, LOGGER,
        CFG_FLOAT_KEYS, CFG_FRACTION_KEYS, CFG_INT_KEYS, CFG_BOOL_KEYS,
        )
    from ultralytics.utils.checks import check_imgsz


# Path: lymonet/apis/yolov8_api.py  
