# 新配置继承了基本配置，并做了必要的修改
_base_ = '../mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3), mask_head=dict(num_classes=3)))

# 修改数据集相关配置
data_root = '/data/tml/lymonet/lymo_coco_unclipped/'
metainfo = {
    'classes': ('metastatic', 'inflammatory', 'normal' ),
    'palette': [
        (106, 0, 228), 
        (119, 11, 32), 
        (165, 42, 42),
    ]
}
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = val_evaluator

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
