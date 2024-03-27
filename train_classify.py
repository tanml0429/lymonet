

import os, sys
from pathlib import Path
here = Path(__file__).parent
p = f'{here.parent}'
if p not in sys.path:
    sys.path.append(p)
from lymonet.apis import YOLO, LYMO
from dataclasses import dataclass, field
import hai

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def run(args):
    # Create a new YOLO model from scratch
    # model_name = args.pop('model')
    kwargs = args.__dict__
    model_name_or_cfg = kwargs.pop('model')
    model_weights = kwargs.pop('weights', None)
    # LYMO.apply_improvements()
    model = LYMO(model_name_or_cfg, task=args.task)
    # model = YOLO(model_name_or_cfg, task=args.task)

    # LYMO.apply_improvements()

    if model_weights:
        model = model.load(model_weights)
    
    # model = YOLO(model_name).load(model_weights)

    # freeze = kwargs.pop('freeze', '')
    # freeze = [x.strip() for x in freeze.split(',') if x]
    # kwargs['freeze'] = freeze

    results = model.train(**kwargs)

    # Evaluate the model's performance on the validation set
    results = model.val()
    print(results)

    # Perform object detection on an image using the model
    results = model(f'{here}/lymonet/data/scripts/image.png')
    print(results)

    # Export the model to ONNX format
    # success = model.export(format='onnx')

@dataclass
class Args:
    mode: str = 'train'
    model: str =  f'{here}/lymonet/configs/classification/yolov8s-cls-lymo_CA_MHSA.yaml'
    # model: str = f"yolov8s-cls.yaml"
    # weights: str = 'yolov8s-cls.pt'
    # data: str = '/data/tml/lymonet/lymo_cls_mini'
    data: str = '/data/tml/lymonet/lymo_yolo_aug1.1'
    # data: str = 'mnist'
    epochs: int = 300
    batch: int = 80
    imgsz: int = 640
    workers: int = 20
    device: str = '0'  # GPU id 
    name: str = 'yolov8s_class'
    patience: int = 0
    # dropout: float = 0.5
    task: str = 'classify'
    # content_loss_gain: float = 1.0
    # texture_loss_gain: float = 1.0
    cls_loss_gain: float = 1.0
    echo_loss_gain: float = 0.0
    
 
if __name__ == '__main__':
    args = hai.parse_args_into_dataclasses(Args)
    run(args)
