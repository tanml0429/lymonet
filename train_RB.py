

import os, sys
from pathlib import Path
here = Path(__file__).parent
p = f'{here.parent}'
if p not in sys.path:
    sys.path.append(p)
from lymonet.apis import YOLO, LYMO
from dataclasses import dataclass, field
import hai


def run(args):
    # Create a new YOLO model from scratch
    # model_name = args.pop('model')
    kwargs = args.__dict__
    model_name_or_cfg = kwargs.pop('model')
    model_weights = kwargs.pop('weights', None)
    LYMO.apply_improvements()
    model = LYMO(model_name_or_cfg, task=args.task)
    # model = YOLO(model_name_or_cfg, task=args.task)

    if model_weights:
        model = model.load(model_weights)
    
    # model = YOLO(model_name).load(model_weights)

    freeze = kwargs.pop('freeze', '')
    freeze = [x.strip() for x in freeze.split(',') if x]
    kwargs['freeze'] = freeze

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
    model: str =  f'{here}/lymonet/configs/yolov8s_1MHSA_CA_RB.yaml'
    # model: str =  f'{here}/lymonet/configs/yolov8s_1MHSA_CA.yaml'
    # model: str = "yolov8x.yaml"
    # weights: str = 'runs/detect/LYMO_MHSA_CA_RB14/weights/best.pt'
    data: str = f'{here}/lymonet/configs/lymo_mixed.yaml'
    # data: str = f'{here}/lymonet/configs/lymo_minimal.yaml'
    epochs: int = 300
    batch: int = 1
    imgsz: int = 640
    workers: int = 1
    device: str = '0'  # GPU id 
    name: str = 'LYMO_MHSA_CA_RB'
    patience: int = 0
    dropout: float = 0.5
    task: str = 'detect'
    # freeze: str = '0'  # freeze layer 0,1,2,3 etc
    # freeze: str = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22"
    box: float = 7.5  # 7.5 bbox loss gain  
    cls: float = 0.5  # 0.5 cls loss gain
    dfl: float = 1.5  # 1.5 dfl loss gain
    content_loss_gain: float = 0.0
    texture_loss_gain: float = 0.0
    augment_in_training: bool = True  # Default True, Indicates whether to augment in training
    load_correspondence: bool = True  # Default False, Indicates whether to load correspondence images, need `metadata.json` file in YOLO dataset
    
 
if __name__ == '__main__':
    args = hai.parse_args_into_dataclasses(Args)
    # print(f'args: {args}')
    # exit()
    run(args)
