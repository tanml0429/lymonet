

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
    model_name = kwargs.pop('model')
    model_weights = kwargs.pop('weights')
    
    # LYMO.apply_improvements()
    model = LYMO(model_name).load(model_weights)
    
    # model = YOLO(model_name).load(model_weights)

    results = model.train(**kwargs)

    # Evaluate the model's performance on the validation set
    results = model.val()
    print(results)

    # Perform object detection on an image using the model
    results = model(f'{here}/lymonet/data/patient0002_node_1_CDFI__3_4048.png')
    print(results)

    # Export the model to ONNX format
    # success = model.export(format='onnx')

@dataclass
class Args:
    model: str =  f'{here}/lymonet/configs/yolov8s_1MHSA_1BF_CR.yaml'
    weights: str = 'yolov8x.pt'
    data: str = f'{here}/lymonet/configs/lymo_mixed.yaml'
    epochs: int = 300
    batch: int = 32
    imgsz: int = 640
    workers: int = 16
    device: str = '7'  # GPU id 
    name: str = 'lymonet'
    patience: int = 0
    
 
if __name__ == '__main__':
    args = hai.parse_args_into_dataclasses(Args)
    run(args)
