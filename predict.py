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
    # LYMO.apply_improvements()
    model = LYMO(model_name_or_cfg)

    if model_weights:
        model = model.load(model_weights)
    
    # model = YOLO(model_name).load(model_weights)

    results = model.predict(**kwargs)

    # Evaluate the model's performance on the validation set
    # results = model.val()
    # print(results)

    # Perform object detection on an image using the model
    results = model(f'{here}/lymonet/data/scripts/image.png')
    print(results)

    # Export the model to ONNX format
    # success = model.export(format='onnx')

@dataclass
class Args:
    mode: str = 'predict'
    model: str =  '/home/tml/VSProjects/LymoNet/runs/detect/yolov8x/weights/best.pt'
    source: str = '/data/tml/lymonet/lymo_yolo_unclipped/images/val'
    save: bool = True
    show_labels: bool = True
    show_conf: bool = True
    device: str = '2'  # GPU id 
    name: str = 'lymonetval'
    save_txt: bool = True
    # visualize: bool = True
    
 
if __name__ == '__main__':
    args = hai.parse_args_into_dataclasses(Args)
    run(args)
