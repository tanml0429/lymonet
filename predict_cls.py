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
    # model = LYMO(model_name_or_cfg)
    model = YOLO(model_name_or_cfg)

    if model_weights:
        model = model.load(model_weights)
    
    # model = YOLO(model_name).load(model_weights)

    results = model.predict(**kwargs)
    print(results)

    # Evaluate the model's performance on the validation set
    # results = model.val()
    # print(results)

    # Perform object detection on an image using the model
    # results = model(f'{here}/lymonet/data/scripts/image.png')
    # print(results)

    # Export the model to ONNX format
    # success = model.export(format='onnx')

@dataclass
class Args:
    model: str =  '/home/tml/VSProjects/lymonet/runs/classify/yolov8s_aug1_valsquare1/weights/best.pt'
    mode: str = 'predict'
    task: str = 'classify'
    source: str = '/data/tml/lymonet/lymo_yolo_aug1.1/test/normal'
    save: bool = True
    # show_labels: bool = True
    # show_conf: bool = True
    device: str = '0'  # GPU id 
    name: str = 'classifypredict'
    # save_txt: bool = True
    # visualize: bool = True
    
 
if __name__ == '__main__':
    args = hai.parse_args_into_dataclasses(Args)
    run(args)
