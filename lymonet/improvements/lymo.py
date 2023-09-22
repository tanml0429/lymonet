
from pathlib import Path
from lymonet.apis import YOLO
from .nn.tasks import LymoDetectionModel

class LYMO(YOLO):
    
    def __init__(self, model: str | Path = 'yolov8n.pt', task=None) -> None:
        super().__init__(model, task)
    
    
    @property
    def task_map(self):
        task_map = super().task_map
        task_map['detect']['model'] = LymoDetectionModel
        return task_map
    
    
    @staticmethod
    def apply_improvements():
        from .nn.nn import ResBlock_CBAM, CBAM, ChannelAttentionModule, SpatialAttentionModule, RecoveryBlock
        # globals().update(locals())
        globals()['RecoveryBlock'] = RecoveryBlock
        print('apply_improvements')

