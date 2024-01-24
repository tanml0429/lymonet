

from lymonet.apis.yolov8_api import DetectionPredictor, RANK
from .nn.tasks import LymoDetectionModel

    
class LymoDetectionPredictor(DetectionPredictor):

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = LymoDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)

        if weights:
            model.load(weights)
        return model