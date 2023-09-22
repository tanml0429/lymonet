"""
执行ROI
"""
import json
import cv2
from pathlib import Path

class LymoROI(object):
    
    def __init__(self) -> None:
        self.file_path = f'/data/tml/lymonet/lymo_dataset_unclipped/lymo_all.json'
        self.data = self.load_json()
        
    @property
    def metadata(self):
        return self.data['metadata']
        
    def load_json(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data

    def __call__(self):
        self.roi()
    
    def roi(self, plot=False):
        entries = self.data['entries']
        img_dir = f'{Path(self.file_path).parent}'
        
        for i, entry in enumerate(entries):
            us_or_cdfi = ["US", "CDFI"]
            for us_cdfi in us_or_cdfi:
                images_info = entry[us_cdfi]
                for j, img_info in enumerate(images_info):
                    img_path = f'{img_dir}/{img_info["img_path"]}'
                    # raw_img = cv2.imread(img_path)
                    raw_h = img_info['height']
                    raw_w = img_info['width']
                    raw_annos = img_info['annotation']
                    for k, raw_anno in enumerate(raw_annos):
                        cls, bxc, byc, bw, bh = raw_anno.split()  # class, box_center_x, box_center_y, box_width, box_height
                        if plot:
                            pass
                        pass
                    self.roi_by_mouse(img_path)
        pass
    
    def roi_by_mouse(self, img_path):
        img = cv2.imread(img_path)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.on_mouse, img)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        
    # def on_mouse(self):
        
        
    
    
if __name__ == '__main__':
    LymoROI()()
        


