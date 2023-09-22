"""
将lymo格式的数据集转换为yolo格式的数据集
"""
import json
import os, sys
from pathlib import Path
import shutil
import damei as dm


here = Path(__file__).parent


class Converter:
    
    def __init__(self) -> None:
        self.file_path = f'/data/tml/lymonet/lymo_dataset/lymo_all.json'
        self.data = self.load_json()
        
    @property
    def metadata(self):
        return self.data['metadata']
        
    def load_json(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def save_yolo_metadata(self, output_dir):
        metadata = dict()
        metadata["version"] = "1.0.0"
        metadata["name"] = "lymo_yolo"
        metadata['description'] = """
由lymo_all.json切分成train/val/test三个子集，并转化为yolo格式保存下来，三个自己不重叠。
images文件夹内保存图片，labels文件夹内保存标注文件。
标注格式为：class_id x_center y_center width height，每行代表一个目标。
        """
        metadata['classes'] = self.metadata['classes']
        metadata['num_of_train_images'] =self.metadata['splited_data']["num_of_train_images"]
        metadata['num_of_val_images'] = self.metadata['splited_data']["num_of_val_images"]
        metadata['num_of_test_images'] = self.metadata['splited_data']["num_of_test_images"]
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        pass
        
    
    def lymo2yolo(self):
        """lymo格式转换为yolo格式"""
        p = Path(self.file_path)
        output_dir = f"{p.parent.parent}/lymo_yolo"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 先保存label.txt
        classes = self.metadata['classes']
        classes = [f"{i} {x}" for i, x in enumerate(classes)]
        classes_str = "\n".join(classes)
        with open(f"{output_dir}/label.txt", 'w') as f:
            f.write(classes_str)
            
        self.save_yolo_metadata(output_dir)
            
        exit()
            
        train_ids = self.metadata['splited_data']['train_ids']
        val_ids = self.metadata['splited_data']['val_ids']
        test_ids = self.metadata['splited_data']['test_ids']
        
        all_ids = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        for k, v in all_ids.items():
            save_img_dir = f"{output_dir}/images/{k}"
            save_label_dir = f"{output_dir}/labels/{k}"
            os.makedirs(save_img_dir, exist_ok=True)
            os.makedirs(save_label_dir, exist_ok=True)
            for i, patient_id in enumerate(v):
                patient = self.data['entries'][int(patient_id)]
                
                imgs = patient['US'] + patient['CDFI']
                for j, img in enumerate(imgs):
                    img_path = f"{p.parent}/{img['img_path']}"
                    annotation = img['annotation']
                    anno_str = "\n".join(annotation)
                    
                    img_stem = Path(img_path).stem
                    # 保存
                    shutil.copy(img_path, f"{save_img_dir}/{img_stem}.png")
                    with open(f"{save_label_dir}/{img_stem}.txt", 'w') as f:
                        f.write(anno_str)
                print(f"\r[{k}] [{i+1}/{len(v)}]", end="")
                pass
        print()
        pass
        
        
    
    def __call__(self):
        self.lymo2yolo()
        pass


if __name__ == "__main__":
    c = Converter()
    c()