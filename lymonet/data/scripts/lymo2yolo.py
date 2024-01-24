"""
将lymo格式的数据集转换为yolo格式的数据集
"""
import json
import os, sys
from pathlib import Path
import shutil
import warnings
import damei as dm
from dataclasses import dataclass, field


here = Path(__file__).parent


class LYMO2YOLO:
    
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.data = self.load_json()
        
    @property
    def metadata(self):
        return self.data['metadata']
        
    def load_json(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def save_yolo_metadata(self, output_dir, **kwargs):
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
        

        save_corresponding_cdfi = kwargs.pop('save_corresponding_cdfi', False)
        if save_corresponding_cdfi:
            correspondence = dict()
            # 保存对应关系，键是us的图像名字，值是对应的cdfi的图像名字，1对多

            for i, patient in enumerate(self.data['entries']):
                patient_id = patient['id']
                us = patient['US']
                cdfi = patient['CDFI']
                for j, im_info in enumerate(us):
                    us_img_id = im_info['img_id']
                    us_img_path = im_info['img_path']
                    us_stem = Path(us_img_path).stem
                    cors_cdfi_ids = im_info.get("corresponding_CDFI_img_ids", [])
                    # assert len(cors_cdfi_ids) != 0, f"corresponding_CDFI_img_ids is empty, patient_id: {patient_id}, img_id: {us_img_id}"
                    # if len(cors_cdfi_ids) == 0:
                        # warnings.warn(f"corresponding_CDFI_img_ids is empty, patient_id: {patient_id}, img_id: {us_img_id}")

                    cors_cdfi_stems = []
                    for k, cdfi_im_info in enumerate(cdfi):
                        cdfi_id = cdfi_im_info['img_id']
                        if cdfi_id in cors_cdfi_ids:
                            cdfi_stem = Path(cdfi_im_info['img_path']).stem
                            cors_cdfi_stems.append(cdfi_stem)
                    assert len(cors_cdfi_stems) == len(cors_cdfi_ids), f"len(cors_cdfi_stems) != len(cors_cdfi_ids), patient_id: {i}, img_id: {j}"

                    correspondence[str(us_stem)] = cors_cdfi_stems
            print(f'Number of correspondence: {len(correspondence)}')
            metadata['correspondence'] = correspondence

        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        pass
        
    
    def lymo2yolo(self, **kwargs):
        """lymo格式转换为yolo格式"""
        
        output_folder = kwargs.pop('output_folder', None)
        p = Path(self.file_path)
        output_dir = f"{p.parent.parent}/{output_folder}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 先保存label.txt
        classes = self.metadata['classes']
        classes = [f"{i} {x}" for i, x in enumerate(classes)]
        classes_str = "\n".join(classes)
        with open(f"{output_dir}/label.txt", 'w') as f:
            f.write(classes_str)
        # 保存yolo格式的metadata
        self.save_yolo_metadata(output_dir, **kwargs)
        
        # 保存图片和标注
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


                    img_stem = Path(img_path).stem  # 文件名字
                    shutil.copy(img_path, f"{save_img_dir}/{img_stem}.png")
                    with open(f"{save_label_dir}/{img_stem}.txt", 'w') as f:
                        f.write(anno_str)
                print(f"\r[{k}] [{i+1}/{len(v)}]", end="")
                pass
            print()
        print(f"Data saved to {output_dir}")
        pass
        

    def __call__(self, **kwargs):
        self.lymo2yolo(**kwargs)
        pass
    
@dataclass
class Args:
    file_path: str = '/data/tml/lymonet/lymo_dataset_unclipped/lymo_all.json'
    output_folder: str = 'lymo_yolo_unclipped'
    save_corres: bool = True

if __name__ == "__main__":
    import hai
    args = hai.parse_args_into_dataclasses(Args)
    c = LYMO2YOLO(file_path=args.file_path)
    c(output_folder=args.output_folder, 
      save_corresponding_cdfi=args.save_corres)
