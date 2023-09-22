
import os, sys
from pathlib import Path
import glob
import json
import shutil
import cv2
import random
import numpy as np


class LymphNodeDatasetAnalysis:
    
    def __init__(self) -> None:
        self.dataset_dir = f'/data/tml/lymonet/lymph_node_dataset'
        self.lymo_path = f'/data/tml/lymonet/lymo_dataset/lymo_all.json'
        self._data = self.load_data_from_lymo()
        
    @property
    def data(self):
        if not self._data:
            self._data = self.load_data_from_lymo()
        return self._data
    
    @property
    def metadata(self):
        return self.data['metadata']
    
    def load_data_from_lymo(self, **kwargs):
        file_path = self.lymo_path
        if not os.path.exists(file_path):
            data = dict()
            data["version"] = "1.0.0"
            data["metadata"] = dict()
            data["metadata"]["description"] = """
Lymph Node Dataset. The US is the B-Mode ultrasound image, the CDFI is the Color Doppler Flow Imaging image.
每个病人0到n张B超图，0到m张CDFI图。
每张图中包含1到a个淋巴结，每个淋巴结有1个标注框。
"""
            data["metadata"]["author"] = "Menglu Tan"
            data["metadata"]["misc"] = dict()
            data["metadata"]["misc"].update(kwargs)
            data["entries"] = []
            self.save2file(data)
            return self.load_data_from_lymo()
        else:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
    
    def save2file(self, data=None):
        data = data or self.data
        if not os.path.exists(self.lymo_path):
            os.makedirs(Path(self.lymo_path).parent, exist_ok=True)
        with open(self.lymo_path, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    
    def read_one_data(self, patient):
        """根据patient读取一条数据"""
        entries = self.data['entries']
        for idx, entry in enumerate(entries):
            if entry['patient'] == patient:
                return idx, entry
        one_data = dict()
        one_data['id'] = f'{len(entries):06d}'
        one_data['patient'] = patient
        one_data["US"] = list()
        one_data["CDFI"] = list()
        return -1, one_data
    
    def read_annotation(self, img_path):
        label_path = img_path.replace('images', 'labels').replace('.png', '.txt')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f'{label_path} not found')
        with open(label_path, 'r') as f:
            lines = f.readlines()
        annotations = list()
        classes = []
        xcycwhs = []
        for line in lines:
            line = line.strip().split()
            classes.append(line[0])
            xcycwhs.append([float(x) for x in line[1:]])
            xcycwh_str = ' '.join([f'{x:.5f}' for x in xcycwhs[-1]])
            annotations.append(f'{line[0]} {xcycwh_str}')
        return annotations
    
    def stem2nodes(self, stem, num_nodes, file_path=None):
        """文件名转换为淋巴结的信息"""
        nodes = []
        import re
        # 匹配node_1&2_CDFI, node_2, node_3, ...
        if "_&_" in stem:
            stems = stem.split('_&_')
        else:
            stems = [stem]
        for stem in stems:
            pattern = re.search(r'node_\d+&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*&?\d*', stem)
            try:
                span = pattern.span()  # 开始和结束的位置
                string = pattern.group()  # 匹配到的字符串
                
                if "&" in string:
                    _nodes = string.split('&')
                    for i, _n in enumerate(_nodes):
                        if "node_" not in _n:
                            _nodes[i] = f'node_{_n}'
                else:
                    _nodes = [string]
                nodes.extend(_nodes)
            except Exception as e:
                # print(f'Error: {e}')
                ipt = input(f'Error: {e}, stem = {stem}, num_nodes = {num_nodes}, file_path = {file_path}, 是否移除到error_data并继续？(y/n)')
                if ipt in ['y', 'Y', 'yes', 'Yes', 'YES', '']:
                    p = f'{Path(self.lymo_path).parent.parent}/error_data/文件名错误'
                    shutil.move(file_path, f'{p}/{stem}.png')
                    txt_path = file_path.replace('images', 'labels').replace('.png', '.txt')
                    shutil.move(txt_path, f'{p}/{stem}.txt')
                    return None
                else:
                    raise e
                
        if len(nodes) != num_nodes:
            if len(nodes) < num_nodes:
                # 不匹配，把有错误的数据放到error_data里
                p = f'{Path(self.lymo_path).parent.parent}/error_data/标注过多'
            elif len(nodes) > num_nodes:
                p = f'{Path(self.lymo_path).parent.parent}/error_data/标注过少'
            shutil.move(file_path, f'{p}/{stem}.png')
            txt_path = file_path.replace('images', 'labels').replace('.png', '.txt')
            shutil.move(txt_path, f'{p}/{stem}.txt')
            return None
        assert len(nodes) == num_nodes, f'len(nodes) = {len(nodes)}, num_nodes = {num_nodes}'
        return nodes
        
    def read_classes(self):
        file_path = f'{self.dataset_dir}/labels.txt'
        with open(file_path, 'r') as f:
            lines = f.readlines()
        classes = [x.strip().split()[-1] for x in lines]
        
        return classes
    
    def from_yolo_format2lymo_format(self):
        """将yolo格式的数据转换为lymo格式的数据"""
        
        classes = self.read_classes()
        self.data['metadata']['classes'] = classes
        
        file_paths = glob.glob(f'{self.dataset_dir}/**/*.png', recursive=True)
        lymo_path = Path(self.lymo_path)
        
        patients = []
        
        data = self.data
        
        file_paths = sorted(file_paths)
        for i, file_path in enumerate(file_paths):
            p = Path(file_path)
            print(f'\r{i+1}/{len(file_paths)} {p.stem:>30s}', end='')
            
            # h, w, c = 0, 0, 0
            patient = p.stem.split('_')[0]  # 
           
            patients.append(patient)
            
    
            is_us = p.parent.name == 'US'
            is_cdfi = p.parent.name == 'CDFI'
            assert not (is_us and is_cdfi), f'File name {p.stem} contains both US and CDFI'
            assert is_us or is_cdfi, f'File name {p.stem} does not contain US or CDFI'
            
            idx, one_data = self.read_one_data(patient)  # 一个病人的所有数据
            
            img_type = 'US' if is_us else 'CDFI'
            img_path = f"entries/{patient}/{p.name}"
            annotation = self.read_annotation(file_path)
            nodes = self.stem2nodes(p.stem, len(annotation), file_path=file_path)
            
            if nodes is None:
                continue
            exist_paths = [x['img_path'] for x in one_data[img_type]]
            if img_path in exist_paths:
                continue
            img = cv2.imread(file_path)
            h, w, c = img.shape
            
            one_img = dict()
            one_img['img_type'] = img_type
            one_img['img_path'] = img_path
            one_img['height'] = h
            one_img['width'] = w
            one_img['annotation'] = annotation
            one_img['nodes'] = nodes
            
            # 保存lymo_all.json文

            one_data[img_type].append(one_img)
            if idx == -1:
                data['entries'].append(one_data)
            else:
                data['entries'][idx] = one_data
            # self.save2file(data)
            # break
            
            # 保存实际文件
            save_dir = f'{lymo_path.parent}/entries/{patient}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            save_path = f'{save_dir}/{p.name}'
            if os.path.exists(save_path):
                continue
            shutil.copy(file_path, save_path)
        print()

        self.save2file(data)
     
    def perform_statistics(self):
        """在已经有数据的情况下，统计数据集的信息，存储到metadata中"""
        data = self.data
        entries = data['entries']
        metadata = data['metadata']
        
        num_of_US_images = sum([len(x["US"]) for x in entries])
        num_of_CDFI_images = sum([len(x["CDFI"]) for x in entries])
        num_of_lymph_nodes_in_US = 0
        num_of_lymph_nodes_in_CDFI = 0
        num_targets_of_each_class = [0] * len(metadata['classes'])
        for entry in entries:
            for img in entry['US']:
                num_of_lymph_nodes_in_US += len(img['annotation'])
            for img in entry['CDFI']:
                num_of_lymph_nodes_in_CDFI += len(img['annotation'])
            annos = [x for img in (entry['US'] + entry['CDFI']) for x in img['annotation']]
            clss = [int(x.split()[0]) for x in annos]
            for cls in clss:
                num_targets_of_each_class[cls] += 1
        total_images = num_of_US_images + num_of_CDFI_images
        total_lymph_nodes = num_of_lymph_nodes_in_US + num_of_lymph_nodes_in_CDFI
        average_lymph_nodes_per_image = total_lymph_nodes / total_images
        
        metadata['num_of_patients'] = len(entries)
        metadata['num_of_US_images'] = num_of_US_images
        metadata['num_of_CDFI_images'] = num_of_CDFI_images
        metadata['num_of_lymph_nodes_in_US'] = num_of_lymph_nodes_in_US
        metadata['num_of_lymph_nodes_in_CDFI'] = num_of_lymph_nodes_in_CDFI
        metadata['total_images'] = total_images
        metadata['total_lymph_nodes'] = total_lymph_nodes
        metadata['average_lymph_nodes_per_image'] = round(average_lymph_nodes_per_image, 2)
        metadata['num_targets_of_each_class'] = num_targets_of_each_class
        
        self.save2file(data)
    
    def split_into_train_val_test(self):
        """按病人划分数据集，存储到metadata中"""
        ratio = [0.7, 0.15, 0.15]
        random.seed(429)
        data = self.data
        entries = data['entries']
        entry_ids = [x['id'] for x in entries]
        random.shuffle(entry_ids)
        
        train_ids = entry_ids[:int(len(entry_ids) * ratio[0])]
        val_ids = entry_ids[int(len(entry_ids) * ratio[0]):int(len(entry_ids) * (ratio[0] + ratio[1]))]
        test_ids = entry_ids[int(len(entry_ids) * (ratio[0] + ratio[1])):]
        
        train_ids = sorted(train_ids)
        val_ids = sorted(val_ids)
        test_ids = sorted(test_ids)
        
        split_data = dict()
        split_data['train_val_test_ratio'] = ratio
        split_data['num_of_train_patient'] = len(train_ids)
        split_data['num_of_val_patient'] = len(val_ids)
        split_data['num_of_test_patient'] = len(test_ids)
        split_data['num_of_train_images'] = sum([len(entries[int(x)]['US']) + len(entries[int(x)]['CDFI']) for x in train_ids])
        split_data['num_of_val_images'] = sum([len(entries[int(x)]['US']) + len(entries[int(x)]['CDFI']) for x in val_ids])
        split_data['num_of_test_images'] = sum([len(entries[int(x)]['US']) + len(entries[int(x)]['CDFI']) for x in test_ids])
        split_data['train_ids'] = train_ids
        split_data['val_ids'] = val_ids
        split_data['test_ids'] = test_ids
        
        self.data["metadata"].pop('split_data', None)
        self.data["metadata"]['splited_data'] = split_data
        self.save2file()
        
    def add_img_id(self):
        """为每张图添加img_id"""
        entries = self.data['entries']
        for i, entry in enumerate(entries):
            count = 0
            
            new_us = []
            for j, img in enumerate(entry['US']):
                _new = dict()
                _new['img_id'] = f'{count:06d}'
                img.pop('img_id', None)
                _new.update(img)
                count += 1
                new_us.append(_new)
            entry['US'] = new_us
            
            new_cdfi = []
            for j, img in enumerate(entry['CDFI']):
                _new = dict()
                _new['img_id'] = f'{count:06d}'
                img.pop('img_id', None)
                _new.update(img)
                count += 1
                new_cdfi.append(_new)
            entry['CDFI'] = new_cdfi
        self.save2file()
            
        
    def search_paired_images(self):
        """
        逐张分析US图，搜索该病人对应的CDFI图，存储到每个条目的每个图的信息中
        """
        self.add_img_id()
        
        entries = self.data['entries']
        num_of_pairs = 0
        for i, entry in enumerate(entries):
            patient = entry['patient']
            us_imgs = entry['US']
            cdfi_imgs = entry['CDFI']
            for j, us_img in enumerate(us_imgs):
                corres_cdfi_ids = self._search(us_img, cdfi_imgs)
                us_img.pop('corresponding_CDFI_ids', None)
                us_img['corresponding_CDFI_img_ids'] = corres_cdfi_ids
                num_of_pairs += len(corres_cdfi_ids)
                print(f'\r{i+1}/{len(entries)} {patient:>30s} {j+1}/{len(us_imgs)} {us_img["img_id"]:>6s}', end='')
                pass
            
        self.data['metadata']['num_of_pairs'] = num_of_pairs
        self.save2file()
        
        print()
    
    def _search(self, us_img, cdfi_imgs):
        us_stem = Path(us_img['img_path']).stem
        classes = self.metadata['classes']
        b_nodes = us_img['nodes']
        b_clss = [classes[int(x.split()[0])] for x in us_img['annotation']]
        
        searched = []
        
        # 先用名字搜索一下，b_nodes和b_clss都相同
        for k, cdfi_img in enumerate(cdfi_imgs):
            c_nodes = cdfi_img['nodes']
            c_clss = [classes[int(x.split()[0])] for x in cdfi_img['annotation']]
            if b_nodes != c_nodes:
                continue
            if set(b_clss) != set(c_clss):
                continue
            searched.append(cdfi_img)
            
        # TODO：可能会搜到0，或者多个，根据iou搜索一下
    
        # assert len(searched) == 1, f'len(searched) = {len(searched)}'
        searched_ids = [x['img_id'] for x in searched]
        return searched_ids
        
        
    def __call__(self):
        # self.from_yolo_format2lymo_format()
        # self.perform_statistics()
        
        self.split_into_train_val_test()
        
        # self.search_paired_images()
        
        
        
    
if __name__ == '__main__':
    ln = LymphNodeDatasetAnalysis()
    ln()