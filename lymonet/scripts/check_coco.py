"""
check coco dataset
"""
import json
import random
import os
import argparse
from pathlib import Path
# import cv2
try:
	import cv2
except:
	pass
import warnings

from damei.functions import general
from damei.controls.color import ColorControl


class CheckCOCO(object):
	def __init__(self, json_path, img_dir=None, line_thickness=3):
		self.jp = json_path  # json annotation path
		p = Path(self.jp)
		self.img_dir = img_dir if img_dir is not None else f'{p.parent.parent}/{str(p.stem).split("_")[-1]}'
		self.line_thickness = line_thickness

	def __call__(self, only_show_id=None):
		print(f'Loading annotations...')
		with open(self.jp, 'r') as f:
			annoj = json.load(f)

		print(f'\n{"":=<30} Dataset Info {"":=<30}')
		print(f'The annotation_json contains {len(annoj)} keys: {list(annoj.keys())}')  # 统计json文件的关键字长度
		# info
		metadata = annoj['metadata']
		print(f'Dataset info: {metadata}')
		licenses = annoj['licenses']
		print(f'Dataset license: {licenses[0]}')

		# 关于类别
		classes = annoj['categories']
		classes = {x['id']: x['name'] for x in classes}
		print(f'The dataset contains {len(classes)} classes: {classes}')
		colors = ColorControl(num=len(classes), rgb='BGR').color

		# 关于images
		images = annoj['images']
		print(f'The dataset contains {len(images)} images. ', end='')  # json文件中包含的图片数量
		print(f'Image example: {images[0]}')

		# 关于annos
		annos = annoj['annotations']
		print(f'The dataset contains {len(annos)} targets. ', end='')
		print(f'Target example: {annos[0]}')

		print(f'\n{"":=<30}  Image Info  {"":=<30}')
		# 可视化图像
		annos_img_ids = [x['image_id'] for x in annos]
		# print(annos_img_ids, len(annos_img_ids))
		for i, img_info in enumerate(images):
			img_name = img_info['file_name']
			img_id = img_info['id']
			height, width = img_info['height'], img_info['width']
			date_captured = img_info['date_captured']

			# 找到对应图像
			img_path = f'{self.img_dir}/{img_name}'
			assert os.path.exists(img_path), f'img {img_path} does not exists.'
			img = cv2.imread(img_path)
			h, w, c = img.shape
			assert height == h and width == w, f'The shape of image: {h}*{w} is inconsistent with that from ' \
											   f'annotation: {height}*{width}. img_path: {img_path}'

			# 找对应的标注
			# anno_idx = annos_img_ids.index(img_id)  # 这是错的，一张图可能对应多个annos，这种方法只会找到一个
			anno_idxes = [j for j, x in enumerate(annos_img_ids) if img_id == x]

			if len(anno_idxes) == 0:
				print(f'Image {img_name} without targets.')

			img_annos = [annos[idx] for idx in anno_idxes]
			num_t = len(img_annos)
			# print(img_name, img_annos)

			for anno in img_annos:
				seg = anno['segmentation']
				bbox = anno['bbox']  # x1 y1 w h, in pixels
				bbox[2] += bbox[0]
				bbox[3] += bbox[1]  # 转为x1 y1 x2 y2
				anno_img_id = anno['image_id']
				cls = anno['category_id']  # 1 2 3 ...
				cls_name = classes[cls]
				area = anno['area']
				# print(anno)

				color_idx = list(classes.values()).index(cls_name)
				# print(f'color: {}')
				img = general.plot_one_box_trace_pose_status(
					bbox, img, color=colors[color_idx], label=cls_name,
					line_thickness=self.line_thickness)

			print(f'img_id: {img_id:<6} img_name: {img_name} num_targets: {num_t}')

			if only_show_id is None or str(img_id) == str(only_show_id):
				cv2.imshow(f'{img_name}', img)
				if cv2.waitKey(0) == ord('q'):
					break
				cv2.destroyAllWindows()
from dataclasses import dataclass

@dataclass
class Args:
    json_path: str = '/data/tml/lymonet/lymo_coco/annotations/instances_train.json'
    img_dir: str = '/data/tml/lymonet/lymo_coco/train'
    only_show_id: str = None
	


if __name__ == "__main__":
    
    import hai
    args = hai.parse_args(Args)

    cc = CheckCOCO(json_path=args.json_path, img_dir=args.img_dir)
    cc(only_show_id=args.only_show_id)