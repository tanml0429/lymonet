
"""
转换训练集，YOLOv5 format to COCO format
"""
import os, sys
import argparse
from pathlib import Path
import json
import shutil
import cv2
from tqdm import tqdm
import damei as dm


class YOLO2COCO(object):
	def __init__(self, source_path, target_path=None):
		self.sp = source_path  # source path
		self.tp = target_path if target_path else f'{Path(source_path).parent}/transformed_coco_format'

		# 检查输出文件夹
		if os.path.exists(self.tp):
			ipt = input(f'Target path: {self.tp} exists, \nremove and continue? [YES/no]: ')
			if ipt in ['Y', '', 'y', 'yes', 'YES']:
				shutil.rmtree(self.tp)
			else:
				print('Exit')
				exit()
		os.makedirs(self.tp)
		os.makedirs(f'{self.tp}/annotations')
		

		# 关于coco格式
		self.annotation_id = 1
		self.type = 'instances'
		self.categories = self.read_categories()
		self.info = {
			'year': 2023,
			'version': '1.0.0',
			'description': 'For object detection',
			'date_created': dm.current_time(),
		}
		self.licenses = [{
			'id': 1,
			'name': 'GNU General Public License v3.0',
			'url': '',
		}]

		# 支持的图像格式
		self.suffix = ['.jpg', '.bmp', '.png']

	def read_categories(self):
		file = f'{self.sp}/label.txt'
		print(self.sp)
		assert os.path.exists(file), f'classes file {file} does not exists.'
		with open(file, 'r') as f:
			data = f.readlines()
		data = [x.replace('\n', '') for x in data]
		categories = []
		for i, cls_name in enumerate(data):
			cls = i
			cls_name = cls_name.split()[1]
			categories.append({
				'id': cls,
				'name': cls_name,
				'supercategory': cls_name,
			})
		return categories

	def __call__(self, *args, **kwargs):
		sp = self.sp
		trtevals = [x for x in os.listdir(f'{sp}/images') if os.path.isdir(f'{sp}/images/{x}')]
		for trte in trtevals:
			print(f'Deal with {trte}')
			imgs = os.listdir(f'{sp}/images/{trte}')
			imgs = [f'{sp}/images/{trte}/{x}' for x in imgs if str(Path(x).suffix) in self.suffix]

			# trval = trte if trte == 'train' else 'val'
			os.makedirs(f'{self.tp}/{trte}')
			self.deal_single(img_paths=imgs, trval=trte)

	def deal_single(self, img_paths, trval):
		tp = self.tp
		bar = tqdm(img_paths)

		images = []
		annotations = []
		for i, img_path in enumerate(bar):
			img_id = i + 1
			stem = Path(img_path).stem
			txt_path = f'{str(Path(img_path).parent).replace("images", "labels")}/{stem}.txt'

			img = cv2.imread(img_path)
			h, w, c = img.shape

			new_stem = stem

			# 保存图像
			if Path(img_path).suffix.lower() == '.jpg':
				shutil.copyfile(img_path, f'{tp}/{trval}/{new_stem}.jpg')
			else:
				cv2.imwrite(f'{tp}/{trval}/{new_stem}.jpg', img=img)

			# 保存标注
			images.append({
				'id': img_id,
				'width': w,
				'height': h,
				'file_name': f'{new_stem}.jpg',
    			'date_captured': '2023',
			})

			annot = self.label2annot(txt_path, h, w, img_id=img_id)
			assert len(annot) > 0, f'{txt_path} is empty.'
			annotations.extend(annot)

		json_data = {
			'metadata': self.info,
			'images': images,
			'annotations': annotations,
			'categories': self.categories,
			'type': self.type,
			'licenses': self.licenses,
		}

		tp_json = f'{tp}/annotations/instances_{trval}.json'
		with open(tp_json, 'w', encoding='utf-8') as f:
			json.dump(json_data, f, indent=4, ensure_ascii=False)

	def label2annot(self, txt_path, h, w, img_id):
		annotation = []
		with open(txt_path, 'r') as f:
			labels = f.readlines()
		labels = [x.replace('\n', '') for x in labels]
		for lb in labels:
			cls = lb.split()[0]
			bbox = lb.split()[1::]  # xc yc w h in fraction
			assert len(bbox) == 4
			segmentation, bbox, area = self._get_annotation(bbox, h, w)  # 转为x1y1wh了
			annotation.append({
				'id': self.annotation_id,
    			'image_id': img_id,
       			'category_id': int(cls),
				'segmentation': segmentation,
				'area': area,
    			'bbox': bbox,
				'iscrowd': 0,
			})
			self.annotation_id += 1
		return annotation

	@staticmethod
	def _get_annotation(vertex_info, height, width):
		cx, cy, w, h = [float(i) for i in vertex_info]

		cx = cx * width
		cy = cy * height
		box_w = w * width
		box_h = h * height

		# left top
		x0 = max(cx - box_w / 2, 0)
		y0 = max(cy - box_h / 2, 0)

		# right bottomt
		x1 = min(x0 + box_w, width)
		y1 = min(y0 + box_h, height)

		segmentation = [[x0, y0, x0, y1, x1, y1, x1, y0]]
		bbox = [x0, y0, box_w, box_h]
		area = box_w * box_h
		return segmentation, bbox, area
	





if __name__ == '__main__':
	sp = f'/data/tml/lymonet/lymo_yolo_unclipped'
	tp = f'/data/tml/lymonet/lymo_coco_unclipped'
	y2c = YOLO2COCO(sp, tp)
	y2c()
