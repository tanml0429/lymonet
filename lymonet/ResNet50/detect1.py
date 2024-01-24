import os
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
from easydict import EasyDict
from PIL import Image
from utils import datasets, detect_utils, torch_utils, general
import torch
from models import resnet_zoo, resnet_zoo_half, resnet_zoo_behavior
import yaml
import glob
import tqdm

def detect(opt):
    # 设置
	cuda = opt.device != 'cpu'
	device_id = [int(x) for x in opt.device.split(',')] if cuda else None
	torch_device = torch_utils.select_device(device=opt.device, batch_size=opt.batch_size)

	# 加载模型
	classes = opt.cfg['statuses']
	nc = len(classes)
	# target_paths = glob.glob(f'{opt.source}/**')
	# test_paths = [os.path.join(root, f) for root, dirs, files in os.walk(f'{opt.source}') for f in files if f.endswith(f'.{opt.suffix}')]
	# test_paths = [f'{opt.source}/{x}' for x in os.listdir(opt.source) if os.path.isdir(f'{opt.source}/{x}')]
	# test_loader = datasets.StatusDataLoader(balance=False, dformat='behavior').get_loader(
	# 	trte_path=test_paths, batch_size=opt.batch_size, cfg=opt.cfg)
	model = resnet_zoo_behavior.attempt_load(model_name='resnet50', pretrained=False, num_classes=nc)  # 加载模型结构


	# optimizer = torch.optim.SGD(model.parameters(), hyp['lr0'], momentum=hyp['momentum'],
	# 							weight_decay=hyp['weight_decay'])
	optimizer = torch.optim.SGD(
		model.parameters(), opt.cfg['lr0'], momentum=opt.cfg['momentum'], weight_decay=opt.cfg['weight_decay'])
	# 加载模型参数
	model, optimizer, start_epoch = general.load_resume2(
		opt.weights, model, optimizer)

	# print(model)

	if cuda:
		model = model.to(torch_device)
		if torch.cuda.device_count() > 1:
			model = torch.nn.DataParallel(model, device_ids=device_id)

	# 加载数据
	if isinstance(opt.source, str):
		sp = opt.source  # source path
		test_paths = []
		for root, dirs, files in os.walk(sp):
			for f in files:
				if f.endswith(opt.suffix):
					test_paths.append(os.path.join(root, f))
		# test_paths = [os.path.join(root, f) for root, dirs, files in os.walk(f'{opt.source}') for f in files if f.endswith(f'.{opt.suffix}')]
		print (len(test_paths))
		# print (opt.source)
		# print (opt.suffix)
		stems = [Path(x).stem for x in test_paths] # 图片名字
		# stems = [Path(x).stem for x in os.listdir(sp) if x.endswith(opt.suffix)] # 图片名字
	else:
		stems = opt.source

	tp = opt.output  # target path
	if os.path.exists(tp):
		shutil.rmtree(tp)
	os.makedirs(tp)
	print (len(stems))
	assert len(stems) > 0
	stems = sorted(stems)
	model.eval() # 设置为评估模式
	stt = general.synchronize_time()
	for i, img_path in enumerate(test_paths):
		img = cv2.imread(img_path)
		img = np.array(img, dtype=np.float32)
		img = np.transpose(img/255.0, (2, 0, 1))
		stem = Path(img_path).stem
		print(f'\r{i}/{len(stems)}: {stem}', end='')

	# for i, stem in enumerate(stems):
	# 	img_path = f'{sp}/{stem}{opt.suffix}' if isinstance(opt.source, str) else stem
	# 	img = cv2.imread(img_path)
	# 	img = np.array(img, dtype=np.float32)
	# 	img = np.transpose(img/255.0, (2, 0, 1))  # [3, 256, 256]

		x = torch.from_numpy(img).unsqueeze(0)  # [1, 3, 256, 256]
		x = x.to(torch_device)
		start_time = general.synchronize_time()
		output = model(x)
		output = output.cpu().detach().numpy()
		label = np.argmax(output)
		inference_time = general.synchronize_time() - start_time
		output_dir = f'{tp}/{classes[label]}'
		label_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
			os.makedirs(f'{output_dir}/images')
			os.makedirs(f'{output_dir}/labels')
		shutil.copy(img_path, f'{output_dir}/images/{stem}{opt.suffix}')
		shutil.copy(label_path, f'{output_dir}/labels/{stem}.txt')
	print()

		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--source', type=str, default='', help='source path')
	parser.add_argument('--cfg_file', type=str, default='/home/tml/VSProjects/polyp_mixed/src/ResNet50/data/cfg.yaml', help='hyperparamers path')
	parser.add_argument('--batch_size', type=int, default=16, help='total batch size')
	# parser.add_argument('--resume', default='runs/exp0/weights/last.pt', help='resume from given path/last.pt')
	parser.add_argument('--weights', default=None, help='resume from given path/last.pt')
	parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0,1,2,3 or cpu')
	parser.add_argument('--save', type=bool, default=True, help='save img')
	opt = parser.parse_args()

	opt.weights = '/home/tml/VSProjects/polyp_mixed/runs/exp25/weights/best.pt'
	opt.output = '/data/tml/splitdone_polyp'
	opt.source = '/data/tml/hybrid_polyp_v5_format'
	opt.suffix = '.jpg'

	with open(opt.cfg_file, 'r') as f:
		opt.cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
	detect(opt)
		
