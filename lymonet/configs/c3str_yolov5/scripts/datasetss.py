"""
-*- coding: utf-8 -*-
@Author  : EricZ
@Time    : 2022/7/22 12:23
@Software: PyCharm
@File    : datasetss.py
"""
import glob
import json
import os
import shutil

import numpy as np
from PIL import Image
import os.path as osp
import re
import cv2
import tqdm
import base64


def draw(img, labels):
    height, width = img.shape[:2]
    for label in labels:
        x = float(label.split(' ')[1])
        y = float(label.split(' ')[2])
        w = float(label.split(' ')[3])
        h = float(label.split(' ')[4])
        xmin = int((x - w / 2) * width)
        ymin = int((y - h / 2) * height)
        xmax = int((x + w / 2) * width)
        ymax = int((y + h / 2) * height)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), 255, 2)
    return img


root_dir = r'e:/dataset/detection/lymph/dataset'
label_CDFI = osp.join(root_dir, 'labels_CDFI/patient0000/patient0000_node_1_CDFI_1470.txt')
label_US = osp.join(root_dir, 'labels_US/patient0000/patient0000_node_1_US_1375.txt')
with open(label_CDFI) as f:
    CDFI = f.readlines()
with open(label_US) as f:
    US = f.readlines()
img_CDFI = cv2.imread(osp.join(root_dir, 'images_CDFI/patient0000/patient0000_node_1_CDFI_1470.jpg'))
img_US = cv2.imread(osp.join(root_dir, 'images_US/patient0000/patient0000_node_1_US_1375.jpg'))
img_US = draw(img_US, US)
img_CDFI = draw(img_CDFI, CDFI)
cv2.imshow('CDFI', img_CDFI)
cv2.imshow('US', img_US)
cv2.waitKey()