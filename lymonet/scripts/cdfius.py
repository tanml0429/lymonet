import os
import shutil

# 定义原始数据集文件夹路径
dataset_path = '/data/tml/lymonet/lymo_yolo_unclipped'

# 定义两个新数据集的文件夹路径
output_dataset_cdfi = '/data/tml/lymonet/lymo_yolo_cdfi'
output_dataset_us = '/data/tml/lymonet/lymo_yolo_us'

# # 如果两个新数据集文件夹不存在，则创建它们
# if not os.path.exists(output_dataset_cdfi):
#     os.makedirs(output_dataset_cdfi)
#     os.makedirs(os.path.join(output_dataset_cdfi, 'images', 'test'))
#     os.makedirs(os.path.join(output_dataset_cdfi, 'images', 'val'))
#     os.makedirs(os.path.join(output_dataset_cdfi, 'images', 'train'))
#     os.makedirs(os.path.join(output_dataset_cdfi, 'labels', 'test'))
#     os.makedirs(os.path.join(output_dataset_cdfi, 'labels', 'val'))
#     os.makedirs(os.path.join(output_dataset_cdfi, 'labels', 'train'))

# if not os.path.exists(output_dataset_us):
#     os.makedirs(output_dataset_us)
#     os.makedirs(os.path.join(output_dataset_us, 'images', 'test'))
#     os.makedirs(os.path.join(output_dataset_us, 'images', 'val'))
#     os.makedirs(os.path.join(output_dataset_us, 'images', 'train'))
#     os.makedirs(os.path.join(output_dataset_us, 'labels', 'test'))
#     os.makedirs(os.path.join(output_dataset_us, 'labels', 'val'))
#     os.makedirs(os.path.join(output_dataset_us, 'labels', 'train'))

# # 遍历test、val和train文件夹
# for split in ['test', 'val', 'train']:
#     images_folder = os.path.join(dataset_path, 'images', split)
#     labels_folder = os.path.join(dataset_path, 'labels', split)

#     for folder in [images_folder, labels_folder]:
#         files = os.listdir(folder)
#         for file in files:
#             if "CDFI" in file:
#                 # 如果文件名中包含"CDFI"，则将文件复制到output_dataset_cdfi文件夹中
#                 shutil.copy(os.path.join(folder, file), os.path.join(output_dataset_cdfi,folder.split('/')[-2], split))
#             elif "US" in file:
#                 # 如果文件名中包含"US"，则将文件复制到output_dataset_us文件夹中
#                 shutil.copy(os.path.join(folder, file), os.path.join(output_dataset_us, folder.split('/')[-2], split))

# 输出数量
print('CDFI train images: ', len(os.listdir(os.path.join(output_dataset_cdfi, 'images', 'train'))))

print('US train images: ', len(os.listdir(os.path.join(output_dataset_us, 'images', 'train'))))

print('CDFI val images: ', len(os.listdir(os.path.join(output_dataset_cdfi, 'images', 'val'))))

print('US val images: ', len(os.listdir(os.path.join(output_dataset_us, 'images', 'val'))))

print('CDFI test images: ', len(os.listdir(os.path.join(output_dataset_cdfi, 'images', 'test'))))

print('US test images: ', len(os.listdir(os.path.join(output_dataset_us, 'images', 'test'))))


