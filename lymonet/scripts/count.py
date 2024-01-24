import os

# 数据集路径
dataset_path = '/data/tml/lymonet/lymo_yolo_bboxclip1'

# 遍历train、val和test三个文件夹
split_folders = ['train', 'val', 'test']
for split_folder in split_folders:
    images_folder = os.path.join(dataset_path, 'images', split_folder)
    labels_folder = os.path.join(dataset_path, 'labels', split_folder)
    
    # 统计文件数量
    num_images = len(os.listdir(images_folder))
    num_labels = len(os.listdir(labels_folder))
    
    print(f"Split: {split_folder}, Number of images: {num_images}, Number of labels: {num_labels}")
