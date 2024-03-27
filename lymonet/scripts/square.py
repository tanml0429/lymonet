import cv2
import os
import numpy as np

def letterbox_image(image, size):
    ih, iw = image.shape[0:2]
    h, w = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.ones((h, w, 3), dtype=np.uint8) * 128 # 128 for gray
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image[dy:dy+nh, dx:dx+nw] = image_resized
    return new_image


if __name__ == '__main__':
    original_dataset_path = '/data/tml/lymonet/lymo_yolo_class1'
    new_dataset_path = '/data/tml/lymonet/lymo_yolo_square1'
    target_size = (640, 640)

    # 遍历原始数据集的每个一级文件夹
    for folder1 in os.listdir(original_dataset_path):
        folder1_path = os.path.join(original_dataset_path, folder1)
        
        # 创建对应的新一级文件夹
        new_folder1_path = os.path.join(new_dataset_path, folder1)
        os.makedirs(new_folder1_path, exist_ok=True)
        
        # 遍历每个一级文件夹中的二级分类文件夹
        for folder2 in os.listdir(folder1_path):
            folder2_path = os.path.join(folder1_path, folder2)
            
            # 创建对应的新二级分类文件夹
            new_folder2_path = os.path.join(new_folder1_path, folder2)
            os.makedirs(new_folder2_path, exist_ok=True)
            
            # 遍历二级分类文件夹中的图像文件
            for file in os.listdir(folder2_path):
                file_path = os.path.join(folder2_path, file)
                
                # 读取图像
                image = cv2.imread(file_path)
                
                # 使用letterbox算法转换为正方形
                letterboxed_image = letterbox_image(image, target_size)
                
                # 保存处理后的图像到新数据集文件夹中
                new_file_path = os.path.join(new_folder2_path, file)
                cv2.imwrite(new_file_path, letterboxed_image)
