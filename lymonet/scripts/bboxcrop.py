import os
import cv2
import shutil

# 数据集路径
dataset_path = '/data/tml/lymonet/lymo_yolo_unclipped'

# 遍历train、val和test三个文件夹
split_folders = ['train', 'val', 'test']
for split_folder in split_folders:
    images_folder = os.path.join(dataset_path, 'images', split_folder)
    labels_folder = os.path.join(dataset_path, 'labels', split_folder)
    output_images_folder = os.path.join("/data/tml/lymonet/lymo_yolo_bboxclip1", 'images', split_folder)
    output_labels_folder = os.path.join("/data/tml/lymonet/lymo_yolo_bboxclip1", 'labels', split_folder)
    classified_images_folder = os.path.join("/data/tml/lymonet/lymo_yolo_class1", split_folder)

    # 创建输出文件夹
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)
    os.makedirs(classified_images_folder, exist_ok=True)

    # 遍历labels文件夹中的文件
    for label_file in os.listdir(labels_folder):
        with open(os.path.join(labels_folder, label_file), 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                # 从标注文件中解析出边界框信息
                class_id, x_center, y_center, width, height = map(float, line.split())
                
                # 读取对应的图像文件
                image_file = os.path.join(images_folder, label_file.replace('txt', 'png'))
                img = cv2.imread(image_file)
                h, w, _ = img.shape
                
                # 计算裁剪框的大小为目标的1倍
                ratio = 1
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                new_width = int((x2 - x1) * ratio)
                new_height = int((y2 - y1) * ratio)
                new_x1 = max(0, center_x - new_width // 2)
                new_y1 = max(0, center_y - new_height // 2)
                new_x2 = min(w, center_x + new_width // 2)
                new_y2 = min(h, center_y + new_height // 2)
                
                # 裁剪图像并保存
                cropped_img = img[new_y1:new_y2, new_x1:new_x2]
                output_image_file = os.path.join(output_images_folder, label_file.replace('txt', f'_{idx}.jpg'))
                output_label_file = os.path.join(output_labels_folder, label_file.replace('txt', f'_{idx}.txt'))
                cv2.imwrite(output_image_file, cropped_img)
                # shutil.copy(os.path.join(labels_folder, label_file), output_label_file)
                # 更新标签文件中的边界框信息
                with open(output_label_file, 'w') as out_file:
                    new_x_center = (x_center - (new_x1/w)) / ((new_x2 - new_x1)/w)
                    new_y_center = (y_center - (new_y1/h)) / ((new_y2 - new_y1)/h)
                    new_width = width / ((new_x2 - new_x1)/w)
                    new_height = height / ((new_y2 - new_y1)/h)
                    out_file.write(f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}\n")
                # 获取类别名
                class_name = f'class_{int(class_id)}'
                
                # 创建目标文件夹
                target_folder = os.path.join(classified_images_folder, class_name)
                os.makedirs(target_folder, exist_ok=True)
                
                # 移动图像文件到目标文件夹
                shutil.copy(output_image_file, os.path.join(target_folder, label_file.replace('txt', f'_{idx}.jpg')))

