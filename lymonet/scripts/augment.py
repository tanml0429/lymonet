from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import numpy as np

# 设置数据增强参数
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant')

# 定义函数用于对文件夹中的图片进行增强并保存到新的文件夹
def augment_images_in_folder_and_save(folder_path, save_path, class_name, max_samples):
    files = os.listdir(folder_path)
    num_samples = len(files)
    times = max_samples // num_samples - 1
    for file in files:
        img_path = os.path.join(folder_path, file)
        img = image.load_img(img_path)  # 读取图像
        img = np.expand_dims(img, axis=0)
        # img = img.reshape((1,) + img.shape)  # 将图像转换为4D张量
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=save_path, save_prefix=class_name, save_format='jpg'):
            i += 1
            if i > times:
                break

# 假设三类样本的路径分别为 class_A_path, class_B_path, class_C_path
train_path = '/data/tml/lymonet/lymo_yolo_square1/train'
test_path = '/data/tml/lymonet/lymo_yolo_square1/test'
val_path = '/data/tml/lymonet/lymo_yolo_square1/val'

# 创建新的文件夹用于保存增强后的图片
augmented_train_path = '/data/tml/lymonet/lymo_yolo_aug1/train'
augmented_test_path = '/data/tml/lymonet/lymo_yolo_aug1/test'
augmented_val_path = '/data/tml/lymonet/lymo_yolo_aug1/val'

# 获取每个文件夹中样本数量最多的类别的数量
max_samples_train = max(len(os.listdir(os.path.join(train_path, 'inflammatory'))), len(os.listdir(os.path.join(train_path, 'metastatic'))), len(os.listdir(os.path.join(train_path, 'normal'))))
max_samples_test = max(len(os.listdir(os.path.join(test_path, 'inflammatory'))), len(os.listdir(os.path.join(test_path, 'metastatic'))), len(os.listdir(os.path.join(test_path, 'normal'))))
max_samples_val = max(len(os.listdir(os.path.join(val_path, 'inflammatory'))), len(os.listdir(os.path.join(val_path, 'metastatic'))), len(os.listdir(os.path.join(val_path, 'normal'))))

# 对每个文件夹中的每个类别进行数据增强并保存到新的文件夹
for folder_path, save_path, max_samples in zip([train_path, test_path, val_path], [augmented_train_path, augmented_test_path, augmented_val_path], [max_samples_train, max_samples_test, max_samples_val]):
    for class_name in ['inflammatory', 'metastatic', 'normal']:
        class_path = os.path.join(folder_path, class_name)
        save_class_path = os.path.join(save_path, class_name)
        os.makedirs(save_class_path, exist_ok=True)  # 如果保存文件夹不存在则创建
        augment_images_in_folder_and_save(class_path, save_class_path, class_name, max_samples)


# 统计数量
print('train inflammatory: ', len(os.listdir(os.path.join(augmented_train_path, 'inflammatory'))))
print('train metastatic: ', len(os.listdir(os.path.join(augmented_train_path, 'metastatic'))))
print('train normal: ', len(os.listdir(os.path.join(augmented_train_path, 'normal'))))
print('test inflammatory: ', len(os.listdir(os.path.join(augmented_test_path, 'inflammatory'))))
print('test metastatic: ', len(os.listdir(os.path.join(augmented_test_path, 'metastatic'))))
print('test normal: ', len(os.listdir(os.path.join(augmented_test_path, 'normal'))))
print('val inflammatory: ', len(os.listdir(os.path.join(augmented_val_path, 'inflammatory'))))
print('val metastatic: ', len(os.listdir(os.path.join(augmented_val_path, 'metastatic'))))
print('val normal: ', len(os.listdir(os.path.join(augmented_val_path, 'normal'))))