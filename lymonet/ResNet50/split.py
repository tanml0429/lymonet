import os, sys
from pathlib import Path
import shutil
import random

class SplitDatasets:
    def __init__(self) -> None:
        pass
    def __call__(self, sp, tp, trte):
        # 创建文件夹        
        if not os.path.exists(f'{tp}'):
            os.makedirs(f'{tp}')

        if not os.path.exists(f'{tp}/images/{trte}'):
            os.makedirs(f'{tp}/images/{trte}')
            os.makedirs(f'{tp}/labels/{trte}')
            # 清空文件夹
        for root, dirs, files in os.walk(f'{tp}/images/{trte}'):
            for file in files:
                os.remove(os.path.join(root, file))
        for root, dirs, files in os.walk(f'{tp}/labels/{trte}'):
            for file in files:
                os.remove(os.path.join(root, file))

        # 划分训练集和测试集
    def split_train_test(self, sp, ratio=0.8):
        # 获取图片文件名
        img_paths = [os.path.join(root, f) for root, dirs, files in os.walk(f'{sp}/images') for f in files if f.endswith('jpg')]
        img_stems = [Path(x).stem for x in img_paths]
        

        # 划分训练集和测试集
        img_stems = sorted(img_stems)
        # 打乱顺序
        random.shuffle(img_stems)
        num = len(img_stems)
        train_num = int(num * ratio)
        return train_num, img_stems

    def copy_files(self, sp, tp, trte, train_num, img_stems):
        if trte == 'train':
            img_stems = img_stems[:train_num]
        else:
            img_stems = img_stems[train_num:]
        print (f'{trte} num: {len(img_stems)}')
        # 复制文件
        for stem in img_stems:
            shutil.copy(f'{sp}/images/{stem}.jpg', f'{tp}/images/{trte}/{stem}.jpg')
            shutil.copy(f'{sp}/labels/{stem}.txt', f'{tp}/labels/{trte}/{stem}.txt')

            

           
    

if __name__ == "__main__":
    sp = '/data/tml/splitdone_polyp/normal_endoscopy'
    tp = '/data/tml/splitdone_polyp_v5_format/normal_endoscopy'
    split_datasets = SplitDatasets()
    split_datasets(sp, tp, 'train')
    split_datasets(sp, tp, 'test')
    train_num, img_items = split_datasets.split_train_test(sp)
    split_datasets.copy_files(sp, tp, 'train', train_num, img_items)
    split_datasets.copy_files(sp, tp, 'test', train_num, img_items)


            