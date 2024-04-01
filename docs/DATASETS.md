# Dataset

The lymph node dataset contains 3090 images for train, 685 images for validation and 737 images for test.

Here, we provide validation data.

## Validation data
+ Download from [Baidu Pan](https://pan.baidu.com/s/1uUvv-28HP99pHenotgaqOQ), passwd: **w1up**
+ Download from [Google Drive](https://drive.google.com/file/d/1UXSmwmgd65vQqVCBTQ9QFZQExheKMTCg/view?usp=share_link).

**Notes**

Please remove proxy if download failed.


The file structure format is YOLOv8 likes:
```
|--data_v8_format
--|--images  # the .jpg images
   --|--val  # trainning images
      -- xxx.png
      -- ...
--|--labels  # corresponding .txt labels
   --|--val  # trainning labels
      -- xxx.txt
      -- ...
```
Each .txt file contains annotations in the format of CLS XC YC W H in each line. 

CLS(Classes): 0 metastatic, 1 inflammatory, 3 normal

XC YC W H in terms of percentage.

# Trained weight

The improved SOTA trained weight is provided.
+ Download from [Baidu Pan](https://pan.baidu.com/s/1FIkvvi19JxOAKioj7AIP6w), passwd: **hbto**
+ Download from [Google Drive](https://drive.google.com/file/d/1663fu4vjoXhvN_2rzh2z7mONcgYPx93n/view?usp=share_link).


